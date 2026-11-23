"""Entry point for the forecasting service (forecast-api) with structured events, Prometheus metrics, and RAG explanations."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
import httpx
import mlflow
from mlflow import MlflowClient
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

from .schemas import PredictRequest, PredictResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("forecast-api")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")

KB_API_URL = os.getenv("KB_API_URL", "http://kb-api:8002")
LLM_API_URL = os.getenv("LLM_API_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

NUM = ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "OverallQual"]
CAT = ["MSZoning"]


def emit_event(level: str, event: str, **fields) -> None:
    """Виведення однорядкової структурованої JSON-події у stdout."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(record), flush=True)


def build_rag_query(req: PredictRequest) -> str:
    """Формування семантичного запиту на основі ознак об'єкта нерухомості."""
    return (
        f"Property valuation factors for above-grade living area {req.gr_liv_area} sq ft, "
        f"overall quality {req.overall_qual}/10, basement {req.total_bsmt_sf} sq ft, "
        f"garage {req.garage_area} sq ft, year built {req.year_built}, "
        f"zoning {req.ms_zoning}."
    )


def build_grounded_prompt(req: PredictRequest, price: float, passages: list[dict]) -> str:
    """Побудова промпта з вимогою опори виключно на наданий контекст."""
    ctx_lines = [
        f"[{idx}] ({p['source']}, {p['section']}, page {p['page']}) {p['content']}"
        for idx, p in enumerate(passages, start=1)
    ]
    context_block = "\n".join(ctx_lines) if ctx_lines else "No context retrieved."
    return (
        "You are an assistant supporting an operator of a property valuation system.\n"
        "Answer ONLY on the basis of the context below. If the context does not contain "
        "the answer, state this explicitly. After each statement, give the source in square brackets.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Data: Gr Liv Area={req.gr_liv_area} sq ft, Overall Qual={req.overall_qual}, "
        f"Total Bsmt SF={req.total_bsmt_sf} sq ft, Garage Area={req.garage_area} sq ft, "
        f"Year Built={req.year_built}, MS Zoning={req.ms_zoning}, "
        f"predicted price={price:.2f} USD.\n"
        "Question: What domain factors in the documentation correspond to this valuation?"
    )


def generate_local_grounded_answer(
    req: PredictRequest,
    price: float,
    passages: list[dict],
) -> str:
    """Детермінований grounded-генератор із цитатами (працює локально без зовнішнього API-ключа)."""
    if not passages:
        return "The retrieved context does not contain sufficient information to explain this valuation."

    statements = [
        f"The production model (@prod) estimated the property value at {price:,.2f} USD "
        f"(Gr Liv Area: {req.gr_liv_area} sq ft, Overall Qual: {req.overall_qual}/10, "
        f"Year Built: {req.year_built}, MS Zoning: {req.ms_zoning})."
    ]
    for idx, p in enumerate(passages, start=1):
        statements.append(f"According to reference documentation, {p['content']} [{idx}]")
    return " ".join(statements)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Завантаження моделі та описової статистики з MLflow Model Registry за аліасом при старті."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        app.state.model = mlflow.pyfunc.load_model(model_uri)
        app.state.model_version = str(mv.version)

        stats_path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="data_stats.json"
        )
        app.state.data_stats = json.loads(Path(stats_path).read_text())
        logger.info(
            "Loaded model '%s@%s' (version=%s, run_id=%s)",
            MODEL_NAME,
            MODEL_ALIAS,
            app.state.model_version,
            mv.run_id,
        )
    except Exception as exc:
        logger.error(
            "CRITICAL: Failed to resolve or load model '%s@%s' from %s: %s",
            MODEL_NAME,
            MODEL_ALIAS,
            MLFLOW_TRACKING_URI,
            exc,
        )
        sys.exit(1)

    yield
    app.state.model = None


app = FastAPI(
    title="Forecast API",
    version="0.6.0",
    lifespan=lifespan,
)

# Експозиція /metrics на forecast-api
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": str(app.state.model_version),
    }


@app.get("/model/data-stats")
def get_model_data_stats():
    return app.state.data_stats


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    x_order_id: str | None = Header(default=None, alias="X-Order-Id"),
):
    t0 = time.perf_counter()

    df = pd.DataFrame([{
        "GrLivArea": float(req.gr_liv_area),
        "TotalBsmtSF": float(req.total_bsmt_sf),
        "GarageArea": float(req.garage_area),
        "YearBuilt": float(req.year_built),
        "OverallQual": float(req.overall_qual),
        "MSZoning": str(req.ms_zoning),
    }])

    price = round(float(app.state.model.predict(df[NUM + CAT])[0]), 2)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Структурована JSON-подія на кожен виклик /predict
    extra = {}
    if x_order_id is not None:
        try:
            extra["order_id"] = int(x_order_id)
        except ValueError:
            extra["order_id"] = x_order_id

    emit_event(
        "info",
        "predict",
        **extra,
        predicted_price=price,
        latency_ms=latency_ms,
        model_version=str(app.state.model_version),
    )

    return PredictResponse(
        predicted_price=price,
        model_version=str(app.state.model_version),
        latency_ms=latency_ms,
    )


@app.post("/explain")
async def explain(
    req: PredictRequest,
    x_order_id: str | None = Header(default=None, alias="X-Order-Id"),
):
    """Отримання ціни від @prod та генерація document-grounded пояснення через RAG."""
    t0 = time.perf_counter()

    # 1. Отримуємо прогноз від існуючого обробника /predict (модель @prod не змінюється)
    pred = predict(req, x_order_id=x_order_id)
    predicted_price = float(pred.predicted_price)
    model_version = str(pred.model_version)

    # 2. Формуємо запит і отримуємо top-3 фрагменти з kb-api
    query_text = build_rag_query(req)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            kb_resp = await client.post(
                f"{KB_API_URL}/search",
                json={"query": query_text, "top_k": 3},
            )
            kb_resp.raise_for_status()
            kb_data = kb_resp.json()
    except httpx.HTTPError as exc:
        emit_event("error", "explain_failed", error=str(exc))
        raise HTTPException(
            status_code=502,
            detail=f"Knowledge base service unavailable: {exc}",
        )

    passages = kb_data.get("results", [])
    prompt = build_grounded_prompt(req, predicted_price, passages)

    # 3. Генерація відповіді: через OpenAI-сумісний API (якщо задано ключ) або локально
    generator_mode = "local-grounded-template"
    explanation_text = generate_local_grounded_answer(req, predicted_price, passages)

    if LLM_API_URL and LLM_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                llm_resp = await client.post(
                    f"{LLM_API_URL.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                    },
                )
                llm_resp.raise_for_status()
                explanation_text = llm_resp.json()["choices"][0]["message"]["content"]
                generator_mode = f"llm:{LLM_MODEL}"
        except Exception:
            generator_mode = "local-grounded-fallback"

    citations = [
        {
            "ref": f"[{idx}]",
            "source": p["source"],
            "section": p["section"],
            "page": p["page"],
            "cosine_similarity": p["cosine_similarity"],
            "excerpt": p["content"],
        }
        for idx, p in enumerate(passages, start=1)
    ]

    extra = {}
    if x_order_id is not None:
        try:
            extra["order_id"] = int(x_order_id)
        except ValueError:
            extra["order_id"] = x_order_id

    emit_event(
        "info",
        "explain",
        **extra,
        model_version=model_version,
        citations_count=len(citations),
        generator_mode=generator_mode,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

    return {
        "predicted_price": predicted_price,
        "model_version": model_version,
        "query": query_text,
        "explanation": explanation_text,
        "citations": citations,
        "generator_mode": generator_mode,
        "disclaimer": (
            "This explanation provides document-grounded domain context and does not "
            "represent internal causal weights or SHAP attributions of the @prod model."
        ),
    }