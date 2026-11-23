import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import httpx
import jwt
from passlib.context import CryptContext
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from . import crud, models, schemas
from .database import Base, SessionLocal, engine, get_db

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("orders-api")


def emit_event(level: str, event: str, **fields) -> None:
    """Emit a single-line JSON structured event to stdout (R1)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(record), flush=True)


# --- DOMAIN METRICS (R3) ---
ORDERS_TOTAL = Counter(
    "orders_total",
    "Total number of orders that reached a terminal status",
    ["status"],
)
# Initialize both label series so they appear on /metrics immediately with 0
ORDERS_TOTAL.labels(status="priced")
ORDERS_TOTAL.labels(status="failed")

PRICING_SECONDS = Histogram(
    "pricing_seconds",
    "Wall time of one pricing operation in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def read_secret(name: str) -> str:
    path = Path(f"/run/secrets/{name}")
    if not path.exists():
        logger.error("Secret %s missing!", name)
        sys.exit(1)
    return path.read_text().strip()


if Path("/run/secrets").exists():
    JWT_SECRET_KEY = read_secret("jwt_secret_key")
else:
    JWT_SECRET_KEY = "local_dev_key"

ALGORITHM = "HS256"
FORECAST_URL = os.getenv("FORECAST_URL", "http://forecast-api:8001")
FORECAST_TIMEOUT = float(os.getenv("FORECAST_TIMEOUT", "3.0"))

Base.metadata.create_all(bind=engine)
with engine.begin() as conn:
    conn.execute(text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS model_version VARCHAR(32);"))

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=FORECAST_TIMEOUT)
    yield
    await http_client.aclose()


app = FastAPI(
    title="Orders API (Observability + MLflow + Auth)",
    version="0.5.0",
    lifespan=lifespan,
)

# R3 & O1: Expose /metrics without authentication
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})

    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid user", headers={"WWW-Authenticate": "Bearer"})
    return user


@app.post("/auth/register", response_model=schemas.UserOut, status_code=201)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    hashed_pw = pwd_context.hash(user_in.password)
    new_user = models.User(username=user_in.username, hashed_password=hashed_pw)
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username taken")


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me", response_model=schemas.UserOut)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.get("/orders/{oid}/explanation")
async def get_order_explanation(
    oid: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Отримання document-grounded пояснення ціни для існуючого замовлення (RAG)."""
    order = crud.get_order(db, oid)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    payload = {
        "gr_liv_area": order.gr_liv_area,
        "total_bsmt_sf": order.total_bsmt_sf,
        "garage_area": order.garage_area,
        "year_built": order.year_built,
        "overall_qual": order.overall_qual,
        "ms_zoning": order.ms_zoning,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{FORECAST_URL}/explain",
                json=payload,
                headers={"X-Order-Id": str(oid)},
            )
            if resp.status_code != 200:
                emit_event("error", "explanation_failed", order_id=order.id, status_code=resp.status_code)
                raise HTTPException(
                    status_code=502,
                    detail=f"Explanation service error: {resp.text}",
                )
            result = resp.json()
    except httpx.HTTPError as exc:
        emit_event("error", "explanation_failed", order_id=order.id, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail=f"Forecast explanation service unavailable: {exc}",
        )

    emit_event("info", "explanation_succeeded", order_id=order.id)
    return {
        "order_id": order.id,
        "client_name": order.client_name,
        "order_status": order.status,
        "stored_predicted_price": order.predicted_price,
        **result,
    }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
    reraise=True,
)
async def call_forecast_with_retry(payload: dict, order_id: int) -> dict:
    """Call forecast-api with retry backoff and X-Order-Id correlation header (O2)."""
    response = await http_client.post(
        f"{FORECAST_URL}/predict",
        json=payload,
        headers={"X-Order-Id": str(order_id)},
    )
    response.raise_for_status()
    return response.json()


async def fetch_forecast_background(order_id: int):
    db: Session = SessionLocal()
    order = None
    t0 = time.perf_counter()
    try:
        order = db.query(models.Order).filter(models.Order.id == order_id).first()
        if not order:
            return

        emit_event("info", "pricing_started", order_id=order.id)

        payload = {
            "gr_liv_area": order.gr_liv_area,
            "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area,
            "year_built": order.year_built,
            "overall_qual": order.overall_qual,
            "ms_zoning": order.ms_zoning,
        }
        res_data = await call_forecast_with_retry(payload, order.id)
        duration_s = time.perf_counter() - t0
        duration_ms = round(duration_s * 1000, 2)

        order.predicted_price = res_data.get("predicted_price")
        order.model_version = (
            str(res_data.get("model_version"))
            if res_data.get("model_version") is not None
            else None
        )
        order.status = "priced"

        PRICING_SECONDS.observe(duration_s)
        ORDERS_TOTAL.labels(status="priced").inc()
        emit_event(
            "info",
            "pricing_succeeded",
            order_id=order.id,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_s = time.perf_counter() - t0
        if order:
            order.status = "failed"
            PRICING_SECONDS.observe(duration_s)
            ORDERS_TOTAL.labels(status="failed").inc()
            emit_event(
                "error",
                "pricing_failed",
                order_id=order.id,
                error=str(exc),
            )
    finally:
        db.commit()
        db.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/orders", response_model=schemas.OrderRead, status_code=status.HTTP_202_ACCEPTED)
async def create_order(
    data: schemas.OrderCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    order = crud.create_order(db, data)
    background_tasks.add_task(fetch_forecast_background, order.id)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=schemas.OrderRead.model_validate(order).model_dump(mode="json"),
        headers={"Location": f"{request.base_url}orders/{order.id}"},
    )


@app.post("/orders/batch", response_model=schemas.BatchResponse)
async def create_orders_batch(
    orders_data: List[schemas.OrderCreate],
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if len(orders_data) > 50:
        raise HTTPException(status_code=422, detail="BATCH_MAX_ITEMS exceeded.")

    saved_orders = [crud.create_order(db, data) for data in orders_data]

    async def fetch_single(index: int, order: models.Order):
        t0 = time.perf_counter()
        emit_event("info", "pricing_started", order_id=order.id)
        payload = {
            "gr_liv_area": order.gr_liv_area,
            "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area,
            "year_built": order.year_built,
            "overall_qual": order.overall_qual,
            "ms_zoning": order.ms_zoning,
        }
        try:
            res_json = await call_forecast_with_retry(payload, order.id)
            duration_s = time.perf_counter() - t0
            duration_ms = round(duration_s * 1000, 2)
            price = res_json.get("predicted_price")
            m_ver = (
                str(res_json.get("model_version"))
                if res_json.get("model_version") is not None
                else None
            )
            order.predicted_price = price
            order.model_version = m_ver
            order.status = "priced"

            PRICING_SECONDS.observe(duration_s)
            ORDERS_TOTAL.labels(status="priced").inc()
            emit_event(
                "info",
                "pricing_succeeded",
                order_id=order.id,
                duration_ms=duration_ms,
            )
            return {
                "index": index,
                "order_id": order.id,
                "status": "priced",
                "predicted_price": price,
                "model_version": m_ver,
            }
        except Exception as exc:
            duration_s = time.perf_counter() - t0
            order.status = "failed"
            PRICING_SECONDS.observe(duration_s)
            ORDERS_TOTAL.labels(status="failed").inc()
            emit_event(
                "error",
                "pricing_failed",
                order_id=order.id,
                error=str(exc),
            )
            return {
                "index": index,
                "order_id": order.id,
                "status": "failed",
                "error": "Unavailable",
            }

    results = await asyncio.gather(*[fetch_single(i, order) for i, order in enumerate(saved_orders)])
    db.commit()
    return {"total": len(results), "results": results}


@app.get("/orders", response_model=list[schemas.OrderRead])
def list_orders(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return crud.list_orders(db, limit, offset)


@app.get("/orders/{oid}", response_model=schemas.OrderRead)
def get_order(oid: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = crud.get_order(db, oid)
    if not order:
        raise HTTPException(status_code=404, detail="Not found")
    return order


@app.patch("/orders/{oid}", response_model=schemas.OrderRead)
async def update_order(
    oid: int,
    data: schemas.OrderUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    order = crud.get_order(db, oid)
    if not order:
        raise HTTPException(status_code=404, detail="Not found")

    update_data = data.model_dump(exclude_unset=True)
    feature_fields = {"gr_liv_area", "total_bsmt_sf", "garage_area", "year_built", "overall_qual", "ms_zoning"}
    needs_recalc = any(f in update_data for f in feature_fields)

    crud.update_order(db, order, data)

    if needs_recalc:
        order.status = "pending"
        order.predicted_price = None
        order.model_version = None
        db.commit()
        db.refresh(order)
        background_tasks.add_task(fetch_forecast_background, order.id)

    return order


@app.delete("/orders/{oid}", status_code=204)
def delete_order(oid: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = crud.get_order(db, oid)
    if not order:
        raise HTTPException(status_code=404, detail="Not found")
    crud.delete_order(db, order)