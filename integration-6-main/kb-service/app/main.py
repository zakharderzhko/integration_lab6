from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import time

from fastapi import FastAPI, HTTPException
from fastembed import TextEmbedding
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
import psycopg

from app.indexer import (
    EMBEDDING_MODEL_NAME,
    INDEX_VERSION,
    SOURCE_DOC_ID,
    ensure_schema,
    index_corpus,
    vector_to_pg_literal,
)

DOC_PATH = Path("/app/data/ames_housing_docs.md")

# Доменні метрики Prometheus для сервісу знань
KB_SEARCH_TOTAL = Counter(
    "kb_search_total",
    "Total number of semantic search requests in kb-api",
    ["status"],
)
KB_SEARCH_SECONDS = Histogram(
    "kb_search_seconds",
    "Latency of semantic search queries in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

embedder: TextEmbedding | None = None


def log_event(level: str, event: str, **fields) -> None:
    """Однорядкове структуроване JSON-логування подій."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)


def get_db_password() -> str:
    pw_file = os.getenv("POSTGRES_PASSWORD_FILE", "/run/secrets/db_password")
    if os.path.exists(pw_file):
        return Path(pw_file).read_text(encoding="utf-8").strip()
    return os.getenv("POSTGRES_PASSWORD", "postgres")


def get_conn() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "orders"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=get_db_password(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    embedder = TextEmbedding(EMBEDDING_MODEL_NAME, cache_dir="/app/model_cache")
    with get_conn() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM passages;")
            count = cur.fetchone()[0]
        if count == 0 and DOC_PATH.exists():
            indexed = index_corpus(conn, embedder, DOC_PATH, INDEX_VERSION)
            log_event(
                "info",
                "corpus_indexed",
                source=SOURCE_DOC_ID,
                passages=indexed,
                model=EMBEDDING_MODEL_NAME,
                version=INDEX_VERSION,
            )
    yield


app = FastAPI(title="Knowledge Base Service (kb-api)", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Free-form semantic search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of nearest passages to retrieve")
    source: str | None = Field(None, description="Optional metadata filter by source document")


class PassageHit(BaseModel):
    id: int
    source: str
    section: str | None
    page: int | None
    version: str
    content: str
    cosine_similarity: float


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    top_k: int
    results: list[PassageHit]


@app.get("/health")
def health():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM passages;")
            total = cur.fetchone()[0]
    return {"status": "ok", "passages_indexed": total, "embedding_model": EMBEDDING_MODEL_NAME}


@app.get("/stats")
def stats():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, version, COUNT(*) AS passages
                FROM passages
                GROUP BY source, version
                ORDER BY source, version;
                """
            )
            rows = cur.fetchall()
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collections": [
            {"source": r[0], "version": r[1], "passages": r[2]} for r in rows
        ],
    }


@app.post("/reindex")
def reindex():
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    with get_conn() as conn:
        count = index_corpus(conn, embedder, DOC_PATH, INDEX_VERSION)
    log_event("info", "corpus_reindexed", source=SOURCE_DOC_ID, passages=count, version=INDEX_VERSION)
    return {"status": "reindexed", "source": SOURCE_DOC_ID, "passages": count, "version": INDEX_VERSION}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")

    t0 = time.perf_counter()
    try:
        query_vec = next(embedder.embed([req.query]))
        vec_lit = vector_to_pg_literal(query_vec)

        with get_conn() as conn:
            with conn.cursor() as cur:
                if req.source:
                    cur.execute(
                        """
                        SELECT id, source, section, page, version, content,
                               1 - (embedding <=> %s::vector) AS cosine_similarity
                        FROM passages
                        WHERE source = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (vec_lit, req.source, vec_lit, req.top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, source, section, page, version, content,
                               1 - (embedding <=> %s::vector) AS cosine_similarity
                        FROM passages
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (vec_lit, vec_lit, req.top_k),
                    )
                rows = cur.fetchall()

        elapsed = time.perf_counter() - t0
        KB_SEARCH_TOTAL.labels(status="ok").inc()
        KB_SEARCH_SECONDS.observe(elapsed)

        hits = [
            PassageHit(
                id=r[0],
                source=r[1],
                section=r[2],
                page=r[3],
                version=r[4],
                content=r[5],
                cosine_similarity=round(float(r[6]), 4),
            )
            for r in rows
        ]
        log_event(
            "info",
            "kb_search",
            query=req.query,
            top_k=req.top_k,
            hits=len(hits),
            latency_ms=round(elapsed * 1000, 2),
        )
        return SearchResponse(
            query=req.query,
            embedding_model=EMBEDDING_MODEL_NAME,
            top_k=req.top_k,
            results=hits,
        )
    except Exception as exc:
        KB_SEARCH_TOTAL.labels(status="error").inc()
        log_event("error", "kb_search_failed", query=req.query, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Search error: {exc}")