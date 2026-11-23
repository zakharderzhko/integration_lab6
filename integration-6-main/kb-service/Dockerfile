FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY kb-service/pyproject.toml ./
RUN uv sync --no-install-project

# Попереднє завантаження моделі ембедингів (384 виміри, косинусна подібність)
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/model_cache')"

COPY kb-service/app ./app
COPY kb-service/data ./data

EXPOSE 8002
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]