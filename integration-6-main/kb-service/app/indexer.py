from datetime import datetime, timezone
from pathlib import Path
import re
import psycopg
from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
INDEX_VERSION = "v1"
SOURCE_DOC_ID = "ames-doc"


def ensure_schema(conn: psycopg.Connection) -> None:
    """Створення розширення pgvector, таблиці passages та HNSW-індексу."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS passages (
                id bigserial PRIMARY KEY,
                source text NOT NULL,
                section text,
                page int,
                passage_idx int NOT NULL,
                version text NOT NULL,
                ingested_at timestamptz NOT NULL,
                content text NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS passages_embedding_hnsw_idx
            ON passages USING hnsw (embedding vector_cosine_ops);
            """
        )
    conn.commit()


def segment_markdown(
    md_text: str,
    max_words: int = 80,
    overlap_words: int = 15,
) -> list[dict]:
    """Структурно-орієнтована сегментація Markdown на пасажі з перекриттям."""
    passages: list[dict] = []
    current_section = "sec-0-intro"
    current_page = 1
    buffer_lines: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer_lines
        text = " ".join(line.strip() for line in buffer_lines if line.strip())
        buffer_lines = []
        if not text:
            return
        words = text.split()
        if len(words) <= max_words:
            passages.append(
                {
                    "source": SOURCE_DOC_ID,
                    "section": current_section,
                    "page": current_page,
                    "content": text,
                }
            )
            return
        step = max(1, max_words - overlap_words)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + max_words])
            if chunk:
                passages.append(
                    {
                        "source": SOURCE_DOC_ID,
                        "section": current_section,
                        "page": current_page,
                        "content": chunk,
                    }
                )
            if start + max_words >= len(words):
                break

    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            flush_buffer()
            header = line[3:].strip()
            match = re.match(r"([^|]+)\|\s*page:\s*(\d+)", header)
            if match:
                current_section = match.group(1).strip()
                current_page = int(match.group(2))
            else:
                current_section = header
        elif line.startswith("# "):
            continue
        elif line:
            buffer_lines.append(line)

    flush_buffer()
    return passages


def vector_to_pg_literal(vec) -> str:
    """Перетворення вектора у текстовий літерал pgvector."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def index_corpus(
    conn: psycopg.Connection,
    embedder: TextEmbedding,
    doc_path: Path,
    version: str = INDEX_VERSION,
) -> int:
    """Повна реіндексація документа у таблицю passages із заміною старих записів source."""
    ensure_schema(conn)
    md_text = doc_path.read_text(encoding="utf-8")
    passages = segment_markdown(md_text)
    if not passages:
        return 0

    texts = [p["content"] for p in passages]
    vectors = list(embedder.embed(texts))
    ingested_at = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        # Підтримання узгодженості: видалення старих фрагментів цього джерела перед вставкою
        cur.execute("DELETE FROM passages WHERE source = %s;", (SOURCE_DOC_ID,))
        for idx, (p, vec) in enumerate(zip(passages, vectors)):
            cur.execute(
                """
                INSERT INTO passages (
                    source, section, page, passage_idx, version, ingested_at, content, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector);
                """,
                (
                    p["source"],
                    p["section"],
                    p["page"],
                    idx,
                    version,
                    ingested_at,
                    p["content"],
                    vector_to_pg_literal(vec),
                ),
            )
    conn.commit()
    return len(passages)