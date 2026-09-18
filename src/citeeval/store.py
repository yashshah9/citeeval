"""Optional durable chunk backends for Corpus."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from citeeval.rag import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS citeeval_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    embedding JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS citeeval_chunks_source_idx ON citeeval_chunks (source);
"""


@runtime_checkable
class ChunkBackend(Protocol):
    def ensure_schema(self) -> None: ...

    def load_all(self) -> list[Chunk]: ...

    def save_chunks(self, chunks: list[Chunk]) -> None: ...

    def clear(self) -> None: ...


class PostgresChunkStore:
    """Persist chunks in Postgres (embeddings as JSON arrays)."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Postgres corpus requires psycopg. Install platformkit[postgres] or psycopg."
            ) from exc
        self._psycopg = psycopg
        self._dsn = dsn

    def ensure_schema(self) -> None:
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def load_all(self) -> list[Chunk]:
        with self._psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT id, document_id, source, text, ordinal, embedding "
                "FROM citeeval_chunks ORDER BY created_at, ordinal"
            ).fetchall()
        out: list[Chunk] = []
        for row in rows:
            emb: Any = row[5]
            if isinstance(emb, str):
                emb = json.loads(emb)
            out.append(
                Chunk(
                    id=str(row[0]),
                    document_id=str(row[1]),
                    source=str(row[2]),
                    text=str(row[3]),
                    ordinal=int(row[4]),
                    embedding=[float(x) for x in emb],
                )
            )
        return out

    def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    cur.execute(
                        "INSERT INTO citeeval_chunks "
                        "(id, document_id, source, text, ordinal, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                        "ON CONFLICT (id) DO NOTHING",
                        (
                            c.id,
                            c.document_id,
                            c.source,
                            c.text,
                            c.ordinal,
                            json.dumps(c.embedding),
                        ),
                    )
            conn.commit()

    def clear(self) -> None:
        with self._psycopg.connect(self._dsn) as conn:
            conn.execute("DELETE FROM citeeval_chunks")
            conn.commit()


def build_backend(driver: str, *, postgres_dsn: str) -> ChunkBackend | None:
    if driver in {"", "memory", "none"}:
        return None
    if driver == "postgres":
        store = PostgresChunkStore(postgres_dsn)
        store.ensure_schema()
        return store
    raise ValueError(f"Unknown corpus driver {driver!r}")
