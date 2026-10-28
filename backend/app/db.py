"""Postgres helpers: URL handling and a minimal, idempotent migration runner.

Migrations live in ``supabase/migrations`` (Supabase CLI naming) and are plain SQL, so the same
files run on Homebrew Postgres, the Supabase CLI's local database and a hosted Supabase project.
The runner records applied files in ``schema_migrations`` and applies each file in its own
transaction. (LangGraph's checkpoint tables are created by its own ``setup()``, not here.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg

from app.config import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "supabase" / "migrations"


@asynccontextmanager
async def async_connection(conninfo: str) -> AsyncIterator[psycopg.AsyncConnection]:
    """One short-lived async connection (scripts and tests; the API server uses a pool)."""
    async with await psycopg.AsyncConnection.connect(conninfo) as conn:
        yield conn


class MissingDatabaseUrl(RuntimeError):
    pass


def require_db_url(settings: Settings) -> str:
    if not settings.supabase_db_url:
        raise MissingDatabaseUrl(
            "SUPABASE_DB_URL is not set. For local Homebrew Postgres use "
            "postgresql://127.0.0.1:5432/osrs_agent (see backend/.env.example)."
        )
    return settings.supabase_db_url


def apply_migrations(conninfo: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending ``*.sql`` files in name order; return the names applied by this call."""
    files = sorted(directory.glob("*.sql"))
    applied_now: list[str] = []
    with psycopg.connect(conninfo) as conn:
        conn.execute(
            "create table if not exists schema_migrations ("
            " name text primary key, applied_at timestamptz not null default now())"
        )
        conn.commit()
        done = {row[0] for row in conn.execute("select name from schema_migrations")}
        for path in files:
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
            applied_now.append(path.name)
    return applied_now


# --------------------------------------------------------------------------- embedding index
def embedding_column_dim(conn: psycopg.Connection) -> int | None:
    """Dimension of ``documents.embedding`` (pgvector keeps N of ``vector(N)`` in ``atttypmod``)."""
    row = conn.execute(
        "select atttypmod from pg_attribute "
        "where attrelid = 'documents'::regclass and attname = 'embedding'"
    ).fetchone()
    return row[0] if row and row[0] > 0 else None


def reset_embedding_column(conn: psycopg.Connection, dim: int) -> None:
    """Empty ``documents`` and re-create its vector column and index for ``dim`` dimensions.

    Needed whenever the embedding model changes: vectors of two models are never comparable, so the
    index is rebuilt from scratch (``scripts.ingest load`` fills it again).
    """
    conn.execute("truncate documents restart identity")
    conn.execute("drop index if exists documents_embedding_idx")
    conn.execute(f"alter table documents alter column embedding type vector({int(dim)})")
    conn.execute(
        "create index documents_embedding_idx on documents using hnsw (embedding vector_cosine_ops)"
    )


def get_index_meta(conn: psycopg.Connection) -> dict[str, str]:
    """What built the index (``index_meta``); empty for a database that predates the table."""
    exists = conn.execute("select to_regclass('index_meta') is not null").fetchone()
    if not exists or not exists[0]:
        return {}
    return dict(conn.execute("select key, value from index_meta").fetchall())


def set_index_meta(conn: psycopg.Connection, values: dict[str, str]) -> None:
    for key, value in values.items():
        conn.execute(
            "insert into index_meta (key, value) values (%s, %s) "
            "on conflict (key) do update set value = excluded.value, updated_at = now()",
            (key, value),
        )


async def aget_index_meta(conn: psycopg.AsyncConnection) -> dict[str, str]:
    cursor = await conn.execute("select to_regclass('index_meta') is not null")
    row = await cursor.fetchone()
    if not row or not row[0]:
        return {}
    cursor = await conn.execute("select key, value from index_meta")
    return dict(await cursor.fetchall())
