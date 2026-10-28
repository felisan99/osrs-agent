"""Postgres loading (SPEC §9): chunks with embeddings, money making guides and quests.

Every page is replaced atomically (delete by title + insert in one transaction), so re-ingesting a
page never leaves duplicates or a half-updated page behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from app.ingest.chunk import Chunk
from app.ingest.mmg import MoneyMakingGuide
from app.ingest.quests import QuestRecord

INSERT_DOCUMENT = (
    "insert into documents (page_title, section, page_type, url, content, metadata, embedding) "
    "values (%s, %s, %s, %s, %s, %s, %s)"
)


def connect(conninfo: str) -> psycopg.Connection:
    """Connection with the pgvector adapter registered (the extension must already exist)."""
    connection = psycopg.connect(conninfo)
    register_vector(connection)
    return connection


def replace_page(
    conn: psycopg.Connection, title: str, chunks: Sequence[Chunk], vectors: Sequence[Any]
) -> None:
    """Delete the page's chunks and insert the new ones. Call inside ``conn.transaction()``."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} embeddings for {title!r}")
    conn.execute("delete from documents where page_title = %s", (title,))
    if not chunks:
        return
    rows = [
        (c.page_title, c.section or None, c.page_type, c.url, c.content, Jsonb(c.metadata), vector)
        for c, vector in zip(chunks, vectors, strict=True)
    ]
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_DOCUMENT, rows)


def upsert_guide(conn: psycopg.Connection, guide: MoneyMakingGuide) -> None:
    conn.execute(
        "insert into moneymaking_guides (page_title, data) values (%s, %s) "
        "on conflict (page_title) do update set data = excluded.data, ingested_at = now()",
        (guide.page_title, Jsonb(guide.model_dump(mode="json"))),
    )


def upsert_quest(conn: psycopg.Connection, quest: QuestRecord) -> None:
    conn.execute(
        "insert into quests (page_title, data) values (%s, %s) "
        "on conflict (page_title) do update set data = excluded.data, ingested_at = now()",
        (quest.title, Jsonb(quest.model_dump(mode="json"))),
    )


def existing_revisions(conn: psycopg.Connection) -> dict[str, int]:
    """``{page title: revision id}`` already loaded, so unchanged pages can be skipped."""
    rows = conn.execute(
        "select page_title, (metadata->>'revision_id')::bigint from documents "
        "where (metadata->>'chunk_index')::int = 0"
    ).fetchall()
    return {title: revision for title, revision in rows if revision is not None}


def counts_by_page_type(conn: psycopg.Connection) -> list[tuple[str, int, int]]:
    """``(page_type, chunks, pages)`` for the acceptance report."""
    return [
        (page_type, chunks, pages)
        for page_type, chunks, pages in conn.execute(
            "select page_type, count(*), count(distinct page_title) from documents "
            "group by page_type order by page_type"
        )
    ]
