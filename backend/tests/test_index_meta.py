"""The index remembers which embedding model built it, and can be rebuilt for another one."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import psycopg
import pytest

from app.config import Settings
from app.db import (
    aget_index_meta,
    embedding_column_dim,
    get_index_meta,
    reset_embedding_column,
    set_index_meta,
)
from app.service import check_index_model
from scripts.ingest import _index_matches_model


def add_row(conn: psycopg.Connection, dim: int) -> None:
    vector = "[" + ",".join("0.1" for _ in range(dim)) + "]"
    conn.execute(
        "insert into documents (page_title, section, page_type, url, content, embedding) "
        "values ('P', null, 'other', 'u', 'c', %s::vector)",
        (vector,),
    )


def test_the_column_dimension_can_be_read_and_rebuilt(scratch_db: str) -> None:
    with psycopg.connect(scratch_db) as conn:
        assert embedding_column_dim(conn) == 384
        add_row(conn, 384)
        reset_embedding_column(conn, 768)  # empties the table: old vectors are meaningless now
        assert embedding_column_dim(conn) == 768
        assert conn.execute("select count(*) from documents").fetchone()[0] == 0
        add_row(conn, 768)  # and the new dimension is accepted, with its index in place
        indexes = [
            r[0]
            for r in conn.execute("select indexname from pg_indexes where tablename = 'documents'")
        ]
        assert "documents_embedding_idx" in indexes


def test_index_meta_round_trips_and_updates(scratch_db: str) -> None:
    with psycopg.connect(scratch_db) as conn:
        assert get_index_meta(conn) == {}
        set_index_meta(conn, {"embedding_model": "a/model", "embedding_dim": "384"})
        set_index_meta(conn, {"embedding_model": "b/model"})
        assert get_index_meta(conn) == {"embedding_model": "b/model", "embedding_dim": "384"}


def test_a_database_without_the_meta_table_reads_as_empty(scratch_db: str) -> None:
    with psycopg.connect(scratch_db) as conn:
        conn.execute("drop table index_meta")
        assert get_index_meta(conn) == {}


def test_loading_refuses_to_mix_models_unless_forced(scratch_db: str, capsys) -> None:
    other = Settings(_env_file=None, embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768)
    same = Settings(_env_file=None)
    with psycopg.connect(scratch_db) as conn:
        add_row(conn, 384)
        set_index_meta(conn, {"embedding_model": same.embedding_model})
        assert _index_matches_model(conn, same, force=False)  # the model that built it: fine
        assert not _index_matches_model(conn, other, force=False)
        assert "re-run with --force" in capsys.readouterr().err
        assert embedding_column_dim(conn) == 384  # nothing was touched
        assert _index_matches_model(conn, other, force=True)  # rebuilt for the new model
        assert embedding_column_dim(conn) == 768
        assert conn.execute("select count(*) from documents").fetchone()[0] == 0


def test_the_same_dimension_from_another_model_is_still_a_mismatch(scratch_db: str) -> None:
    first = Settings(_env_file=None, embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=384)
    second = Settings(_env_file=None, embedding_model="x/other-model-384", embedding_dim=384)
    with psycopg.connect(scratch_db) as conn:
        add_row(conn, 384)
        set_index_meta(conn, {"embedding_model": first.embedding_model})
        assert not _index_matches_model(conn, second, force=False)


async def test_the_api_refuses_to_start_with_another_embedder(scratch_db: str) -> None:
    @asynccontextmanager
    async def connection():
        async with await psycopg.AsyncConnection.connect(scratch_db) as conn:
            yield conn

    pool = SimpleNamespace(connection=connection)
    settings = Settings(_env_file=None)
    await check_index_model(pool, settings)  # nothing recorded yet: nothing to compare
    with psycopg.connect(scratch_db) as conn:
        set_index_meta(conn, {"embedding_model": settings.embedding_model})
    await check_index_model(pool, settings)  # same model: fine
    other = Settings(_env_file=None, embedding_model="BAAI/bge-base-en-v1.5", embedding_dim=768)
    with pytest.raises(RuntimeError, match="vectors cannot be compared"):
        await check_index_model(pool, other)
    async with connection() as conn:
        assert (await aget_index_meta(conn))["embedding_model"] == settings.embedding_model
