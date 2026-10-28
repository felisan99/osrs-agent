"""The initial migration (SPEC §9) against a real Postgres + pgvector."""

from __future__ import annotations

import json

import psycopg
import pytest
from psycopg import errors

from app.db import apply_migrations

DIM = 384


def vec(*head: float) -> str:
    """A 384-dim pgvector literal starting with ``head`` and zero-padded."""
    values = [*head, *([0.0] * (DIM - len(head)))]
    return "[" + ",".join(str(v) for v in values) + "]"


def insert_doc(conn, title: str, content: str, embedding: str, page_type: str = "other") -> None:
    conn.execute(
        "insert into documents (page_title, section, page_type, url, content, embedding)"
        " values (%s, %s, %s, %s, %s, %s::vector)",
        (
            title,
            "Intro",
            page_type,
            f"https://oldschool.runescape.wiki/w/{title}",
            content,
            embedding,
        ),
    )


@pytest.fixture
def conn(db):
    with psycopg.connect(db) as connection:
        yield connection
        connection.rollback()


def test_migrations_are_idempotent(pg_url) -> None:
    assert apply_migrations(pg_url) == []  # already applied by the fixture


def test_expected_tables_and_indexes_exist(conn) -> None:
    tables = {
        r[0] for r in conn.execute("select tablename from pg_tables where schemaname='public'")
    }
    assert {"documents", "moneymaking_guides", "jev_judgments", "schema_migrations"} <= tables
    indexes = {
        r[0]: r[1]
        for r in conn.execute(
            "select indexname, indexdef from pg_indexes where tablename='documents'"
        )
    }
    embedding_index = indexes["documents_embedding_idx"].lower()
    assert "using hnsw" in embedding_index and "vector_cosine_ops" in embedding_index
    assert "using gin" in indexes["documents_tsv_idx"].lower()
    assert "documents_page_title_idx" in indexes


def test_embedding_dimension_is_enforced(conn) -> None:
    insert_doc(conn, "Ok", "fine", vec(1.0))
    with pytest.raises(errors.DataException, match="384"):
        insert_doc(conn, "Bad", "wrong size", "[1,2,3]")


def test_page_type_is_constrained(conn) -> None:
    for good in ("quest", "mmg", "skill_guide", "item", "npc", "other"):
        insert_doc(conn, f"T-{good}", "x", vec(1.0), page_type=good)
    with pytest.raises(errors.CheckViolation):
        insert_doc(conn, "Nope", "x", vec(1.0), page_type="spell")


def test_full_text_search_uses_english_stemming(conn) -> None:
    insert_doc(conn, "Whip", "The abyssal whip is a one-handed melee weapon.", vec(1.0))
    insert_doc(conn, "Cake", "A delicious cake that heals three hitpoints.", vec(0.0, 1.0))
    rows = conn.execute(
        "select page_title from documents where tsv @@ plainto_tsquery('english', %s)",
        ("abyssal whips weapons",),  # plural forms still match the stored singular text
    ).fetchall()
    assert rows == [("Whip",)]


def test_cosine_nearest_neighbour(conn) -> None:
    insert_doc(conn, "East", "e", vec(1.0, 0.0))
    insert_doc(conn, "North", "n", vec(0.0, 1.0))
    insert_doc(conn, "NorthEast", "ne", vec(1.0, 1.0))
    ranked = conn.execute(
        "select page_title from documents order by embedding <=> %s::vector limit 3",
        (vec(0.9, 0.2),),
    ).fetchall()
    assert [r[0] for r in ranked] == ["East", "NorthEast", "North"]


def test_moneymaking_guides_one_row_per_page_and_jsonb_roundtrip(conn) -> None:
    payload = {"activity": "Aerial fishing", "skill_requirements": {"Fishing": 43}}
    conn.execute(
        "insert into moneymaking_guides (page_title, data) values (%s, %s)",
        ("MMG/A", json.dumps(payload)),
    )
    with pytest.raises(errors.UniqueViolation):
        conn.execute(
            "insert into moneymaking_guides (page_title, data) values (%s, %s)", ("MMG/A", "{}")
        )
    conn.rollback()
    conn.execute(
        "insert into moneymaking_guides (page_title, data) values (%s, %s)",
        ("MMG/B", json.dumps(payload)),
    )
    (data,) = conn.execute(
        "select data from moneymaking_guides where page_title='MMG/B'"
    ).fetchone()
    assert data["skill_requirements"]["Fishing"] == 43


def test_jev_judgments_kind_is_constrained(conn) -> None:
    conn.execute(
        "insert into jev_judgments (run_id, kind, probs) values ('r1', 'relevance', '{\"p\": 0.8}')"
    )
    with pytest.raises(errors.CheckViolation):
        conn.execute("insert into jev_judgments (run_id, kind, probs) values ('r1', 'guess', '{}')")
