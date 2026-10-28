"""Page cache, per-page pipeline and the Postgres loader."""

from __future__ import annotations

import json
import pathlib

import httpx
import numpy as np
import psycopg
import pytest

from app.clients.prices import ItemMeta
from app.clients.wiki import WikiClient
from app.config import Settings
from app.ingest.chunk import ChunkConfig, build_quest_matcher
from app.ingest.dump import CachedPage, cache_path, done_path, fetch_entry, read_entry
from app.ingest.load import (
    connect,
    counts_by_page_type,
    existing_revisions,
    replace_page,
    upsert_guide,
    upsert_quest,
)
from app.ingest.mmg import parse_money_making_guide
from app.ingest.pipeline import PipelineContext, process_page
from app.ingest.quests import parse_quest
from app.ingest.scope import SCOPES, ScopeEntry
from app.tools.items import ItemIndex
from tests.conftest import TEST_UA, RecordingSleep

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wiki"
API = "https://oldschool.runescape.wiki/api.php"


# --------------------------------------------------------------------------- scope
def test_scope_definitions_are_consistent() -> None:
    valid = {"quest", "mmg", "skill_guide", "item", "npc", "other"}
    for name, entries in SCOPES.items():
        categories = [e.category for e in entries]
        assert len(categories) == len(set(categories)), f"duplicate category in {name}"
        assert {e.page_type for e in entries} <= valid
    assert set(SCOPES["core"]) < set(SCOPES["v0"])
    assert [e.category for e in SCOPES["v0"][:3]] == ["Quests", "Miniquests", "Quick guides"]
    assert ScopeEntry("Money making guides", "mmg").slug == "money-making-guides"
    assert ScopeEntry("Free-to-play quests", "quest").slug == "free-to-play-quests"


# --------------------------------------------------------------------------- cache
def api_page(title: str, text: str, revid: int = 1) -> dict:
    return {
        "pageid": revid,
        "title": title,
        "revisions": [
            {
                "revid": revid,
                "timestamp": "2026-10-01T00:00:00Z",
                "slots": {"main": {"content": text}},
            }
        ],
    }


@pytest.fixture
def wiki(http, retry, sleeper: RecordingSleep) -> WikiClient:
    settings = Settings(_env_file=None, prices_user_agent=TEST_UA, wiki_request_delay_s=0)
    return WikiClient(settings, http=http, retry=retry, sleep=sleeper)


ENTRY = ScopeEntry("Quests", "quest")


async def test_fetch_entry_caches_pages_skips_redirects_and_marks_completion(
    respx_mock, wiki, tmp_path
) -> None:
    body = {
        "query": {
            "pages": [
                api_page("Cook's Assistant", "text", 5),
                api_page("Old name", "#REDIRECT [[X]]", 6),
            ]
        }
    }
    route = respx_mock.get(API).mock(return_value=httpx.Response(200, json=body))

    written, skipped = await fetch_entry(wiki, ENTRY, tmp_path)

    assert (written, skipped) == (1, False)
    pages = list(read_entry(tmp_path, ENTRY))
    assert [(p.title, p.revision_id, p.page_type, p.category) for p in pages] == [
        ("Cook's Assistant", 5, "quest", "Quests")
    ]
    assert done_path(tmp_path, ENTRY).exists()
    assert json.loads(done_path(tmp_path, ENTRY).read_text())["pages"] == 1

    # A complete category is not downloaded again...
    assert await fetch_entry(wiki, ENTRY, tmp_path) == (1, True)
    assert route.call_count == 1
    # ...unless a refresh is asked for.
    assert await fetch_entry(wiki, ENTRY, tmp_path, refresh=True) == (1, False)
    assert route.call_count == 2


async def test_limited_fetch_never_looks_complete(respx_mock, wiki, tmp_path) -> None:
    body = {
        "query": {"pages": [api_page(f"P{i}", "t", i) for i in range(5)]},
        "continue": {"gcmcontinue": "x"},
    }
    respx_mock.get(API).mock(return_value=httpx.Response(200, json=body))
    written, _ = await fetch_entry(wiki, ENTRY, tmp_path, limit=2)
    assert written == 2 and not done_path(tmp_path, ENTRY).exists()


async def test_a_failed_fetch_leaves_no_partial_cache(respx_mock, wiki, tmp_path) -> None:
    respx_mock.get(API).mock(return_value=httpx.Response(500))
    with pytest.raises(Exception, match="HTTP 500"):
        await fetch_entry(wiki, ENTRY, tmp_path)
    assert not cache_path(tmp_path, ENTRY).exists() and not done_path(tmp_path, ENTRY).exists()


def test_reading_a_missing_cache_yields_nothing(tmp_path) -> None:
    assert list(read_entry(tmp_path, ENTRY)) == []


# --------------------------------------------------------------------------- pipeline
def cached(name: str, page_type: str, revision: int = 3) -> CachedPage:
    text = (FIXTURES / name).read_text()
    title = text.split("-->")[0].removeprefix("<!-- title:").strip()
    return CachedPage(title, 1, revision, "2026-10-01T00:00:00Z", text, page_type, "X")


def context(**overrides) -> PipelineContext:
    overrides.setdefault("chunk_config", ChunkConfig())
    return PipelineContext(**overrides)


def test_quest_pages_yield_chunks_and_a_structured_record() -> None:
    titles = {"Legends' Quest", "Dream Mentor", "Rune Mysteries"}
    result = process_page(
        cached("quest_dragon_slayer_ii_head.wiki", "quest"),
        context(quest_titles=titles, quest_matcher=build_quest_matcher(titles)),
    )
    assert result is not None and result.guide is None
    assert result.quest is not None and result.quest.skill_requirements["Magic"] == 75
    assert {q.quest for q in result.quest.quest_requirements} == titles
    assert result.chunks and all(c.page_type == "quest" for c in result.chunks)
    assert {"Legends' Quest", "Dream Mentor"} <= {
        q for c in result.chunks for q in c.metadata["quests"]
    }


def test_guide_pages_get_items_resolved_and_the_reference_profit_attached() -> None:
    index = ItemIndex([ItemMeta(id=1, name="Fish offcuts"), ItemMeta(id=2, name="Shark lure")])
    result = process_page(
        cached("mmg_plain_per_kill.wiki", "mmg"),
        context(
            item_index=index, reference_profits={"Money making guide/Aerial fishing": 1_234_567}
        ),
    )
    assert result is not None and result.guide is not None and result.quest is None
    assert result.guide.source_hourly_profit == 1_234_567
    assert [i.item_id for i in result.guide.items_in + result.guide.items_out] == [1, 2]
    assert result.chunks[0].content.startswith(
        "Money making guide/Aerial fishing\nMoney making guide:"
    )


def test_a_guide_with_an_unpriceable_item_becomes_unsupported() -> None:
    result = process_page(
        cached("mmg_plain_per_kill.wiki", "mmg"), context(item_index=ItemIndex([]))
    )
    assert result is not None and result.guide is not None
    assert not result.guide.rankable and any(
        "not in /mapping" in u for u in result.guide.unsupported
    )


def test_only_real_quests_stay_as_quest_requirements_of_a_guide() -> None:
    text = (
        "{{Mmgtable\n|Activity = X\n|Quest =\n* [[Ghosts Ahoy]]\n"
        "* [[Nightmare Zone/Strategies]]\n|Output1 = Coins\n|Output1num = 1\n}}"
    )
    page = CachedPage("Money making guide/X", 1, 1, "t", text, "mmg", "C")
    result = process_page(page, context(quest_titles={"Ghosts Ahoy", "Dream Mentor"}))
    assert result is not None and result.guide is not None
    assert result.guide.quest_requirements == ["Ghosts Ahoy"]
    assert any("Nightmare Zone/Strategies" in issue for issue in result.guide.issues)
    # without a list of known quests nothing can be judged, so nothing is dropped
    untouched = process_page(page, context())
    assert untouched is not None and untouched.guide is not None
    assert untouched.guide.quest_requirements == ["Ghosts Ahoy", "Nightmare Zone/Strategies"]


def test_plain_pages_have_neither_guide_nor_quest() -> None:
    result = process_page(cached("item_abyssal_whip.wiki", "item"), context())
    assert result is not None and result.guide is None and result.quest is None
    assert result.revision_id == 3 and result.chunks


def test_redirects_and_empty_pages_produce_nothing() -> None:
    page = CachedPage("X", 1, 1, "t", "#REDIRECT [[Y]]", "other", "C")
    assert process_page(page, context()) is None


# --------------------------------------------------------------------------- loader (real Postgres)
def vectors(n: int, seed: int = 0) -> np.ndarray:
    raw = np.random.default_rng(seed).random((n, 384), dtype=np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def chunks_for(title: str, page_type: str = "other", n: int = 2, revision: int = 1):
    page = CachedPage(
        title,
        1,
        revision,
        "t",
        "\n\n".join(f"Paragraph {i} of {title}. " * 30 for i in range(n)),
        page_type,
        "C",
    )
    result = process_page(
        page, context(chunk_config=ChunkConfig(min_tokens=50, max_tokens=150, overlap_tokens=0))
    )
    assert result is not None
    return result.chunks


@pytest.fixture
def conn(db):
    with connect(db) as connection:
        yield connection
        connection.rollback()


def titles_in(conn) -> list[str]:
    return [r[0] for r in conn.execute("select distinct page_title from documents order by 1")]


def test_replace_page_inserts_chunks_with_metadata_and_embeddings(conn) -> None:
    chunks = chunks_for("Alpha", "quest", revision=11)
    with conn.transaction():
        replace_page(conn, "Alpha", chunks, vectors(len(chunks)))
    rows = conn.execute(
        "select section, page_type, url, metadata, vector_dims(embedding), content "
        "from documents where page_title = 'Alpha' order by id"
    ).fetchall()
    assert len(rows) == len(chunks) >= 2
    section, page_type, url, metadata, dims, content = rows[0]
    assert (page_type, dims) == ("quest", 384) and url.endswith("/Alpha")
    assert section is None  # the lead has no section
    assert metadata["chunk_index"] == 0 and metadata["revision_id"] == 11
    assert content.startswith("Alpha\n")


def test_replacing_a_page_never_duplicates_and_leaves_other_pages_alone(conn) -> None:
    for title in ("Alpha", "Beta"):
        chunks = chunks_for(title)
        with conn.transaction():
            replace_page(conn, title, chunks, vectors(len(chunks)))
    before = conn.execute("select count(*) from documents where page_title = 'Beta'").fetchone()[0]
    smaller = chunks_for("Alpha", n=1, revision=2)
    with conn.transaction():
        replace_page(conn, "Alpha", smaller, vectors(len(smaller), seed=1))
    assert conn.execute("select count(*) from documents where page_title = 'Alpha'").fetchone()[
        0
    ] == len(smaller)
    assert (
        conn.execute("select count(*) from documents where page_title = 'Beta'").fetchone()[0]
        == before
    )
    with conn.transaction():
        replace_page(conn, "Alpha", [], [])  # the page became empty: its old chunks must go
    assert titles_in(conn) == ["Beta"]


def test_a_chunk_embedding_mismatch_is_rejected_and_rolls_back(conn) -> None:
    chunks = chunks_for("Alpha")
    with conn.transaction():
        replace_page(conn, "Alpha", chunks, vectors(len(chunks)))
    with pytest.raises(ValueError, match="embeddings"), conn.transaction():
        replace_page(conn, "Alpha", chunks, vectors(len(chunks) - 1))
    assert conn.execute("select count(*) from documents where page_title = 'Alpha'").fetchone()[
        0
    ] == len(chunks)


def test_a_failure_mid_batch_rolls_the_whole_batch_back(conn) -> None:
    good = chunks_for("Good")
    bad = chunks_for("Bad")
    with pytest.raises(psycopg.errors.DataException), conn.transaction():
        replace_page(conn, "Good", good, vectors(len(good)))
        replace_page(conn, "Bad", bad, np.zeros((len(bad), 3), dtype=np.float32))  # wrong dimension
    assert titles_in(conn) == []


def test_existing_revisions_and_counts_by_page_type(conn) -> None:
    for title, page_type, revision in (("Q1", "quest", 5), ("Q2", "quest", 6), ("I1", "item", 7)):
        chunks = chunks_for(title, page_type, revision=revision)
        with conn.transaction():
            replace_page(conn, title, chunks, vectors(len(chunks)))
    assert existing_revisions(conn) == {"Q1": 5, "Q2": 6, "I1": 7}
    report = {page_type: (chunks, pages) for page_type, chunks, pages in counts_by_page_type(conn)}
    assert report["quest"][1] == 2 and report["item"][1] == 1
    assert report["quest"][0] >= 4


def test_guides_and_quests_are_upserted(conn) -> None:
    guide = parse_money_making_guide(
        "Money making guide/A", "{{Mmgtable\n|Activity = A\n|Output1 = X\n}}"
    )
    assert guide is not None
    upsert_guide(conn, guide)
    guide.activity = "A, revised"
    guide.source_hourly_profit = 99
    upsert_guide(conn, guide)
    rows = conn.execute(
        "select data from moneymaking_guides where page_title = 'Money making guide/A'"
    ).fetchall()
    assert (
        len(rows) == 1
        and rows[0][0]["activity"] == "A, revised"
        and rows[0][0]["source_hourly_profit"] == 99
    )

    quest = parse_quest(
        "Q", "{{Quest details\n|difficulty = Novice\n|requirements = *{{SCP|Magic|75}}\n}}"
    )
    assert quest is not None
    upsert_quest(conn, quest)
    upsert_quest(conn, quest)
    data = conn.execute("select data from quests where page_title = 'Q'").fetchall()
    assert len(data) == 1 and data[0][0]["skill_requirements"] == {"Magic": 75}
