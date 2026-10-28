"""Hybrid retrieval against a real Postgres + pgvector (SPEC §13.4)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import psycopg
import pytest

from app.config import Settings
from app.rag.retrieve import Retriever, keyword_query, vector_literal
from app.rag.titles import TitleIndex

DIM = 384


def unit(axis: int, tilt: float = 0.0, axis2: int | None = None) -> list[float]:
    """A vector mostly along ``axis`` (optionally tilted towards ``axis2``)."""
    v = [0.0] * DIM
    v[axis] = 1.0
    if axis2 is not None:
        v[axis2] = tilt
    return v


def add(conn, title: str, content: str, vector: list[float], page_type: str = "other") -> None:
    conn.execute(
        "insert into documents (page_title, section, page_type, url, content, metadata, embedding)"
        " values (%s, '', %s, %s, %s, '{}', %s::vector)",
        (title, page_type, f"https://example.org/{title}", content, vector_literal(vector)),
    )


@pytest.fixture
def corpus(db):
    with psycopg.connect(db) as conn:
        add(
            conn,
            "Whip",
            "The abyssal whip is a one-handed melee weapon requiring 70 Attack.",
            unit(0),
        )
        add(conn, "Bones", "Dragon bones give 72 Prayer experience when buried.", unit(1))
        add(conn, "Cake", "A cake heals three hitpoints over three bites.", unit(2))
        add(conn, "Lobster", "Cooked lobster heals twelve hitpoints.", unit(3))
        add(conn, "Zulrah", "Zulrah is a snake boss found at Zul-Andra.", unit(4), page_type="npc")
        conn.commit()
    return db


def retriever(
    db: str, vector: list[float], titles: TitleIndex | None = None, **settings
) -> Retriever:
    @asynccontextmanager
    async def connect():
        async with await psycopg.AsyncConnection.connect(db) as conn:
            yield conn

    return Retriever(connect, lambda text: vector, Settings(_env_file=None, **settings), titles)


# --------------------------------------------------------------------------- query building
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How do I train prayer?", "how or train or prayer"),
        ("Zulrah's drop table", "zulrah's or drop or table"),
        ("dragon dagger p++ special", "dragon or dagger or p++ or special"),
        ("a an of to", ""),
        ("prayer prayer PRAYER", "prayer"),
        ("¿cuánto cuesta?", "cuánto or cuesta"),  # accents stay inside the word
        ("Monkey Madness II", "monkey or madness or ii"),  # "II" separates it from "I"
        ("xp or gp", "xp or gp"),  # game vocabulary stays, the word "or" is never an operator
        ("", ""),
    ],
)
def test_keyword_query(text: str, expected: str) -> None:
    assert keyword_query(text) == expected


def test_vector_literal_is_a_pgvector_text_literal() -> None:
    assert vector_literal([1.0, 0.5, -0.25]) == "[1,0.5,-0.25]"


# --------------------------------------------------------------------------- fusion
async def test_vector_match_ranks_first_and_reports_its_rank(corpus) -> None:
    hits = await retriever(corpus, unit(1, 0.2, axis2=0)).search("completely unrelated words zzz")
    assert hits[0].page_title == "Bones"
    assert hits[0].vector_rank == 1 and hits[0].keyword_rank is None


async def test_keyword_only_match_still_surfaces_when_the_vector_is_far(corpus) -> None:
    # The vector points at "Cake", but the keywords only appear in the Whip passage.
    hits = await retriever(corpus, unit(2)).search("abyssal whip attack", k=5)
    titles = [h.page_title for h in hits]
    assert titles[0] in {"Cake", "Whip"} and "Whip" in titles
    whip = next(h for h in hits if h.page_title == "Whip")
    assert whip.keyword_rank == 1


async def test_a_passage_found_by_both_lists_beats_one_found_by_either(corpus) -> None:
    hits = await retriever(corpus, unit(0, 0.5, axis2=1)).search("abyssal whip", k=5)
    assert hits[0].page_title == "Whip"
    assert hits[0].vector_rank == 1 and hits[0].keyword_rank == 1
    assert hits[0].score > hits[1].score
    assert hits[0].score == pytest.approx(2 / 61)  # 1/(60+1) from each list


async def test_english_rewrite_drives_the_keywords_while_the_vector_uses_the_original(
    corpus,
) -> None:
    # Spanish original: the english tsvector cannot match it, but the rewrite can.
    hits = await retriever(corpus, unit(3)).search(
        "¿cuánta experiencia de prayer da dragon bones?",
        keyword_text="dragon bones prayer experience",
    )
    by_title = {h.page_title: h for h in hits}
    assert by_title["Bones"].keyword_rank == 1
    assert by_title["Lobster"].vector_rank == 1


async def test_stopword_only_queries_fall_back_to_the_vector_side(corpus) -> None:
    hits = await retriever(corpus, unit(2)).search("what is the of")
    assert hits and hits[0].page_title == "Cake" and all(h.keyword_rank is None for h in hits)


async def test_k_limits_the_result_and_scores_are_sorted(corpus) -> None:
    hits = await retriever(corpus, unit(0)).search("heals hitpoints whip", k=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


async def test_default_k_comes_from_settings(corpus) -> None:
    assert len(await retriever(corpus, unit(0), crag_topk=2).search("whip")) == 2


# --------------------------------------------------------------------------- filters
async def test_page_type_filter_applies_to_both_lists(corpus) -> None:
    hits = await retriever(corpus, unit(0)).search("snake boss whip", page_types=["npc"])
    assert [h.page_title for h in hits] == ["Zulrah"]
    both = await retriever(corpus, unit(0)).search("whip", page_types=["npc", "other"], k=10)
    assert {h.page_title for h in both} == {"Whip", "Bones", "Cake", "Lobster", "Zulrah"}


async def test_filtered_vector_search_is_not_starved_by_closer_documents_of_other_types(db) -> None:
    # 150 "other" documents sit closer to the query than the 5 "quest" documents. A naive filtered
    # HNSW scan (ef_search=40) would find none of the quests; iterative scan must keep going.
    with psycopg.connect(db) as conn:
        for i in range(150):
            add(
                conn,
                f"near{i}",
                f"filler text number {i}",
                unit(0, 0.01 * (i + 1), axis2=10 + (i % 50)),
            )
        for i in range(5):
            add(conn, f"far{i}", f"quest passage {i}", unit(200 + i), page_type="quest")
        conn.commit()
    hits = await retriever(db, unit(0)).search("zzz", k=5, page_types=["quest"])
    assert sorted(h.page_title for h in hits) == [f"far{i}" for i in range(5)]


async def test_an_empty_corpus_returns_nothing(db) -> None:
    assert await retriever(db, unit(0)).search("anything") == []


# --------------------------------------------------------------------------- headers and titles
async def test_words_in_the_chunk_header_outrank_the_same_words_in_a_body(db) -> None:
    with psycopg.connect(db) as conn:
        add(conn, "Notes", "Misc\nThe grand exchange tax is mentioned here once.", unit(10))
        add(conn, "Grand Exchange", "Grand Exchange > Fees\nSellers pay a small fee.", unit(11))
        conn.commit()
    hits = await retriever(db, unit(50)).search("grand exchange tax", k=5)
    by_title = {h.page_title: h for h in hits}
    assert by_title["Grand Exchange"].keyword_rank == 1  # header match (weight A) beats body (D)
    assert by_title["Notes"].keyword_rank == 2


@pytest.fixture
def mm2(db):
    """Monkey Madness II: the requirements chunk is a bare list that neither the vector nor the
    keywords of the question reach; only the page name does."""
    with psycopg.connect(db) as conn:
        add(
            conn,
            "Monkey Madness II",
            "Monkey Madness II > Details\n- 69 Slayer\n- 70 Crafting",
            unit(20),
        )
        add(
            conn,
            "Monkey Madness II",
            "Monkey Madness II > Rewards\n80,000 Slayer experience",
            unit(21),
        )
        add(conn, "Monkey Madness I", "Monkey Madness I > Details\n- 20 Attack", unit(22))
        add(conn, "Cake", "A cake heals three hitpoints.", unit(23))
        conn.commit()
    return db, TitleIndex(["Monkey Madness II", "Monkey Madness I", "Cake"])


async def test_the_page_named_in_the_question_contributes_its_chunks_as_a_third_list(mm2) -> None:
    db, titles = mm2
    question = "What do I need for Monkey Madness II?"
    hits = await retriever(db, unit(23), titles).search(question, k=10)
    by_text = {h.content.split("\n")[0]: h for h in hits}
    details = by_text["Monkey Madness II > Details"]
    assert details.title_rank is not None
    assert by_text["Monkey Madness II > Rewards"].title_rank is not None
    assert "Monkey Madness I > Details" not in by_text or (
        by_text["Monkey Madness I > Details"].title_rank is None
    )
    assert next(h for h in hits if h.page_title == "Cake").title_rank is None


async def test_title_chunks_are_ordered_by_distance_to_the_question(mm2) -> None:
    db, titles = mm2
    hits = await retriever(db, unit(21), titles).search("Monkey Madness II", k=10)
    ranks = {h.content.split("\n")[0]: h.title_rank for h in hits if h.title_rank}
    assert ranks == {"Monkey Madness II > Rewards": 1, "Monkey Madness II > Details": 2}


async def test_a_title_hit_is_added_to_the_other_lists_scores(mm2) -> None:
    db, titles = mm2
    hits = await retriever(db, unit(20), titles).search("Monkey Madness II", k=5)
    best = hits[0]
    assert best.content.startswith("Monkey Madness II > Details")
    assert best.vector_rank == 1 and best.title_rank == 1 and best.keyword_rank is not None
    assert best.score == pytest.approx(3 / 61, rel=0.2)  # up to 1/(60+1) from each of 3 lists


async def test_the_caller_can_force_pages_the_question_does_not_name(mm2) -> None:
    db, titles = mm2
    # "what do I need?" names nothing; the workflow knows the quest and insists on its page.
    plain = await retriever(db, unit(23), titles).search("what do I need?", k=10)
    assert all(h.title_rank is None for h in plain)
    forced = await retriever(db, unit(23), titles).search(
        "what do I need?", k=10, titles=["Monkey Madness II"]
    )
    ranked = [h for h in forced if h.title_rank]
    assert {h.content.split("\n")[0] for h in ranked} == {
        "Monkey Madness II > Details",
        "Monkey Madness II > Rewards",
    }
    # forcing works even with no title index at all
    bare = await retriever(db, unit(23)).search(
        "what do I need?", k=10, titles=["Monkey Madness I"]
    )
    assert [h.page_title for h in bare if h.title_rank] == ["Monkey Madness I"]


async def test_without_a_title_index_there_is_no_title_list(mm2) -> None:
    db, _ = mm2
    hits = await retriever(db, unit(20)).search("Monkey Madness II", k=5)
    assert all(h.title_rank is None for h in hits)


async def test_the_title_list_respects_the_page_type_filter(db) -> None:
    with psycopg.connect(db) as conn:
        add(conn, "Zulrah", "Zulrah > Drops\nScales", unit(30), page_type="npc")
        add(conn, "Zulrah", "Money guide text", unit(31), page_type="mmg")
        conn.commit()
    titles = TitleIndex(["Zulrah"])
    hits = await retriever(db, unit(30), titles).search("Zulrah", page_types=["mmg"], k=5)
    assert [h.page_type for h in hits] == ["mmg"] and hits[0].title_rank == 1


async def test_titles_named_only_in_the_english_rewrite_count_too(mm2) -> None:
    db, titles = mm2
    hits = await retriever(db, unit(23), titles).search(
        "¿qué necesito para la segunda?", keyword_text="requirements Monkey Madness II", k=10
    )
    assert any(h.title_rank for h in hits if h.content.startswith("Monkey Madness II"))


async def test_a_boss_named_in_the_question_also_searches_its_strategy_page(db) -> None:
    with psycopg.connect(db) as conn:
        add(
            conn,
            "Vorkath",
            "Vorkath > Drops\n- Superior dragon bones: 2",
            unit(40),
            page_type="npc",
        )
        add(
            conn,
            "Vorkath/Strategies",
            "Vorkath/Strategies > Equipment\nRecommended equipment: dragon hunter crossbow",
            unit(41),
            page_type="npc",
        )
        add(conn, "Cake", "A cake heals three hitpoints.", unit(42))
        conn.commit()
    titles = TitleIndex(["Vorkath", "Vorkath/Strategies", "Cake"])
    # "gear" is not in either title: only the name "Vorkath" is, and it brings the strategy page
    hits = await retriever(db, unit(41), titles).search("what gear do I need for Vorkath?", k=10)
    by_title = {h.page_title: h for h in hits}
    assert by_title["Vorkath/Strategies"].title_rank == 1  # nearest to the question
    assert by_title["Vorkath"].title_rank == 2
    assert by_title["Cake"].title_rank is None
