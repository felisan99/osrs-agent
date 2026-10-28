"""Matching page names inside questions (app/rag/titles.py)."""

from __future__ import annotations

import psycopg
import pytest

from app.rag.titles import MAX_TITLES, TitleIndex, singular, tokens

TITLES = [
    "Monkey Madness I",
    "Monkey Madness II",
    "Monkey",
    "Cook's Assistant",
    "Abyssal whip",
    "Abyssal demon",
    "Gargoyle",
    "Dragon bones",
    "Desert Treasure II - The Fallen Empire",
    "Grand Exchange",
    "Combat",  # generic single word: never indexed
    "Nex",  # too short to be distinctive on its own
    "Prayer",
]


@pytest.fixture
def index() -> TitleIndex:
    return TitleIndex(TITLES)


def test_tokens_fold_accents_apostrophes_and_punctuation() -> None:
    assert tokens("Cook's Assistant") == ["cooks", "assistant"]
    assert tokens("¿Cómo empiezo la quest Cook\u2019s Assistant?") == [
        "como", "empiezo", "la", "quest", "cooks", "assistant",
    ]  # fmt: skip
    assert tokens("Desert Treasure II - The Fallen Empire") == [
        "desert", "treasure", "ii", "the", "fallen", "empire",
    ]  # fmt: skip
    assert tokens("") == []


def test_singular() -> None:
    assert [singular(w) for w in ("gargoyles", "harpies", "bones", "grass", "is", "bus")] == [
        "gargoyle", "harpy", "bone", "grass", "is", "bus",
    ]  # fmt: skip


def test_a_named_page_is_found_whatever_the_case_or_apostrophes(index: TitleIndex) -> None:
    assert index.match("how do i start cooks assistant?") == ["Cook's Assistant"]
    assert index.match("ABYSSAL WHIP requirements") == ["Abyssal whip"]


def test_the_longest_name_wins_and_consumed_words_cannot_match_again(index: TitleIndex) -> None:
    assert index.match("What do I need for Monkey Madness II?") == ["Monkey Madness II"]
    assert index.match("Monkey Madness I walkthrough") == ["Monkey Madness I"]
    assert index.match("a monkey") == ["Monkey"]


def test_digits_and_roman_numerals_are_interchangeable(index: TitleIndex) -> None:
    assert tokens("Dragon Slayer 2") == tokens("Dragon Slayer II") == ["dragon", "slayer", "ii"]
    assert index.match("how do i start monkey madness 2?") == ["Monkey Madness II"]
    assert index.match("monkey madness 1 walkthrough") == ["Monkey Madness I"]
    assert tokens("Sir Prysin 25 gp") == ["sir", "prysin", "25", "gp"]  # only 1-10 are folded


def test_long_quest_names_are_matched_in_full(index: TitleIndex) -> None:
    question = "Can I do Desert Treasure II - The Fallen Empire yet?"
    assert index.match(question) == ["Desert Treasure II - The Fallen Empire"]


def test_plural_forms_match_their_singular_title(index: TitleIndex) -> None:
    assert index.match("where do gargoyles spawn?") == ["Gargoyle"]
    assert index.match("how much xp do dragon bones give") == ["Dragon bones"]  # exact form first
    assert index.match("kill abyssal demons") == ["Abyssal demon"]


def test_spanish_text_with_english_names_still_matches(index: TitleIndex) -> None:
    assert index.match("¿Qué nivel necesito para la Abyssal whip?") == ["Abyssal whip"]
    assert index.match("¿dónde están las gárgolas?") == []  # "gárgolas" is not an English title


def test_generic_and_short_single_words_are_not_indexed(index: TitleIndex) -> None:
    assert index.match("what is the best combat style") == []
    assert index.match("how do I kill nex") == []
    assert index.match("how do I train prayer") == ["Prayer"]


def test_several_texts_are_matched_together_without_duplicates(index: TitleIndex) -> None:
    assert index.match("¿requisitos de Monkey Madness II?", "Monkey Madness II requirements") == [
        "Monkey Madness II"
    ]
    assert index.match("la whip", "Abyssal whip attack level") == ["Abyssal whip"]


def test_the_number_of_matched_titles_is_capped() -> None:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
    names = [f"Quest {w}" for w in [*words, "juliet"]]
    many = TitleIndex(names)
    assert len(many.match(" ".join(names))) == MAX_TITLES


def test_case_variants_of_one_title_are_all_returned() -> None:
    both = TitleIndex(["Slayer Master", "Slayer master"])
    assert both.match("slayer master tasks") == ["Slayer Master", "Slayer master"]


def test_empty_and_blank_inputs() -> None:
    assert TitleIndex([]).match("anything") == []
    assert TitleIndex(["", "---"]).match("anything") == [] and len(TitleIndex(["", "---"])) == 0


async def test_load_indexes_every_distinct_title_in_the_documents_table(db: str) -> None:
    from contextlib import asynccontextmanager

    zeros = "[" + ",".join("0" for _ in range(384)) + "]"
    with psycopg.connect(db) as conn:
        for title in ("Zulrah", "Zulrah", "Abyssal whip"):
            conn.execute(
                "insert into documents (page_title, section, page_type, url, content, metadata, "
                "embedding) values (%s, '', 'npc', 'https://x', 'text', '{}', %s::vector)",
                (title, zeros),
            )
        conn.commit()

    @asynccontextmanager
    async def connect():
        async with await psycopg.AsyncConnection.connect(db) as conn:
            yield conn

    loaded = await TitleIndex.load(connect)
    assert len(loaded) == 2 and loaded.match("zulrah strategy") == ["Zulrah"]


def test_a_named_page_brings_its_strategy_page_along() -> None:
    index = TitleIndex(["Vorkath", "Vorkath/Strategies", "Vorkath/Quick guide", "Zulrah"])
    assert index.match("what gear for vorkath?") == ["Vorkath"]  # matching itself is unchanged
    assert index.guide_pages("Vorkath") == ["Vorkath/Strategies"]
    assert index.guide_pages("Zulrah") == []
    assert index.guide_pages("Nobody") == []
