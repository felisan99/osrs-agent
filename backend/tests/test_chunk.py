from __future__ import annotations

import pathlib
import re
from itertools import pairwise

import pytest

from app.clients.wiki import WikiPage
from app.ingest.chunk import (
    ChunkConfig,
    build_quest_matcher,
    chunk_page,
    estimate_tokens,
    mentioned_skills,
)
from app.ingest.parse import ParsedPage, Section, parse_page

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wiki"


def words(text: str) -> int:
    return len(text.split())


def para(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


def parsed(sections: list[Section], title: str = "Page", page_type: str = "other") -> ParsedPage:
    return ParsedPage(title, "https://example.org/Page", page_type, 9, sections)


def chunks_of(sections, cfg=None, **kwargs):
    cfg = cfg or ChunkConfig(min_tokens=50, max_tokens=100, overlap_tokens=10)
    return chunk_page(parsed(sections), cfg, count=words, **kwargs)


# --------------------------------------------------------------------------- packing
def test_a_small_page_is_one_chunk_with_header_and_metadata() -> None:
    (chunk,) = chunks_of([Section((), "Short intro.")])
    assert chunk.content == "Page\nShort intro."
    assert chunk.page_title == "Page" and chunk.page_type == "other"
    assert chunk.metadata["chunk_index"] == 0 and chunk.metadata["revision_id"] == 9
    assert chunk.metadata["url"] == "https://example.org/Page"


def test_top_level_sections_never_share_a_chunk_however_small() -> None:
    lead, drops, locs = chunks_of(
        [
            Section((), "Moss giants are large."),
            Section(("Drops",), "- Mossy key: 1, rarity 1/150"),
            Section(("Locations",), "- Crandor"),
        ]
    )
    assert lead.content == "Page\nMoss giants are large."
    assert (
        drops.content == "Page > Drops\n- Mossy key: 1, rarity 1/150" and drops.section == "Drops"
    )
    assert locs.content == "Page > Locations\n- Crandor"


def test_header_carries_the_section_path_of_the_first_section() -> None:
    chunk = chunks_of([Section(("Training", "Methods"), "Body text.")])[0]
    assert chunk.content.startswith("Page > Training > Methods\nBody text.")
    assert chunk.section == "Training > Methods"


def test_sub_sections_of_one_parent_merge_under_the_parent_header() -> None:
    secs = [Section(("Drops", f"S{i}"), para(30, f"s{i}_")) for i in range(6)]  # 180 words
    out = chunks_of(secs)
    assert len(out) >= 2
    assert all(c.tokens <= 100 for c in out)
    # the header is the path the whole chunk shares; each sub-section keeps a relative label
    assert all(c.content.startswith("Page > Drops\n") and c.section == "Drops" for c in out)
    assert "[S0]\ns0_0" in out[0].content and "[S1]" in out[0].content
    assert "[Drops" not in out[0].content  # labels are relative to the header


def test_a_chunk_inside_one_leaf_section_names_the_leaf_in_its_header() -> None:
    (chunk,) = chunks_of([Section(("Drops", "Tertiary"), "- Mossy key: 1, rarity 1/150")])
    assert chunk.content == "Page > Drops > Tertiary\n- Mossy key: 1, rarity 1/150"
    assert chunk.section == "Drops > Tertiary"


def test_a_chunk_whose_sub_sections_diverge_is_headed_by_their_common_parent() -> None:
    secs = [
        Section(("Drops", "100%"), "- Big bones: 1, rarity Always"),
        Section(("Drops", "Tertiary"), "- Mossy key: 1, rarity 1/150"),
    ]
    (chunk,) = chunks_of(secs)
    assert chunk.section == "Drops"
    assert chunk.content == (
        "Page > Drops\n[100%]\n- Big bones: 1, rarity Always\n\n"
        "[Tertiary]\n- Mossy key: 1, rarity 1/150"
    )


def test_a_section_boundary_ends_the_chunk_once_the_minimum_is_reached() -> None:
    out = chunks_of([Section(("A",), para(60, "a")), Section(("B",), para(30, "b"))])
    assert len(out) == 2 and out[0].section == "A" and out[1].section == "B"
    assert "b0" not in out[0].content


def test_long_sections_split_at_paragraphs_without_losing_text() -> None:
    paragraphs = [para(40, f"p{i}_") for i in range(7)]
    out = chunks_of([Section(("Long",), "\n\n".join(paragraphs))])
    assert len(out) >= 3 and all(c.tokens <= 100 for c in out)
    joined = " ".join(c.content for c in out)
    for i in range(7):
        assert f"p{i}_0" in joined and f"p{i}_39" in joined
    assert all(c.section == "Long" for c in out)


def test_lists_and_tables_split_by_row_never_mid_line() -> None:
    rows = [f"- Item {i}: quantity {i}, rarity 1/{i + 2}" for i in range(80)]
    out = chunks_of([Section(("Drops",), "\n".join(rows))])
    assert len(out) >= 3
    body_lines = [line for c in out for line in c.content.split("\n")[1:] if line]
    assert all(c.content.startswith("Page > Drops\n") for c in out)  # each piece says what it is
    assert set(rows) <= set(body_lines)  # every row appears intact
    assert all(re.fullmatch(r"- Item \d+: quantity \d+, rarity 1/\d+", line) for line in body_lines)


def test_a_giant_paragraph_splits_by_sentence_with_overlap() -> None:
    sentences = [f"Sentence number {i} says something useful here." for i in range(60)]
    out = chunks_of([Section((), " ".join(sentences))])
    assert len(out) >= 3 and all(c.tokens <= 100 for c in out)
    # one sentence of overlap: the last sentence of a chunk opens the next one
    for first, second in pairwise(out):
        last = first.content.rstrip().split(". ")[-1]
        assert last.rstrip(".") in second.content


def test_a_runaway_line_without_punctuation_is_split_by_words() -> None:
    out = chunks_of([Section((), "word " * 500)])
    assert len(out) >= 5 and all(c.tokens <= 100 for c in out)


def test_a_tiny_trailing_chunk_is_merged_back() -> None:
    cfg = ChunkConfig(min_tokens=80, max_tokens=100, overlap_tokens=0)
    # x and y fill one chunk (87 of 90); the 8-word remainder would be a runt, so it is merged back
    body = "\n\n".join([para(45, "x"), para(40, "y"), para(8, "z")])
    out = chunk_page(parsed([Section((), body)]), cfg, count=words)
    assert len(out) == 1 and "z7" in out[0].content


def test_chunk_indexes_are_sequential_and_output_is_deterministic() -> None:
    secs = [Section((f"S{i}",), para(70, f"t{i}_")) for i in range(5)]
    first, second = chunks_of(secs), chunks_of(secs)
    assert [c.content for c in first] == [c.content for c in second]
    assert [c.metadata["chunk_index"] for c in first] == list(range(len(first)))


def test_header_and_labels_are_inside_the_budget() -> None:
    long_title = "A very long page title " * 3
    secs = [Section((f"Section number {i}",), para(40, f"u{i}_")) for i in range(8)]
    out = chunk_page(parsed(secs, title=long_title), ChunkConfig(50, 100, 10), count=words)
    assert all(c.tokens <= 100 for c in out), [c.tokens for c in out]


def test_pages_without_sections_yield_no_chunks() -> None:
    assert chunk_page(parsed([]), ChunkConfig(), count=words) == []


# --------------------------------------------------------------------------- metadata
def test_mentioned_skills_need_a_level_context_and_wiki_capitalisation() -> None:
    text = (
        "Requires 70 Attack and 43+ Prayer; Slayer level is also needed. "
        "Use the special attack. level 50 Magic"
    )
    assert mentioned_skills(text) == ["Attack", "Magic", "Prayer", "Slayer"]
    assert mentioned_skills("the special attack hits hard; a prayer of thanks") == []


def test_quest_matcher_finds_whole_titles_only() -> None:
    match = build_quest_matcher(
        {"Dragon Slayer I", "Dragon Slayer II", "Rune Mysteries", "Lost City"}
    )
    text = "Finish Dragon Slayer II, then Rune Mysteries. Not Dragon Slayer III or Lost Cities."
    assert match(text) == ["Dragon Slayer II", "Rune Mysteries"]
    assert build_quest_matcher(set())("anything") == []


def test_chunks_carry_skill_and_quest_metadata() -> None:
    out = chunks_of(
        [Section((), "Needs 75 Magic before Dragon Slayer II and 60 Agility.")],
        quest_matcher=build_quest_matcher({"Dragon Slayer II"}),
    )
    assert out[0].metadata["skills"] == ["Agility", "Magic"]
    assert out[0].metadata["quests"] == ["Dragon Slayer II"]


def test_default_token_estimate_never_underestimates_a_sum_of_pieces() -> None:
    pieces = ["a" * n for n in (1, 2, 7, 11, 36, 37, 100)]
    assert sum(estimate_tokens(p) for p in pieces) >= estimate_tokens("\n".join(pieces)) - len(
        pieces
    )
    assert estimate_tokens("") == 0 and estimate_tokens("x") == 1


# --------------------------------------------------------------------------- real pages
REAL = [
    ("quest_dragon_slayer_ii_head.wiki", "quest"),
    ("quest_cooks_assistant.wiki", "quest"),
    ("monster_zulrah.wiki", "npc"),
    ("guide_prayer_training.wiki", "skill_guide"),
    ("item_abyssal_whip.wiki", "item"),
    ("mmg_plain_per_kill.wiki", "mmg"),
    ("mechanic_experience.wiki", "other"),
]


@pytest.mark.parametrize(("name", "page_type"), REAL)
def test_real_pages_chunk_within_the_embedding_limit(name: str, page_type: str) -> None:
    text = (FIXTURES / name).read_text()
    title = text.split("-->")[0].removeprefix("<!-- title:").strip()
    page = parse_page(WikiPage(title, 1, 5, "t", text), page_type)
    assert page is not None
    cfg = ChunkConfig()
    out = chunk_page(page, cfg)
    assert out and all(0 < c.tokens <= cfg.max_tokens for c in out), [c.tokens for c in out]
    assert all(c.content.startswith(title) for c in out)
    assert all(c.metadata["page_type"] == page_type for c in out)


def test_dragon_slayer_requirements_chunk_mentions_the_skills_and_quests() -> None:
    text = (FIXTURES / "quest_dragon_slayer_ii_head.wiki").read_text()
    page = parse_page(WikiPage("Dragon Slayer II", 1, 5, "t", text), "quest")
    assert page is not None
    out = chunk_page(
        page, ChunkConfig(), quest_matcher=build_quest_matcher({"Legends' Quest", "Dream Mentor"})
    )
    skills = {s for c in out for s in c.metadata["skills"]}
    assert {"Magic", "Smithing", "Mining", "Crafting"} <= skills
    assert {q for c in out for q in c.metadata["quests"]} == {"Legends' Quest", "Dream Mentor"}


def test_a_list_that_continues_repeats_the_line_that_names_it() -> None:
    rows = [f"- Item {i}: quantity {i}, rarity 1/{i + 2}" for i in range(80)]
    block = "Items required:\n" + "\n".join(rows)
    out = chunks_of([Section(("Details",), block)])
    assert len(out) >= 3
    for chunk in out:
        assert chunk.content.split("\n")[1] == "Items required:"  # first line after the header
    joined = "\n".join(chunk.content for chunk in out)
    assert all(row in joined for row in rows)
    assert all(chunk.tokens <= 100 for chunk in out)


def test_a_table_that_continues_repeats_its_column_names() -> None:
    header = "Level | XP per tome | Tomes needed"
    rows = [f"{i} | {i * 5} | {i + 3}" for i in range(1, 90)]
    out = chunks_of([Section(("Strategy",), "Runecrafting\n" + header + "\n" + "\n".join(rows))])
    assert len(out) >= 3
    for chunk in out:
        assert header in chunk.content.split("\n")
    assert all(chunk.tokens <= 100 for chunk in out)


def test_prose_is_not_given_a_repeated_context_line() -> None:
    sentences = [f"Sentence number {i} says something useful here." for i in range(60)]
    out = chunks_of([Section((), "Note: " + " ".join(sentences))])
    assert all("Note:" not in c.content for c in out[1:])


def test_the_repeated_context_never_pushes_a_piece_over_the_budget() -> None:
    # rows nearly as long as the budget behind a long label: label + row would not fit, so the
    # label is left out of the continuation (the embedding input limit is a hard one)
    label = "Items required for this part of the quest are listed below in order:"
    rows = [f"- Row {i}: " + " ".join(f"w{i}_{j}" for j in range(84)) for i in range(8)]
    out = chunks_of([Section(("Details",), label + "\n" + "\n".join(rows))])
    assert len(out) >= 5
    assert all(chunk.tokens <= 100 for chunk in out), [chunk.tokens for chunk in out]
    assert all(row in "\n".join(c.content for c in out) for row in rows)
