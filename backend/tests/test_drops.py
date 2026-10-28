"""Inverting the monsters' drop tables into an item's "Drop sources" (app/ingest/drops.py)."""

from __future__ import annotations

import pytest

from app.ingest.chunk import ChunkConfig
from app.ingest.drops import (
    MAX_SOURCES,
    DropRow,
    build_index,
    extract_rows,
    sources_text,
    table_version,
)
from app.ingest.dump import CachedPage
from app.ingest.pipeline import PipelineContext, process_page
from app.ingest.render import Renderer


def page(title: str, wikitext: str, page_type: str = "npc") -> CachedPage:
    return CachedPage(title, 1, 5, "t", wikitext, page_type, "Monsters")


ABYSSAL = """
{{Infobox Monster|name=Abyssal demon}}
==Drops==
===Standard and Catacombs of Kourend===
====100%====
{{DropsTableHead}}
{{DropsLine|name=Bones|quantity=1|rarity=Always}}
{{DropsTableBottom}}
====Tertiary====
{{DropsTableHead}}
{{DropsLine|name=Abyssal whip|quantity=1|rarity=1/512|raritynotes={{NamedRef|drop}}}}
{{DropsLineClue|type=hard|rarity=1/150}}
{{DropsTableBottom}}
===Wilderness Slayer Cave===
====Other====
{{DropsLine|name=Abyssal whip|namenotes={{(m)}}|quantity=1|rarity=1/430}}
{{DropsLine|name=Abyssal whip|namenotes={{(m)}}|quantity=1|rarity=1/430}}
{{DropsLine|name=Coal|quantity=10-20 (noted)|rarity=3/128}}
"""


@pytest.mark.parametrize(
    ("path", "version"),
    [
        (("Drops", "Tertiary"), ""),
        (("Drops", "Wilderness Slayer Cave", "Runes"), "Wilderness Slayer Cave"),
        (("Level 79 drops", "100%"), "Level 79 drops"),  # no "Drops" parent: the first is the table
        (("Drops",), ""),
        ((), ""),
    ],
)
def test_the_table_version_is_the_heading_above_the_kind_of_drop(path, version) -> None:
    assert table_version(path) == version


def test_every_drop_line_becomes_a_row_with_its_table_and_without_duplicates() -> None:
    rows = extract_rows(page("Abyssal demon", ABYSSAL), Renderer())
    whips = [r for r in rows if r.item == "Abyssal whip"]
    assert [(r.version, r.rarity, r.flags) for r in whips] == [
        ("Standard and Catacombs of Kourend", "1/512", ""),
        ("Wilderness Slayer Cave", "1/430", "(members only)"),  # the repeated line is one row
    ]
    assert {r.item for r in rows} == {"Bones", "Abyssal whip", "Clue scroll (hard)", "Coal"}
    assert all(r.monster == "Abyssal demon" for r in rows)


def test_pages_without_drop_tables_cost_nothing() -> None:
    assert extract_rows(page("Cake", "A cake heals {{SCP|Hitpoints|3}}.", "item"), Renderer()) == []


def test_the_index_groups_rows_by_item_across_pages() -> None:
    other = "==Drops==\n{{DropsLine|name=Abyssal whip|quantity=1|rarity=1/1000}}"
    index = build_index([page("Abyssal demon", ABYSSAL), page("Abyssal Sire", other)])
    assert sorted(r.monster for r in index["abyssal whip"]) == [
        "Abyssal Sire",
        "Abyssal demon",
        "Abyssal demon",
    ]


def test_rarity_orders_rows_from_always_down_to_rare() -> None:
    rows = [
        DropRow("M", "X", "1", rarity, "", "") for rarity in ("1/512", "Always", "5/128", "Rare")
    ]
    ordered = sorted(rows, key=lambda r: -r.probability)
    # the colour words are rough: "Rare" counts as about 1/200, a touch above 1/512
    assert [r.rarity for r in ordered] == ["Always", "5/128", "Rare", "1/512"]
    assert DropRow("M", "X", "1", "~1/1,000", "", "").probability == pytest.approx(0.001)
    assert DropRow("M", "X", "1", "Varies", "", "").probability == 0.0


def test_the_sources_section_lists_the_most_likely_droppers_and_caps_the_rest() -> None:
    rows = [
        DropRow(f"Monster {i:02d}", "Dragon pickaxe", "1", f"1/{100 + i}", "", "")
        for i in range(40)
    ]
    rows.append(DropRow("Boss", "Dragon pickaxe", "1", "1/2", "Phase 2", "(members only)"))
    text = sources_text("Dragon pickaxe", rows)
    lines = text.splitlines()
    assert lines[0] == "Monsters that drop Dragon pickaxe, from their drop tables on the wiki:"
    assert lines[1] == "- Boss (Phase 2) (members only): 1, rarity 1/2"
    assert len(lines) == 1 + MAX_SOURCES + 1
    assert lines[-1] == f"(and {41 - MAX_SOURCES} more drop tables)"


def test_an_item_page_gets_a_drop_sources_chunk_under_its_own_heading() -> None:
    index = build_index([page("Abyssal demon", ABYSSAL)])
    ctx = PipelineContext(chunk_config=ChunkConfig(), drop_index=index)
    item = page(
        "Abyssal whip",
        "{{Infobox Item|name=Abyssal whip|members=Yes}}\nThe abyssal whip is a weapon.",
        "item",
    )
    result = process_page(item, ctx)
    assert result is not None
    sources = [c for c in result.chunks if c.section == "Drop sources"]
    assert len(sources) == 1
    assert sources[0].content.startswith(
        "Abyssal whip > Drop sources\nMonsters that drop Abyssal whip"
    )
    assert (
        "- Abyssal demon (Standard and Catacombs of Kourend): 1, rarity 1/512" in sources[0].content
    )
    # monster pages and items nobody drops are left alone
    npc = process_page(page("Abyssal demon", ABYSSAL), ctx)
    assert npc is not None and all(c.section != "Drop sources" for c in npc.chunks)
    cake = process_page(page("Cake", "A cake.", "item"), ctx)
    assert cake is not None and all(c.section != "Drop sources" for c in cake.chunks)
