from __future__ import annotations

import pathlib

import pytest

from app.clients.wiki import WikiPage
from app.ingest.mmg import parse_money_making_guide
from app.ingest.parse import guide_summary, parse_page, split_sections
from app.ingest.render import Renderer

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wiki"


def sections(wikitext: str):
    return split_sections(wikitext, Renderer())


def page(title: str, text: str, revision: int = 7) -> WikiPage:
    return WikiPage(title, 1, revision, "2026-10-01T00:00:00Z", text)


def fixture_page(name: str) -> WikiPage:
    text = (FIXTURES / name).read_text()
    return page(text.split("-->")[0].removeprefix("<!-- title:").strip(), text)


# --------------------------------------------------------------------------- sections
def test_lead_and_heading_paths_follow_the_nesting() -> None:
    out = sections(
        "Intro text.\n==A==\nA body.\n===B===\nB body.\n====C====\nC body.\n==D==\nD body."
    )
    assert [(s.path, s.text) for s in out] == [
        ((), "Intro text."),
        (("A",), "A body."),
        (("A", "B"), "B body."),
        (("A", "B", "C"), "C body."),
        (("D",), "D body."),
    ]


def test_sibling_after_a_deeper_branch_resets_the_stack() -> None:
    out = sections("==A==\n===B===\nx\n===C===\ny\n==D==\n===E===\nz")
    assert [s.path for s in out] == [("A", "B"), ("A", "C"), ("D", "E")]


def test_noise_sections_are_dropped_together_with_their_subsections() -> None:
    out = sections(
        "==Real==\nkeep\n==References==\nref text\n===Sub of refs===\nmore refs\n"
        "==Trivia==\nfun fact\n==Also real==\nkeep too\n===Child===\nkeep three"
    )
    assert [s.path for s in out] == [("Real",), ("Also real",), ("Also real", "Child")]


def test_empty_sections_vanish_and_headings_are_rendered_as_text() -> None:
    out = sections(
        "==Only noise==\n{{Subject changes header}}{{Clear}}\n"
        "==[[Dragon bones|Bones]] and '''more'''==\ntext"
    )
    assert [(s.path, s.text) for s in out] == [(("Bones and more",), "text")]


def test_section_bodies_are_rendered() -> None:
    out = sections("==Reqs==\n* {{SCP|Fishing|43}}\n* {{Coins|1000}}\n")
    assert out[0].text == "- 43 Fishing\n- 1000 coins"


def test_multiline_parser_functions_do_not_break_section_splitting() -> None:
    wikitext = "==A==\n{{#expr:\n1 +\n2}} done\n==B==\nafter"
    assert [s.path for s in sections(wikitext)] == [("A",), ("B",)]


# --------------------------------------------------------------------------- pages
def test_redirects_and_empty_pages_are_skipped() -> None:
    assert parse_page(page("X", "#REDIRECT [[Y]]"), "other") is None
    assert parse_page(page("X", "{{Subject changes header}}"), "other") is None


def test_parse_page_carries_title_url_type_and_revision() -> None:
    parsed = parse_page(page("Cook's Assistant", "Text.", revision=42), "quest")
    assert parsed is not None
    assert (parsed.title, parsed.page_type, parsed.revision_id) == ("Cook's Assistant", "quest", 42)
    assert parsed.url == "https://oldschool.runescape.wiki/w/Cook's_Assistant"


def test_real_quest_page_has_infobox_details_and_walkthrough_sections() -> None:
    parsed = parse_page(fixture_page("quest_cooks_assistant.wiki"), "quest")
    assert parsed is not None
    paths = [s.path for s in parsed.sections]
    assert () in paths and ("Details",) in paths and ("Walkthrough",) in paths
    lead = parsed.sections[0].text
    assert lead.startswith("[Quest] Cook's Assistant")
    details = next(s.text for s in parsed.sections if s.path == ("Details",))
    assert "[Quest details]" in details and "Difficulty: Novice" in details


def test_real_boss_page_drops_noise_sections() -> None:
    parsed = parse_page(fixture_page("monster_zulrah.wiki"), "npc")
    assert parsed is not None
    assert all("References" not in s.path and "Trivia" not in s.path for s in parsed.sections)


# --------------------------------------------------------------------------- money making guides
def test_guide_pages_get_a_price_free_summary_section_first() -> None:
    parsed = parse_page(fixture_page("mmg_plain_per_kill.wiki"), "mmg")
    assert parsed is not None and parsed.guide is not None
    summary = parsed.sections[0]
    assert summary.path == ()
    text = summary.text
    assert text.startswith("Money making guide: Aerial fishing")
    for fact in (
        "Category: Skilling",
        "Intensity: High",
        "Members only: yes",
        "43 Fishing",
        "Rate: 2,200 (Creatures caught per hour)",
        "Inputs (per hour):",
        "- 629 Fish offcuts",
        "Outputs (per hour):",
        "- 4,400 Shark lure",
    ):
        assert fact in text, fact
    assert "Aerial fishing is an activity in Lake Molch" in text  # the `Details` prose is kept
    assert "profit" not in text.lower().split("details")[0]  # no profit figures in the facts block


def test_recurring_guide_summary_uses_per_instance_amounts() -> None:
    guide = parse_money_making_guide(
        "Money making guide/Bird house trapping", (FIXTURES / "mmg_recurring.wiki").read_text()
    )
    assert guide is not None
    text = guide_summary(guide)
    assert "Takes 1.4 minutes per instance; can be repeated every 50 minutes" in text
    assert "Inputs (per instance):" in text and "- 4 Redwood logs" in text


def test_summary_formatting_helpers() -> None:
    guide = parse_money_making_guide(
        "Money making guide/T",
        "{{Mmgtable\n|Activity = T\n|Members = No\n|isperkill = y\n|kph = 10\n"
        "|Input1 = {{Cheap food}}\n|Input1num = 1.5\n|Output1 = A\n|Output1num = 0.001953\n"
        "|Experience1 = Fishing\n|Experience1num = 12345\n}}",
    )
    assert guide is not None
    text = guide_summary(guide, details="Strategy text.")
    assert "Members only: no" in text
    assert (
        "- 15 Cheap food (cheapest of Manta ray, Tuna potato, Anglerfish, Dark crab, Bluefin)"
        in text
    )
    assert "- 0.01953 A" in text  # small amounts keep significant digits
    assert "Experience: 123,450 Fishing" in text
    assert text.endswith("\n\nStrategy text.")


@pytest.mark.parametrize("page_type", ["quest", "npc", "item", "skill_guide", "other"])
def test_only_mmg_pages_attempt_guide_parsing(page_type: str) -> None:
    # Outside the guide category the table is just a dropped template: nothing is rendered.
    parsed = parse_page(fixture_page("mmg_plain_per_kill.wiki"), page_type)
    assert parsed is None or parsed.guide is None


def test_a_bare_reuse_of_a_note_before_its_definition_shows_the_text() -> None:
    wikitext = (
        "==Drops==\n"
        '{{DropsLine|name=Coins|quantity=5|rarity=35/128|raritynotes=<ref group="d" name="f2"/>}}\n'
        '{{DropsLine|name=Coins|quantity=10|rarity=5/128|raritynotes=<ref group="d" name="f2">'
        "Only dropped in free-to-play</ref>}}\n"
    )
    sections = split_sections(wikitext, Renderer())
    (drops,) = [s for s in sections if s.path == ("Drops",)]
    lines = drops.text.splitlines()
    assert lines[0] == "- Coins: 5, rarity 35/128 (note: Only dropped in free-to-play)"
    assert lines[1] == "- Coins: 10, rarity 5/128"  # spelled out once per section
