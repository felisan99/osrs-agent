"""Principle 1 as a check: big numbers in an answer need a source (app/graph/grounding.py)."""

from __future__ import annotations

import pytest

from app.graph.grounding import (
    drop_sentences_with,
    numbers_in,
    profit_rate_mentions,
    ungrounded_numbers,
)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1,500,000", 1_500_000),
        ("1.500.000", 1_500_000),
        ("30,000", 30_000),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("1.5M", 1_500_000),
        ("1,5 M", 1_500_000),
        ("2.3 millones", 2_300_000),
        ("300k", 300_000),
        ("12 mil", 12_000),
        ("0.125", 0.125),
        ("117600", 117_600),
        ("1195000.0", 1_195_000),
        ("4 bn", 4e9),
    ],
)
def test_numbers_are_read_in_the_formats_players_use(text: str, value: float) -> None:
    ((_, parsed, _),) = numbers_in(text)
    assert parsed == pytest.approx(value)


def test_a_number_inside_a_word_or_an_id_is_not_split() -> None:
    assert [w for w, _, _ in numbers_in("Prayer potion(4) costs 9,500")] == ["4", "9,500"]
    assert numbers_in("abc123") == []


TOOL = '{"profit_per_hour":2961443,"unit_price":30228.48,"tax_per_unit":30000}'


def test_figures_that_come_from_the_tool_are_grounded_as_written_or_rounded() -> None:
    answer = (
        "Killing Callisto pays 2,961,443 gp/h; that's about 2.96M. Each unit pays 30,000 in tax "
        "and a seed costs 30.2k."
    )
    assert ungrounded_numbers(answer, [TOOL]) == []


def test_a_number_nobody_supplied_is_flagged() -> None:
    answer = "You can make 3,500,000 gp per hour, or 450k on a bad day."
    assert ungrounded_numbers(answer, [TOOL]) == ["3,500,000", "450k"]


def test_the_players_own_numbers_and_the_wiki_passages_count_as_sources() -> None:
    answer = "With 10M gp you can buy 8,000 of them; the cap is 5,000,000 per item."
    sources = [TOOL, "I have 10M gp", "The tax is capped at 5,000,000 coins per item. Limit 8,000"]
    assert ungrounded_numbers(answer, sources) == []


def test_derived_figures_are_not_grounded() -> None:
    # 3 x 2,961,443 is arithmetic the model did itself: exactly what principle 1 forbids.
    assert ungrounded_numbers("Three hours is 8,884,329 gp.", [TOOL]) == ["8,884,329"]


def test_rounding_tolerance_is_tight_for_exact_figures_and_looser_for_abbreviated_ones() -> None:
    assert ungrounded_numbers("It pays 2,990,000.", [TOOL]) == ["2,990,000"]  # 1% off, as written
    assert ungrounded_numbers("It pays about 2.99M.", [TOOL]) == []  # abbreviated: 1% is rounding
    assert ungrounded_numbers("It pays about 3.5M.", [TOOL]) == ["3.5M"]


def test_small_numbers_and_years_are_not_checked() -> None:
    assert ungrounded_numbers("You need 70 Attack, 3 quests and 1.5 hours in 2026.", []) == []
    assert ungrounded_numbers("In 2026 it costs 2026 gp", []) == []  # a bare 2026 reads as a year


def test_spanish_formats_are_understood() -> None:
    assert ungrounded_numbers("Gana 2.961.443 gp por hora (casi 3 millones).", [TOOL]) == []
    assert ungrounded_numbers("Gana 5.000.000 gp por hora.", [TOOL]) == ["5.000.000"]


def test_bad_sentences_can_be_dropped_as_a_last_resort() -> None:
    answer = (
        "Killing Callisto pays 2,961,443 gp/h. Selling 50 of them gives 9,999,999 gp! "
        "Check the wiki.\nSecond line mentions 9,999,999 too."
    )
    assert drop_sentences_with(answer, ["9,999,999"]) == (
        "Killing Callisto pays 2,961,443 gp/h. Check the wiki."
    )
    assert drop_sentences_with("Only 9,999,999 here.", ["9,999,999"]) == ""


@pytest.mark.parametrize(
    "text",
    [
        "Gives about 30,000 GP/hora",
        "makes 1.2M gp per hour",
        "unos 900k por hora",
        "127,000 gp/hr with rogue gear",
        "280000 coins an hour",
    ],
)
def test_rates_per_hour_are_recognised_in_both_languages(text: str) -> None:
    assert profit_rate_mentions(text), text


@pytest.mark.parametrize(
    "text",
    [
        "You need 70 Attack and 3 quests",
        "It takes about 2 hours to finish",
        "Costs 1,500,000 gp",
        "Kill 30 monsters per task",
    ],
)
def test_other_numbers_are_not_rates(text: str) -> None:
    assert profit_rate_mentions(text) == []
