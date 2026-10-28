"""Answer tidying for a plain-text panel (app/graph/format.py)."""

from __future__ import annotations

from app.graph.format import clean_answer
from app.rag.retrieve import Passage


def passage(title: str) -> Passage:
    return Passage(
        id=1,
        page_title=title,
        section="",
        page_type="other",
        url="u",
        content="c",
        metadata={},
        score=0,
    )


def test_a_two_column_table_becomes_a_list_of_label_value_lines() -> None:
    text = (
        "**Dragon bones**\n\n| Concepto | Valor |\n|---|---|\n| Precio de compra | 3 841 gp |\n"
        "| Margen | 1 gp |\n\nSolo 1 gp de margen."
    )
    assert clean_answer(text) == (
        "**Dragon bones**\n\n- Precio de compra: 3 841 gp\n- Margen: 1 gp\n\nSolo 1 gp de margen."
    )


def test_a_wider_table_keeps_the_column_names() -> None:
    text = "| Item | Profit | Limit |\n| --- | ---: | :--- |\n| Whip | 100 | 70 |\n| Bow | 5 | 8 |"
    assert clean_answer(text) == "- Whip (Profit: 100; Limit: 70)\n- Bow (Profit: 5; Limit: 8)"


def test_headings_become_bold_and_horizontal_rules_disappear() -> None:
    text = "## Top picks\n\n---\n1. First\n***\n### Notes ###\nDone"
    assert clean_answer(text) == "**Top picks**\n\n1. First\n**Notes**\nDone"


def test_a_lone_pipe_line_is_not_mistaken_for_a_table() -> None:
    assert clean_answer("| just one row |") == "just one row"


def test_passage_numbers_become_page_names_and_runs_are_merged() -> None:
    passages = [passage("Abyssal whip"), passage("Abyssal whip"), passage("Slayer")]
    text = "Needs 70 Attack [1][2]. Slayer 85 [3], see also [1] and [2] [3]."
    assert clean_answer(text, passages) == (
        "Needs 70 Attack [Abyssal whip]. Slayer 85 [Slayer], see also [Abyssal whip] and "
        "[Abyssal whip][Slayer]."
    )


def test_numbers_that_are_not_passage_references_are_left_alone() -> None:
    passages = [passage("Abyssal whip")]
    assert clean_answer("The array [7] and level [1].", passages) == (
        "The array [7] and level [Abyssal whip]."
    )
    assert clean_answer("See [1].") == "See [1]."  # no passages: nothing to resolve against


def test_plain_text_and_blank_lines_are_preserved() -> None:
    assert clean_answer("One.\n\n\n\nTwo.  \n") == "One.\n\nTwo."
