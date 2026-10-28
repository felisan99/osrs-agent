"""The evaluation set (SPEC §12) is well formed, and its gold labels can be checked."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import psycopg
import pytest
from pydantic import ValidationError

from app.graph.intents import Intent
from eval.cases import Case, check_gold, load_cases

DIM = 384


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def minimal(**overrides) -> dict:
    case = {
        "id": "c1",
        "lang": "en",
        "question": "q?",
        "intent_esperado": "wiki_general",
        "corpus": "n/a",
        "behavior": "answer",
    }
    return case | overrides


# --------------------------------------------------------------------------- the shipped file
def test_the_shipped_set_has_30_to_50_cases_covering_every_category_the_spec_asks_for() -> None:
    cases = load_cases()
    assert 30 <= len(cases) <= 50
    assert {c.intent_esperado for c in cases} == set(Intent)
    assert {c.lang for c in cases} == {"en", "es"}
    behaviors = {c.behavior for c in cases}
    assert behaviors == {"answer", "no_evidence", "clarify", "refuse"}
    assert any(c.history for c in cases)  # multi-turn follow-ups
    assert any(c.player_state for c in cases)  # player-dependent questions
    assert any(c.corpus == "unanswerable" for c in cases)


def test_ids_are_unique_and_profiles_are_resolved_into_player_states() -> None:
    cases = load_cases()
    assert len({c.id for c in cases}) == len(cases)
    mm2 = next(c for c in cases if c.id == "quest-mm2-eligibility-es")
    assert mm2.player_state is not None
    assert mm2.player_state["skills"]["crafting"] == 62
    assert mm2.player_state["quests"]["Monkey Madness I"] == "FINISHED"


def test_every_profile_follows_the_player_state_contract() -> None:
    skills = {
        "attack", "defence", "strength", "hitpoints", "ranged", "prayer", "magic", "cooking",
        "woodcutting", "fletching", "fishing", "firemaking", "crafting", "smithing", "mining",
        "herblore", "agility", "thieving", "slayer", "farming", "runecraft", "hunter",
        "construction", "sailing",
    }  # fmt: skip
    states = [c.player_state for c in load_cases() if c.player_state]
    assert states
    for state in states:
        assert state["account_type"] in {
            "main", "ironman", "hardcore_ironman", "ultimate_ironman", "group_ironman", "unknown"
        }  # fmt: skip
        assert set(state["skills"]) == skills
        assert set(state["quests"].values()) <= {"NOT_STARTED", "IN_PROGRESS", "FINISHED"}


def test_avoid_patterns_catch_the_numbers_they_are_meant_to_catch() -> None:
    ironman = next(c for c in load_cases() if c.id == "mm-ironman-es")
    pattern = re.compile(ironman.avoid_patterns[0])
    assert pattern.search("Podrías ganar 1.200.000 gp por hora con esto")
    assert pattern.search("unos 900k GP/h")
    assert not pattern.search("Para un ironman el beneficio en GP del GE no aplica")


# --------------------------------------------------------------------------- loader strictness
def test_unknown_keys_intents_and_tools_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(minimal(surprise=1))
    with pytest.raises(ValidationError):
        Case.model_validate(minimal(intent_esperado="smalltalk"))
    with pytest.raises(ValidationError, match="unknown tools"):
        Case.model_validate(minimal(tools_expected=["calc_everything"]))
    with pytest.raises(ValidationError):
        Case.model_validate(minimal(avoid_patterns=["(unclosed"]))


def test_label_consistency_rules() -> None:
    with pytest.raises(ValidationError, match="needs gold_pages and evidence"):
        Case.model_validate(minimal(corpus="answerable"))
    with pytest.raises(ValidationError, match="only answerable"):
        Case.model_validate(minimal(gold_pages=["Zulrah"], evidence=["x"]))
    with pytest.raises(ValidationError, match="plain answer"):
        Case.model_validate(minimal(corpus="unanswerable", behavior="answer"))
    with pytest.raises(ValidationError, match="out_of_domain"):
        Case.model_validate(minimal(behavior="refuse"))


def test_a_profile_must_exist_and_cannot_be_combined_with_an_inline_state(tmp_path: Path) -> None:
    missing = write(
        tmp_path,
        """
        profiles: {}
        cases:
          - {id: a, lang: en, question: q, intent_esperado: wiki_general, corpus: n/a,
             behavior: answer, profile: ghost}
        """,
    )
    with pytest.raises(ValueError, match="unknown profile"):
        load_cases(missing)
    both = write(
        tmp_path,
        """
        profiles: {p: {members: true}}
        cases:
          - {id: a, lang: en, question: q, intent_esperado: wiki_general, corpus: n/a,
             behavior: answer, profile: p, player_state: {members: false}}
        """,
    )
    with pytest.raises(ValueError, match="both profile and player_state"):
        load_cases(both)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        cases:
          - {id: a, lang: en, question: q, intent_esperado: wiki_general, corpus: n/a,
             behavior: answer}
          - {id: a, lang: en, question: r, intent_esperado: wiki_general, corpus: n/a,
             behavior: answer}
        """,
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_only_wiki_questions_reach_crag() -> None:
    by_id = {c.id: c for c in load_cases()}
    assert by_id["quest-ds2-reqs-en"].reaches_crag
    assert by_id["wiki-zulrah-combat-en"].reaches_crag
    assert not by_id["flip-ranking-en"].reaches_crag
    assert not by_id["price-twisted-bow-en"].reaches_crag
    assert not by_id["ood-pizza-en"].reaches_crag


# --------------------------------------------------------------------------- gold check
@pytest.fixture
def corpus(db: str):
    zeros = "[" + ",".join("0" for _ in range(DIM)) + "]"
    with psycopg.connect(db) as conn:
        for title, content in [
            ("Abyssal whip", "The whip requires an Attack level of 70 to wield."),
            ("Abyssal whip", "It can be dropped by abyssal demons."),
            ("Dragon bones", "Dragon bones give 72 Prayer experience."),
        ]:
            conn.execute(
                "insert into documents (page_title, section, page_type, url, content, metadata, "
                "embedding) values (%s, '', 'item', 'https://x', %s, '{}', %s::vector)",
                (title, content, zeros),
            )
        conn.commit()
        yield conn


def case_with(**overrides) -> Case:
    return Case.model_validate(
        minimal(corpus="answerable", gold_pages=["Abyssal whip"], evidence=["attack level of 70"])
        | overrides
    )


def test_gold_check_accepts_labels_that_match_the_corpus_case_insensitively(corpus) -> None:
    assert check_gold(corpus, [case_with()]) == []


def test_gold_check_reports_missing_pages_and_missing_evidence(corpus) -> None:
    ghost = case_with(id="ghost", gold_pages=["Blorb"], evidence=["anything"])
    wrong = case_with(id="wrong", gold_pages=["Dragon bones"], evidence=["Attack level of 70"])
    problems = check_gold(corpus, [ghost, wrong])
    assert problems == [
        "ghost: gold page 'Blorb' is not in the corpus",
        "wrong: evidence 'Attack level of 70' not found in ['Dragon bones']",
    ]


def test_gold_check_ignores_cases_without_gold_labels(corpus) -> None:
    assert check_gold(corpus, [Case.model_validate(minimal())]) == []
