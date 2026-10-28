"""The small pure pieces of the graph: routing rules, language detection, parsing, summaries."""

from __future__ import annotations

import pytest

from app.api.schemas import SKILL_NAMES, PlayerState
from app.clients.jev import ChoiceAnswer
from app.config import Settings
from app.graph.footer import latest_price_time, with_price_footer
from app.graph.intents import Intent
from app.graph.nodes.condense import detect_language, is_underspecified
from app.graph.nodes.router import decide_intent
from app.graph.nodes.workflows import stated_category, stated_combat_level
from app.graph.parsing import parse_json_object
from app.graph.smalltalk import classify_smalltalk
from app.graph.state import (
    MAX_STORED_MESSAGES,
    append_messages,
    passage_from_dict,
    passage_to_dict,
)
from app.graph.summary import (
    compact_gp,
    conversation_summary,
    player_headline,
    player_summary,
)
from app.rag.retrieve import Passage
from app.rag.titles import TitleIndex

SETTINGS = Settings(_env_file=None)


def answer(choice: str, probabilities: dict[str, float], confidence: float | None) -> ChoiceAnswer:
    return ChoiceAnswer(
        type="choice", choice=choice, probabilities=probabilities, confidence=confidence
    )


# --------------------------------------------------------------------------- routing rules
def test_a_confident_in_domain_choice_is_kept() -> None:
    assert decide_intent(answer("quest", {"quest": 0.9}, 0.85), SETTINGS) is Intent.QUEST
    assert decide_intent(answer("flipping", {"flipping": 0.7}, 0.6), SETTINGS) is Intent.FLIPPING


def test_low_confidence_falls_back_to_the_general_path() -> None:
    low = answer("moneymaking", {"moneymaking": 0.5, "wiki_general": 0.4}, 0.4)
    assert decide_intent(low, SETTINGS) is Intent.WIKI_GENERAL
    assert SETTINGS.route_min_confidence == 0.45


def test_missing_confidence_uses_the_probability_of_the_choice() -> None:
    assert decide_intent(answer("quest", {"quest": 0.9}, None), SETTINGS) is Intent.QUEST
    assert decide_intent(answer("quest", {"quest": 0.3}, None), SETTINGS) is Intent.WIKI_GENERAL


def test_rejecting_a_question_needs_high_probability_not_just_the_top_choice() -> None:
    # Seen live: a legitimate Spanish Dragon Slayer II question got out_of_domain = 0.35.
    doubtful = answer("out_of_domain", {"out_of_domain": 0.6, "quest": 0.35}, 0.6)
    assert decide_intent(doubtful, SETTINGS) is Intent.WIKI_GENERAL
    sure = answer("out_of_domain", {"out_of_domain": 0.93, "wiki_general": 0.05}, 0.9)
    assert decide_intent(sure, SETTINGS) is Intent.OUT_OF_DOMAIN


def test_an_unknown_label_is_the_general_path() -> None:
    assert decide_intent(answer("smalltalk", {"smalltalk": 1.0}, 1.0), SETTINGS) is (
        Intent.WIKI_GENERAL
    )


# --------------------------------------------------------------------------- language
@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("What Attack level do I need to wield an abyssal whip?", "en"),
        ("how do I train prayer", "en"),
        ("¿Cuánta experiencia de Prayer da enterrar un hueso de dragón?", "es"),
        ("Qué necesito para empezar Dragon Slayer II", "es"),
        ("cuanto cuesta la abyssal whip", "es"),  # no accents, but two Spanish function words
        ("abyssal whip price", None),  # nothing to tell: ask the model
        ("tbow", None),
        ("", None),
    ],
)
def test_clear_languages_are_detected_without_the_model(text: str, language: str | None) -> None:
    assert detect_language(text) == language


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"query": "dragon bones", "language": "es"}',
            {"query": "dragon bones", "language": "es"},
        ),
        ('```json\n{"query": "x"}\n```', {"query": "x"}),
        (
            'Sure! Here it is: {"query": "x", "n": {"a": 1}} hope it helps',
            {"query": "x", "n": {"a": 1}},
        ),
        ('{"query": "unterminated', None),
        ("no json here", None),
        ("[1, 2]", None),
        ("", None),
        (None, None),
    ],
)
def test_json_objects_are_found_in_model_output(text: str | None, expected: dict | None) -> None:
    assert parse_json_object(text) == expected


# --------------------------------------------------------------------------- state helpers
def test_the_threads_memory_is_capped_to_the_most_recent_messages() -> None:
    history: list = []
    for i in range(MAX_STORED_MESSAGES + 7):
        history = append_messages(history, {"role": "user", "content": str(i)})
    assert len(history) == MAX_STORED_MESSAGES
    assert history[0]["content"] == "7" and history[-1]["content"] == str(MAX_STORED_MESSAGES + 6)
    assert append_messages(None, [{"role": "user", "content": "a"}]) == [
        {"role": "user", "content": "a"}
    ]


def test_passages_survive_a_round_trip_through_the_state() -> None:
    passage = Passage(
        id=-5, page_title="T", section="S", page_type="quest", url="u", content="c",
        metadata={"live": True}, score=0.5,
    )  # fmt: skip
    again = passage_from_dict(passage_to_dict(passage))
    assert (again.id, again.page_title, again.section, again.content, again.url) == (
        -5, "T", "S", "c", "u",
    )  # fmt: skip


# --------------------------------------------------------------------------- summaries
def test_compact_gp() -> None:
    assert [compact_gp(n) for n in (950, 1_000, 6_500_000, 2_147_483_647, 9_360_000_000)] == [
        "950",
        "1K",
        "6.5M",
        "2.1B",
        "9.4B",
    ]


def test_the_conversation_summary_shows_the_earlier_turns_not_the_new_message() -> None:
    messages = [
        {"role": "user", "content": "How do I start Monkey Madness II?"},
        {"role": "assistant", "content": "Talk to King Narnode Shareen. " * 30},
        {"role": "user", "content": "¿y si no tengo ese ítem?"},
    ]
    summary = conversation_summary(messages)
    assert summary.startswith("user: How do I start Monkey Madness II?")
    assert "assistant: Talk to King Narnode Shareen." in summary and "ítem" not in summary
    assert len(summary.splitlines()[1]) < 300  # clipped
    assert conversation_summary(messages[:1]) == "(none: first message)"


def test_the_player_summary_says_what_is_known_and_what_is_not() -> None:
    assert "UNKNOWN" in player_summary(None) and "no snapshot" in player_headline(None)
    empty = PlayerState(account_type="ironman")  # type: ignore[arg-type]
    text = player_summary(empty)
    assert "ironman" in text and "membership unknown" in text and "GP not reported" in text
    assert "Skills: not reported" in text and "Quests: not reported" in text
    assert "bank not captured" in text
    full = PlayerState.model_validate(
        {
            "account_type": "main",
            "members": True,
            "skills": dict.fromkeys(SKILL_NAMES, 70),
            "quests": {"A": "FINISHED", "B": "NOT_STARTED"},
            "quest_points": 150,
            "gp": 6_500_000,
            "bank": {"captured_at": "2026-10-14T11:00:00Z", "items": []},
        }
    )
    text = player_summary(full)
    assert "Combat level 89" in text and "total level 1,680" in text and "150 quest points" in text
    assert "6,500,000 GP" in text and "Quests: 1 finished out of 2" in text
    assert "attack 70" in text and "bank captured 2026-10-14 11:00 UTC" in text
    assert player_headline(full) == "main, members, combat 89, 6.5M gp"
    # quests named in the question or the evidence are listed with their state
    mentioned = player_summary(full, mentioned_in="can I wield it? needs a + B and not C")
    assert "State of the quests named in this conversation: A FINISHED; B NOT_STARTED." in mentioned
    assert "named in this conversation" not in player_summary(full)


# --------------------------------------------------------------------------- footer
def test_the_price_footer_names_the_newest_timestamp() -> None:
    results = [
        {"tool": "a", "result": {"prices_fetched_at": "2026-10-14T11:59:30.1Z"}},
        {"tool": "b", "result": {"prices_fetched_at": "2026-10-14T12:03:05Z"}},
        {"tool": "c", "result": {"error": "x"}},
    ]
    assert latest_price_time(results) == "2026-10-14T12:03:05Z"
    es = with_price_footer("Hola.", "es", results)
    assert es.startswith("Hola.\n\n") and "2026-10-14 12:03 UTC" in es and "Precios" in es
    assert "Prices from" in with_price_footer("Hi.", "en", results)
    assert with_price_footer("Hi.", "en", [{"tool": "c", "result": {}}]) == "Hi."


# ------------------------------------------------------------------- underspecified questions
VAGUE_TITLES = TitleIndex(["Abyssal whip", "Zulrah", "Dragon bones"])


VAGUE = [
    "Where do I get it?",
    "Where to get it",
    "How much does it cost?",
    "What is its weakness?",
    "What if I don't have that?",
    "how many points does this give",
]
SPECIFIC = [
    "Where do I get the abyssal whip?",
    "What is Zulrah's weakness?",
    "How do I train Agility?",
    "How do I get rich?",
    "Tell me about that weapon from abyssal demons",
    "What is the best way to level Prayer without a gilded altar or a house?",
]


@pytest.mark.parametrize("query", VAGUE)
def test_a_pronoun_only_first_message_is_underspecified(query: str) -> None:
    assert is_underspecified(query, VAGUE_TITLES, first_message=True)


@pytest.mark.parametrize("query", SPECIFIC)
def test_a_question_that_names_something_or_has_no_pronoun_is_not(query: str) -> None:
    assert not is_underspecified(query, VAGUE_TITLES, first_message=True)


def test_a_follow_up_is_never_underspecified_the_conversation_says_what_it_is() -> None:
    assert not is_underspecified("Where do I get it?", VAGUE_TITLES, first_message=False)


# =========================================================================== small talk
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello!", ("greeting", "en")),
        ("hey there", ("greeting", "en")),
        ("Hola", ("greeting", "es")),
        ("Buenas tardes", ("greeting", "es")),
        ("buen día", ("greeting", "es")),
        ("thanks", ("thanks", "en")),
        ("thank you so much", ("thanks", "en")),
        ("gracias!", ("thanks", "es")),
        ("bye", ("bye", "en")),
        ("chau", ("bye", "es")),
    ],
)
def test_a_message_made_only_of_greetings_is_small_talk(text: str, expected) -> None:
    assert classify_smalltalk(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hi, what is a whip?",
        "hola, ¿cuánto cuesta una whip?",
        "thanks, and what about the whip?",
        "what is ok?",
        "",
        "!!!",
        "hello there general kenobi how are you",
    ],
)
def test_anything_with_other_words_in_it_is_not_small_talk(text: str) -> None:
    assert classify_smalltalk(text) is None


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("Cual es la mejor forma de ganar dinero para un personaje de lvl 80 combat?", 80),
        ("best money for combat level 90", 90),
        ("I have 70 combat", 70),
        ("combat 85", 85),
        ("what is the combat triangle", None),
        ("combat level 500", None),
        ("level 3 skiller", None),
    ],
)
def test_a_combat_level_the_player_wrote_is_read_from_the_message(text: str, level) -> None:
    assert stated_combat_level(text) == level


def test_the_guide_category_comes_from_the_players_words_only_when_it_is_unambiguous() -> None:
    assert stated_category("only combat methods") == "Combat"
    assert stated_category("solo con cosas de combate") == "Combat"
    assert stated_category("combat or skilling") is None
    assert stated_category("any way to make gp") is None


def test_unreported_membership_is_probably_members_when_members_only_skills_are_trained() -> None:
    from app.graph.summary import membership_text

    trained = PlayerState.model_validate({"skills": {"herblore": 41, "attack": 75}})
    fresh = PlayerState.model_validate({"skills": {"herblore": 1, "attack": 75}})
    assert "probably a member" in membership_text(trained)
    assert membership_text(fresh) == "membership unknown"
    assert membership_text(PlayerState.model_validate({"members": False})) == "free-to-play"
