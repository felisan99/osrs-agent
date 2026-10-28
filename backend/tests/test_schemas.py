"""The plugin <-> backend contract (SPEC §4, §5): validation, tolerance and derived facts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    SKILL_NAMES,
    AccountType,
    ChatRequest,
    ChatResponse,
    PlayerState,
    QuestStatus,
)


def skills(**overrides: int) -> dict[str, int]:
    base = dict.fromkeys(SKILL_NAMES, 1) | {"hitpoints": 10}
    return base | overrides


# --------------------------------------------------------------------------- combat level
@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ({}, 3),  # a fresh account: (1 + 10 + 0)/4 + 0.325 * 2 = 3.4
        ({s: 99 for s in SKILL_NAMES}, 126),
        (
            {
                "attack": 60,
                "strength": 60,
                "defence": 60,
                "hitpoints": 60,
                "prayer": 43,
                "ranged": 99,
                "magic": 60,
            },
            83,
        ),
        # pure melee: (1 + 99 + 0)/4 + 0.325 * (99 + 99) = 25 + 64.35
        ({"attack": 99, "strength": 99, "defence": 1, "hitpoints": 99, "prayer": 1}, 89),
        # pure mage: (1 + 10 + 0)/4 + 0.325 * floor(1.5 * 99) = 2.75 + 48.1
        ({"magic": 99}, 50),
    ],
)
def test_combat_level_follows_the_wiki_formula(levels: dict[str, int], expected: int) -> None:
    state = PlayerState(skills=skills(**levels))
    assert state.combat_level == expected


def test_combat_level_is_unknown_without_the_combat_skills() -> None:
    assert PlayerState(skills={"attack": 70}).combat_level is None
    assert PlayerState().combat_level is None


# --------------------------------------------------------------------------- tolerance
def test_unknown_account_types_degrade_to_unknown_instead_of_failing() -> None:
    assert PlayerState(account_type="regular_chaos_mode").account_type is AccountType.UNKNOWN  # type: ignore[arg-type]
    assert PlayerState(account_type="  Ironman ").account_type is AccountType.IRONMAN  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "ironman"),
    [
        ("main", False),
        ("unknown", False),
        ("ironman", True),
        ("hardcore_ironman", True),
        ("ultimate_ironman", True),
        ("group_ironman", True),
        ("hardcore_group_ironman", True),
    ],
)
def test_every_ironman_variant_counts_as_an_ironman(kind: str, ironman: bool) -> None:
    assert PlayerState(account_type=kind).is_ironman is ironman  # type: ignore[arg-type]


def test_only_real_skills_are_kept_and_names_are_normalised() -> None:
    state = PlayerState(skills={"Attack": 70, "OVERALL": 2000, "Necromancy": 5})  # type: ignore[arg-type]
    assert state.skills == {"attack": 70}
    assert state.level("ATTACK") == 70 and state.level("magic") is None


def test_levels_are_bounded_by_the_virtual_level_cap() -> None:
    with pytest.raises(ValidationError):
        PlayerState(skills={"attack": 127})
    with pytest.raises(ValidationError):
        PlayerState(skills={"attack": 0})
    assert PlayerState(skills={"attack": 126}).skills["attack"] == 126


def test_unknown_fields_are_ignored_not_stored() -> None:
    state = PlayerState.model_validate({"members": True, "password": "hunter2", "extra": [1]})
    assert state.members is True and not hasattr(state, "password")
    assert "password" not in state.model_dump()


def test_quest_lookup_ignores_case_and_reports_missing_quests_as_none() -> None:
    state = PlayerState(quests={"Monkey Madness I": "FINISHED", "Cook's Assistant": "IN_PROGRESS"})  # type: ignore[arg-type]
    assert state.quest_status("monkey madness i") is QuestStatus.FINISHED
    assert state.quest_status("COOK'S ASSISTANT") is QuestStatus.IN_PROGRESS
    assert (
        state.quest_status("Dragon Slayer II") is None
    )  # never reported: unknown, not NOT_STARTED


def test_invalid_quest_states_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PlayerState(quests={"X": "DONE"})  # type: ignore[arg-type]


def test_total_level_needs_every_skill() -> None:
    assert PlayerState(skills={s: 50 for s in SKILL_NAMES}).total_level == 50 * len(SKILL_NAMES)
    assert PlayerState(skills={"attack": 50}).total_level is None


def test_big_stacks_do_not_overflow() -> None:
    state = PlayerState.model_validate(
        {"gp": 9_360_000_000, "bank": {"items": [{"id": 995, "qty": 12_000_000_000}]}}
    )
    assert state.gp == 9_360_000_000 and state.bank and state.bank.items[0].qty == 12_000_000_000


@pytest.mark.parametrize("field", ["inventory", "quests", "skills"])
def test_oversized_collections_are_rejected(field: str) -> None:
    bloated = {
        "inventory": [{"id": 1, "qty": 1}] * 101,
        "quests": {f"quest {i}": "FINISHED" for i in range(501)},
        "skills": {f"skill{i}": 1 for i in range(41)},
    }
    if field == "skills":  # unknown skills are dropped before the size check: nothing to reject
        assert PlayerState(skills=bloated["skills"]).skills == {}  # type: ignore[arg-type]
        return
    with pytest.raises(ValidationError):
        PlayerState.model_validate({field: bloated[field]})


# --------------------------------------------------------------------------- request / response
def test_a_chat_request_needs_a_well_formed_thread_id_and_a_message() -> None:
    ok = ChatRequest(thread_id="3f2b8c1e-aaaa-bbbb-cccc-0123456789ab", message="hola")
    assert ok.player_state is None
    for bad in ("short", "has spaces in it!", "x" * 65, "ünïcode-thread-id"):
        with pytest.raises(ValidationError):
            ChatRequest(thread_id=bad, message="hola")
    with pytest.raises(ValidationError):
        ChatRequest(thread_id="3f2b8c1e-aaaa", message="")


def test_a_chat_request_never_carries_the_users_key() -> None:
    # The key travels in a header (SPEC §4); a body field with that name would be ignored.
    request = ChatRequest.model_validate(
        {"thread_id": "abcdefgh12", "message": "hi", "openrouter_key": "sk-or-v1-secret"}
    )
    assert "sk-or" not in request.model_dump_json()


def test_the_response_shape_matches_the_spec() -> None:
    response = ChatResponse(
        answer="x",
        intent="moneymaking",
        sources=[{"title": "T", "url": "https://oldschool.runescape.wiki/w/T"}],  # type: ignore[list-item]
    )
    dumped = response.model_dump(exclude_none=True)
    assert set(dumped) == {"answer", "intent", "sources"}  # debug only when enabled
    with pytest.raises(ValidationError):
        ChatResponse(answer="x", intent="chitchat")  # type: ignore[arg-type]
