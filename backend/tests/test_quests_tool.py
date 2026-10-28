"""Quest requirements vs the player's snapshot (app/tools/quests.py): deterministic, by hand."""

from __future__ import annotations

import pytest

from app.api.schemas import SKILL_NAMES, PlayerState
from app.clients.prices import ItemMeta
from app.ingest.quests import QuestItem, QuestRecord, QuestRequirement
from app.tools.items import ItemIndex
from app.tools.quests import QuestStore, check_quest

ITEMS = ItemIndex(
    [
        ItemMeta(id=2102, name="Lemon"),
        ItemMeta(id=1987, name="Grapes"),
        ItemMeta(id=1511, name="Logs"),
        ItemMeta(id=1265, name="Bronze pickaxe"),
    ]
)

MM2 = QuestRecord(
    title="Monkey Madness II",
    members=True,
    start="Talk to King Narnode Shareen.",
    difficulty="Master",
    length="Very Long",
    skill_requirements={"Slayer": 69, "Crafting": 70, "Hunter": 60},
    boostable={"Slayer": False, "Crafting": False, "Hunter": True},
    quest_requirements=[  # the wiki's tree: deepest prerequisites first
        QuestRequirement(quest="Tree Gnome Village", depth=3),
        QuestRequirement(quest="The Grand Tree", depth=3),
        QuestRequirement(quest="Monkey Madness I", depth=2),
        QuestRequirement(quest="Recipe for Disaster", depth=1),
        QuestRequirement(quest="Enlightened Journey", depth=1),
        QuestRequirement(quest="Watchtower", qualifier="partial", depth=1),
    ],
    other_requirements=["Balloon flight route to Gnome Stronghold unlocked"],
    items_required=[
        QuestItem(text="A lemon", item="Lemon"),
        QuestItem(text="Grapes", item="Grapes"),
        QuestItem(text="Logs", item="Logs", qty=2),
        QuestItem(
            text="A pickaxe (if you choose the tanking route)", item="Pickaxe", optional=True
        ),
        QuestItem(text="A light source"),
    ],
    recommended=["Food", "Stamina potions"],
)


def player(**overrides) -> PlayerState:
    base = {
        "account_type": "main",
        "members": True,
        "skills": {"slayer": 66, "crafting": 70},  # hunter not reported
        "quests": {
            "Tree Gnome Village": "FINISHED",
            "The Grand Tree": "FINISHED",
            "Monkey Madness I": "IN_PROGRESS",
            "Recipe for Disaster": "NOT_STARTED",
            "Watchtower": "IN_PROGRESS",
        },  # Enlightened Journey not reported
        "inventory": [{"id": 2102, "qty": 1}],
    }
    return PlayerState.model_validate(base | overrides)


def test_requirements_are_split_into_met_missing_and_unknown() -> None:
    out = check_quest(MM2, player(), ITEMS)
    assert out["skills"]["met"] == [{"skill": "Crafting", "have": 70, "need": 70}]
    assert out["skills"]["missing"] == [
        {"skill": "Slayer", "have": 66, "need": 69, "levels_short": 3, "boostable": False}
    ]
    assert out["skills"]["unknown"] == [{"skill": "Hunter"}]  # never reported: not "missing"
    assert out["quests"]["done"] == ["Tree Gnome Village", "The Grand Tree"]
    assert [(m["quest"], m["status"]) for m in out["quests"]["missing"]] == [
        ("Monkey Madness I", "IN_PROGRESS"),
        ("Recipe for Disaster", "NOT_STARTED"),
    ]
    assert out["quests"]["unknown"] == ["Enlightened Journey"]
    assert out["quests"]["verify_in_game"] == [
        "Watchtower"
    ]  # "partial" cannot be read from a state
    assert out["ready_to_start"] is False


def test_the_suggested_order_is_the_missing_prerequisites_deepest_first_then_the_quest() -> None:
    out = check_quest(MM2, player(), ITEMS)
    assert out["suggested_order"] == [
        "Monkey Madness I",
        "Recipe for Disaster",
        "Monkey Madness II",
    ]


def test_a_quest_with_nothing_missing_is_ready_and_unknowns_make_it_uncertain() -> None:
    everything = player(
        skills=dict.fromkeys(SKILL_NAMES, 80),
        quests={q.quest: "FINISHED" for q in MM2.quest_requirements},
    )
    out = check_quest(MM2, everything, ITEMS)
    assert out["ready_to_start"] is True and out["suggested_order"] == ["Monkey Madness II"]
    unsure = player(
        skills=dict.fromkeys(SKILL_NAMES, 80),
        quests={q.quest: "FINISHED" for q in MM2.quest_requirements if q.quest != "Watchtower"},
    )
    assert check_quest(MM2, unsure, ITEMS)["ready_to_start"] is None  # unreported quest


def test_started_and_partial_qualifiers() -> None:
    record = QuestRecord(
        title="Test",
        quest_requirements=[
            QuestRequirement(quest="A", qualifier="started"),
            QuestRequirement(quest="B", qualifier="partial"),
            QuestRequirement(quest="C", qualifier="completed"),
        ],
    )
    one = check_quest(record, player(quests={"A": "IN_PROGRESS", "B": "FINISHED", "C": "FINISHED"}))
    assert one["quests"]["missing"] == [] and one["quests"]["verify_in_game"] == []
    two = check_quest(
        record, player(quests={"A": "NOT_STARTED", "B": "IN_PROGRESS", "C": "IN_PROGRESS"})
    )
    assert [m["quest"] for m in two["quests"]["missing"]] == ["A", "C"]
    assert two["quests"]["verify_in_game"] == ["B"]


def test_quest_points_and_combat_level_come_from_the_snapshot() -> None:
    record = QuestRecord(title="Big", quest_points=200, combat_level=100)
    out = check_quest(record, player(skills=dict.fromkeys(SKILL_NAMES, 70), quest_points=150))
    assert out["numeric_requirements"] == {
        "quest_points": {"need": 200, "have": 150},
        "combat_level": {"need": 100, "have": 89},
    }
    assert out["ready_to_start"] is False
    unknown = check_quest(record, player(skills={}))
    assert unknown["ready_to_start"] is None  # neither quest points nor combat level reported


def test_members_only_quests_are_blocked_for_free_players() -> None:
    out = check_quest(QuestRecord(title="X", members=True), player(members=False))
    assert out["members_requirement_unmet"] is True and out["ready_to_start"] is False
    assert (
        check_quest(QuestRecord(title="X", members=True), player(members=None))[
            "members_requirement_unmet"
        ]
        is False
    )


# --------------------------------------------------------------------------- items
def item_status(out: dict) -> dict[str, str]:
    return {i["text"]: i["status"] for i in out["items"]["required"]}


def test_items_are_looked_for_in_the_inventory_and_only_a_known_bank() -> None:
    out = check_quest(MM2, player(), ITEMS)
    assert item_status(out) == {
        "A lemon": "in_inventory",
        "Grapes": "not_in_inventory_bank_unknown",  # the bank was never opened
        "Logs": "not_in_inventory_bank_unknown",
        "A light source": "unverified",
    }
    assert out["items"]["bank_known"] is False and "bank" in out["items"]["note"]


def test_a_captured_bank_settles_where_the_items_are() -> None:
    bank = {
        "captured_at": "2026-10-14T11:00:00Z",
        "items": [{"id": 1511, "qty": 5}, {"id": 1987, "qty": 1}],
    }
    out = check_quest(MM2, player(bank=bank), ITEMS)
    assert item_status(out)["Logs"] == "in_bank" and item_status(out)["Grapes"] == "in_bank"
    short = {"captured_at": "2026-10-14T11:00:00Z", "items": [{"id": 1511, "qty": 1}]}
    out = check_quest(MM2, player(bank=short), ITEMS)
    assert item_status(out)["Logs"] == "not_found"  # needs 2, bank has 1, inventory none
    assert out["items"]["note"] is None and out["items"]["bank_known"] is True


def test_optional_items_are_not_required_and_coins_use_the_gp_total() -> None:
    record = QuestRecord(
        title="Pay",
        items_required=[
            QuestItem(text="500 coins", item="Coins", qty=500),
            QuestItem(text="Pickaxe if mining", item="Bronze pickaxe", optional=True),
        ],
    )
    out = check_quest(record, player(gp=1_000, inventory=[]), ITEMS)
    assert [i["status"] for i in out["items"]["required"]] == ["in_inventory"]
    poor = check_quest(record, player(gp=100, inventory=[]), ITEMS)
    assert poor["items"]["required"][0]["status"] == "not_in_inventory_bank_unknown"


# --------------------------------------------------------------------------- store
def test_the_store_finds_quests_named_in_text_with_numerals_and_spelling_variants() -> None:
    cooks = QuestRecord(title="Cook's Assistant")
    mm1 = QuestRecord(title="Monkey Madness I")
    store = QuestStore([MM2, mm1, cooks])
    assert [r.title for r in store.resolve("how do I start monkey madness 2?")] == [
        "Monkey Madness II"
    ]
    assert [r.title for r in store.resolve("¿cómo hago cooks assistant?")] == ["Cook's Assistant"]
    assert [r.title for r in store.resolve("Monkey Madness I walkthrough")] == ["Monkey Madness I"]
    assert store.resolve("what is the best quest for xp?") == [] and len(store) == 3


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_text_names_no_quest(text: str) -> None:
    assert QuestStore([MM2]).resolve(text) == []
