from __future__ import annotations

import pathlib

from app.ingest.quests import parse_quest

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wiki"

DS2_QUESTS = {
    "Legends' Quest",
    "Family Crest",
    "Heroes' Quest",
    "Shield of Arrav",
    "Lost City",
    "Merlin's Crystal",
    "Dragon Slayer I",
    "Underground Pass",
    "Biohazard",
    "Plague City",
    "Waterfall Quest",
    "Dream Mentor",
    "Lunar Diplomacy",
    "The Fremennik Trials",
    "Rune Mysteries",
    "Shilo Village",
    "Jungle Potion",
    "Eadgar's Ruse",
    "Druidic Ritual",
    "Troll Stronghold",
    "Death Plateau",
    "A Tail of Two Cats",
    "Icthlarin's Little Helper",
    "Gertrude's Cat",
    "Animal Magnetism",
    "Ernest the Chicken",
    "Priest in Peril",
    "The Restless Ghost",
    "Ghosts Ahoy",
    "Bone Voyage",
    "The Dig Site",
    "Client of Kourend",
    "X Marks the Spot",
    "Barbarian Training",
}


def quest(body: str, known: set[str] | None = None, members: str = "Yes"):
    wikitext = (
        f"{{{{Infobox Quest\n|name = T\n|members = {members}\n}}}}\n"
        "{{Quest details\n|difficulty = Novice\n|length = Short\n|start = Talk to X\n"
        f"{body}\n}}}}"
    )
    record = parse_quest("T", wikitext, quest_titles=known or set())
    assert record is not None
    return record


# --------------------------------------------------------------------------- real quests
def test_dragon_slayer_ii_requirements() -> None:
    record = parse_quest(
        "Dragon Slayer II",
        (FIXTURES / "quest_dragon_slayer_ii_head.wiki").read_text(),
        quest_titles=DS2_QUESTS,
    )
    assert record is not None
    assert (record.members, record.difficulty, record.length) == (True, "Grandmaster", "Very Long")
    assert record.skill_requirements == {
        "Magic": 75,
        "Smithing": 70,
        "Mining": 68,
        "Crafting": 62,
        "Agility": 60,
        "Thieving": 60,
        "Construction": 50,
        "Hitpoints": 50,
    }
    assert record.quest_points == 200 and record.combat_level is None
    assert set(record.boostable.values()) == {False}  # every one is "(not boostable)"


def test_quest_prerequisite_tree_gives_a_depth_ordered_list() -> None:
    record = parse_quest(
        "Dragon Slayer II",
        (FIXTURES / "quest_dragon_slayer_ii_head.wiki").read_text(),
        quest_titles=DS2_QUESTS,
    )
    assert record is not None
    depth = {r.quest: r.depth for r in record.quest_requirements}
    assert len(depth) == 34 - 1 + 1  # 33 quests + the Barbarian Training miniquest portion
    assert (
        depth["Legends' Quest"] == 2
        and depth["Heroes' Quest"] == 3
        and depth["Dragon Slayer I"] == 4
    )
    assert depth["Death Plateau"] == 5  # deepest: must be done before Troll Stronghold (4), etc.
    ordered = [r.quest for r in record.quest_requirements]
    assert (
        ordered.index("Death Plateau")
        < ordered.index("Troll Stronghold")
        < ordered.index("Eadgar's Ruse")
    )
    assert depth["Dragon Slayer I"] > depth["Heroes' Quest"] > depth["Legends' Quest"]


def test_non_quest_links_and_partial_requirements_are_not_quests() -> None:
    record = parse_quest(
        "Dragon Slayer II",
        (FIXTURES / "quest_dragon_slayer_ii_head.wiki").read_text(),
        quest_titles=DS2_QUESTS,
    )
    assert record is not None
    assert "Kudos" not in {r.quest for r in record.quest_requirements}
    assert "100 Kudos" in record.other_requirements
    partial = next(r for r in record.quest_requirements if r.quest == "Barbarian Training")
    assert partial.qualifier == "partial"


def test_items_have_quantities_and_optional_flags() -> None:
    record = parse_quest(
        "Dragon Slayer II",
        (FIXTURES / "quest_dragon_slayer_ii_head.wiki").read_text(),
        quest_titles=DS2_QUESTS,
    )
    assert record is not None
    items = {i.item: i for i in record.items_required if i.item}
    assert items["oak plank"].qty == 8 and items["swamp paste"].qty == 10
    assert items["nail"].qty == 12 and not items["nail"].optional  # "bringing more is recommended"
    assert items["pickaxe"].qty is None
    assert record.recommended[0].startswith("combat level 100")


def test_cooks_assistant_has_no_requirements_but_has_items() -> None:
    record = parse_quest("Cook's Assistant", (FIXTURES / "quest_cooks_assistant.wiki").read_text())
    assert record is not None
    assert record.members is False and record.difficulty == "Novice"
    assert not record.skill_requirements and not record.quest_requirements
    names = [i.item for i in record.items_required]
    assert names == ["Egg", "Bucket of milk", "bucket", "Pot of flour", "pot"]
    optional = {i.item: i.optional for i in record.items_required}
    assert optional["Egg"] and optional["bucket"] and not optional["Bucket of milk"]


# --------------------------------------------------------------------------- synthetic
def test_qualifiers() -> None:
    record = quest(
        "|requirements =\n* [[Quest A]]\n* Started [[Quest B]]\n* Completion of [[Quest C]]\n"
        "* Partial completion of [[Quest D]]\n",
        known={"Quest A", "Quest B", "Quest C", "Quest D"},
    )
    assert {r.quest: r.qualifier for r in record.quest_requirements} == {
        "Quest A": "completed",
        "Quest B": "started",
        "Quest C": "completed",
        "Quest D": "partial",
    }


def test_header_line_and_none_are_ignored() -> None:
    record = quest(
        "|requirements =\n*Completion of the following quests:\n**[[Quest A]]\n", known={"Quest A"}
    )
    assert [r.quest for r in record.quest_requirements] == ["Quest A"]
    assert record.other_requirements == []
    assert quest("|requirements = None").quest_requirements == []
    assert quest("").skill_requirements == {}


def test_shared_prerequisites_keep_the_deepest_level_and_completed_wins() -> None:
    record = quest(
        "|requirements =\n*[[Quest A]]\n**[[Quest B]]\n*Started [[Quest B]]\n"
        "***[[Quest C]]\n**[[Quest C]]\n",
        known={"Quest A", "Quest B", "Quest C"},
    )
    by = {r.quest: r for r in record.quest_requirements}
    assert by["Quest B"].depth == 2 and by["Quest B"].qualifier == "completed"
    assert by["Quest C"].depth == 3
    assert [r.quest for r in record.quest_requirements] == ["Quest C", "Quest B", "Quest A"]


def test_skill_boost_flags_and_soft_text_requirements() -> None:
    record = quest(
        "|requirements =\n*{{SCP|Slayer|75|link=yes}} {{Boostable|yes}}\n"
        "*{{SCP|Magic|50}} {{Boostable|no}}\n*{{SCP|Combat|100}}\n*{{SCP|Quest|50}}\n"
        "*Access to the [[Fremennik Isles]]\n"
    )
    assert record.skill_requirements == {"Slayer": 75, "Magic": 50}
    assert record.boostable == {"Slayer": True, "Magic": False}
    assert (record.combat_level, record.quest_points) == (100, 50)
    assert record.other_requirements == ["Access to the Fremennik Isles"]


def test_members_flag_comes_from_the_infobox() -> None:
    assert quest("", members="Yes").members is True
    assert quest("", members="No").members is False
    assert parse_quest("T", "{{Quest details\n|difficulty = Easy\n}}").members is None


def test_pages_without_quest_details_return_none() -> None:
    assert parse_quest("Quests", "A list of quests.") is None
