from __future__ import annotations

import pytest

from app.clients.prices import ItemMeta
from app.tools.items import FACE_VALUE, ITEM_ALIASES, ItemIndex, resolution_error

# A miniature /mapping with the traps seen on the real one (4.6k items, tuned against it).
NAMES = [
    "Abyssal whip",
    "Frozen whip mix",
    "Volcanic whip mix",
    "Whisky",
    "Dragon bones",
    "Superior dragon bones",
    "Frost dragon bones",
    "Lava dragon bones",
    "Saradomin brew(1)",
    "Saradomin brew(2)",
    "Saradomin brew(3)",
    "Saradomin brew(4)",
    "Super combat potion(1)",
    "Super combat potion(2)",
    "Super combat potion(3)",
    "Super combat potion(4)",
    "Divine super combat potion(1)",
    "Divine super combat potion(4)",
    "Prayer potion(3)",
    "Prayer potion(4)",
    "Super restore(4)",
    "Super restore mix(2)",
    "Dragon dagger",
    "Dragon dagger(p)",
    "Dragon dagger(p+)",
    "Dragon dagger(p++)",
    "Rune scimitar",
    "Rune sword",
    "Rune platebody",
    "Rune platelegs",
    "Air rune",
    "Nature rune",
    "Twisted bow",
    "Magic shortbow",
    "Maple shortbow",
    "Amulet of fury",
    "Fury ornament kit",
    "Old school bond",
    "Zamorak's grapes",
    "Zamorak godsword",
    "Ahrim's robetop",
    "Bandos godsword",
    "Bandos chestplate",
    "Toadflax",
    "Grimy toadflax",
]
ITEMS = [ItemMeta(id=1000 + i, name=n, members=i % 2 == 0) for i, n in enumerate(NAMES)]
INDEX = ItemIndex(ITEMS)


def name(query: str) -> str | None:
    result = INDEX.resolve(query)
    return result.item.name if result.item else None


def test_exact_match_ignores_case_whitespace_and_curly_apostrophes() -> None:
    for query in ("abyssal whip", "ABYSSAL WHIP", "  Abyssal   whip  "):
        result = INDEX.resolve(query)
        assert result.status == "exact" and result.item and result.item.name == "Abyssal whip"
    assert name("zamorak’s grapes") == "Zamorak's grapes"  # typographic apostrophe


def test_nicknames_resolve_to_the_full_item_name() -> None:
    for nickname, target in {
        "tbow": "Twisted bow",
        "bgs": "Bandos godsword",
        "fury": "Amulet of fury",
        "bond": "Old school bond",
        "TBOW": "Twisted bow",
    }.items():
        result = INDEX.resolve(nickname)
        assert result.status == "exact" and name(nickname) == target
        assert result.note and "nickname" in result.note


def test_nickname_whose_target_is_missing_from_mapping_is_skipped_not_fatal() -> None:
    # Only items from NAMES exist, so e.g. "ags" (Armadyl godsword) has no target here.
    assert "ags" in ITEM_ALIASES
    assert INDEX.resolve("ags").item is None


def test_family_named_without_dose_defaults_to_the_full_dose_variant() -> None:
    for query, expected in {
        "saradomin brew": "Saradomin brew(4)",
        "prayer potion": "Prayer potion(4)",
        "super restore": "Super restore(4)",  # not the "mix" family
        "super combat potion": "Super combat potion(4)",  # not the Divine family
    }.items():
        result = INDEX.resolve(query)
        assert result.item and result.item.name == expected, query
        assert result.note and "full-dose" in result.note


def test_explicit_dose_is_respected() -> None:
    assert name("saradomin brew(2)") == "Saradomin brew(2)"
    assert name("Super combat potion(1)") == "Super combat potion(1)"


def test_prefix_preference_beats_substring_matches() -> None:
    # "super combat" is a prefix of Super combat potion(n) but only a substring of the Divine one.
    assert name("super combat") == "Super combat potion(4)"


def test_typos_are_accepted_when_there_is_a_clear_winner() -> None:
    assert name("abbysal whip") == "Abyssal whip"
    assert name("twisted bwo") == "Twisted bow"
    assert INDEX.resolve("abbysal whip").status == "fuzzy"


def test_plus_signs_are_significant() -> None:
    assert name("dragon dagger p++") == "Dragon dagger(p++)"
    assert name("dragon dagger p+") == "Dragon dagger(p+)"
    assert name("dragon dagger(p)") == "Dragon dagger(p)"


def test_generic_queries_are_ambiguous_and_list_candidates() -> None:
    for query in ("rune", "whip"):
        result = INDEX.resolve(query)
        assert result.status == "ambiguous" and result.item is None
        assert 2 <= len(result.candidates) <= 5
        assert all(c.score >= 80 for c in result.candidates)


def test_junk_is_not_found_and_does_not_pretend_to_be_ambiguous() -> None:
    for query in ("xyzzy", "asdf", "", "   "):
        result = INDEX.resolve(query)
        assert result.status == "not_found" and result.item is None
    assert INDEX.resolve("asdf").candidates == [] or INDEX.resolve("asdf").candidates[0].score < 80


def test_special_items_missing_from_mapping_resolve_with_face_value() -> None:
    coins = INDEX.resolve("coins")
    assert coins.item and (coins.item.id, coins.item.value) == (995, 1)
    plat = INDEX.resolve("Platinum token")
    assert plat.item and (plat.item.id, plat.item.value) == (13204, 1000)
    assert FACE_VALUE == {995: 1, 13204: 1000}


def test_lookup_by_id_and_names_for_unknown_ids() -> None:
    first = ITEMS[0]
    assert INDEX.by_id(first.id) == first and INDEX.name_of(first.id) == "Abyssal whip"
    assert INDEX.name_of(424242) == "item 424242"
    assert len(INDEX) == len(ITEMS) + 2  # + Coins + Platinum token


def test_items_sharing_an_exact_name_are_ambiguous() -> None:
    twin = ItemIndex([ItemMeta(id=1, name="Mystery box"), ItemMeta(id=2, name="Mystery box")])
    result = twin.resolve("mystery box")
    assert result.status == "ambiguous" and {c.id for c in result.candidates} == {1, 2}


def test_resolution_error_payload_is_llm_readable() -> None:
    ambiguous = resolution_error(INDEX.resolve("rune"))
    assert ambiguous["error"] == "item_ambiguous" and ambiguous["candidates"]
    assert "exact candidate names" in ambiguous["hint"]
    missing = resolution_error(INDEX.resolve("xyzzy"))
    assert missing["error"] == "item_not_found" and "spelling" in missing["hint"]


@pytest.mark.live
async def test_live_mapping_resolves_the_common_queries_and_every_alias() -> None:
    import httpx

    from app.clients.prices import PricesClient
    from app.config import Settings

    settings = Settings(_env_file=None, prices_user_agent="osrs-agent-tests - live check")
    async with httpx.AsyncClient() as http:
        mapping = await PricesClient(settings, http=http).mapping()
    live = ItemIndex(mapping.data.values())
    for query, expected in {
        "abyssal whip": "Abyssal whip",
        "saradomin brew": "Saradomin brew(4)",
        "super combat potion": "Super combat potion(4)",
        "dragon dagger p++": "Dragon dagger(p++)",
        "tbow": "Twisted bow",
    }.items():
        result = live.resolve(query)
        assert result.item and result.item.name == expected, query
    missing = [nick for nick, target in ITEM_ALIASES.items() if live.resolve(nick).item is None]
    assert not missing, f"alias targets no longer exist in /mapping: {missing}"
