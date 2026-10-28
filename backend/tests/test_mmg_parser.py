from __future__ import annotations

import pathlib

import pytest

from app.ingest.mmg import MoneyMakingGuide, parse_money_making_guide
from app.ingest.render import flatten_parser_functions

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wiki"


def fixture(name: str) -> MoneyMakingGuide:
    text = (FIXTURES / name).read_text()
    title = text.split("-->", 1)[0].removeprefix("<!-- title: ").strip()
    guide = parse_money_making_guide(title, text)
    assert guide is not None, name
    return guide


def parse(body: str, header: str = "") -> MoneyMakingGuide:
    guide = parse_money_making_guide(
        "Money making guide/Test", "{{Mmgtable\n|Activity = Testing\n" + header + body + "\n}}"
    )
    assert guide is not None
    return guide


def by_name(items, name):
    return next(i for i in items if i.name == name)


# --------------------------------------------------------------------------- real guides
def test_per_kill_guide_aerial_fishing() -> None:
    g = fixture("mmg_plain_per_kill.wiki")
    assert (g.kind, g.is_per_kill, g.kph, g.kph_label) == (
        "standard",
        True,
        2200.0,
        "Creatures caught per hour",
    )
    assert g.members_only is True and g.category == "Skilling"
    assert g.skill_requirements == {
        "Fishing": 43,
        "Hunter": 35,
    }  # "(99 recommended)" is not required
    fish, lure = g.items_in[0], g.items_out[0]
    # 0.286 offcuts and 2 lures PER ACTION x 2200 actions/hour; the wiki's text says 4,400 lures.
    assert fish.qty == pytest.approx(0.286) and fish.qty_per_hour == pytest.approx(629.2)
    assert lure.name == "Shark lure" and lure.qty_per_hour == pytest.approx(4400.0)
    assert fish.value_expr is None and fish.fixed_value is None  # {{GEP|<itself>}} = no override
    assert g.rankable and not g.unsupported


def test_per_hour_guide_without_kph() -> None:
    g = fixture("mmg_no_kph.wiki")
    assert g.is_per_kill is False and g.kph is None
    assert all(i.is_per_hour and i.qty_per_hour == i.qty for i in g.items_in + g.items_out)


def test_recurring_guide_is_per_instance_and_converted_with_activity_time() -> None:
    g = fixture("mmg_recurring.wiki")
    assert g.kind == "recurring" and g.activity_minutes == 1.4 and g.recurrence == "50 minutes"
    logs = by_name(g.items_in, "Redwood logs")
    assert logs.qty == 4 and logs.is_per_hour is False
    assert logs.qty_per_hour == pytest.approx(4 * 60 / 1.4)  # (output - input) * 60 / Activity Time
    nest = by_name(g.items_out, "Bird nest (empty)")
    assert nest.qty == pytest.approx(4 * 5 * 0.525 * (60 / 95))  # '4*5*.525*(60/95)'
    # Its outputs use {{Average drop value}}: computed in Lua by the wiki, so not reproducible.
    assert not g.rankable and any("Average drop value" in u for u in g.unsupported)
    pendant = by_name(g.items_in, "Digsite pendant")
    assert pendant.value_expr == 'gep("Ruby necklace", 1) + gep("Cosmic rune", 1)'
    assert g.quest_requirements == ["Bone Voyage"]
    assert g.skill_requirements == {}  # every skill line there is a recommendation or ironman-only


def test_price_formulas_are_stored_as_text_for_live_valuation() -> None:
    g = fixture("mmg_gep_value.wiki")
    formulas = [i.value_expr for i in g.items_in + g.items_out if i.value_expr]
    assert formulas, "this guide values items with {{GEP}} formulas"
    assert all("gep(" in f for f in formulas)
    assert all(i.fixed_value is None for i in g.items_in + g.items_out if i.value_expr)


def test_lua_computed_values_make_a_guide_unsupported_not_wrong() -> None:
    g = fixture("mmg_gdt_value.wiki")
    assert not g.rankable
    assert any("GDTValue" in u for u in g.unsupported)


def test_high_alchemy_values_become_ha_calls() -> None:
    g = fixture("mmg_ha_value.wiki")
    assert any(i.value_expr and "ha(" in i.value_expr for i in g.items_out + g.items_in)


def test_guide_using_expr_wrappers_evaluates_quantities() -> None:
    g = fixture("mmg_expr_wrapped.wiki")
    assert g.items_out and all(i.qty > 0 for i in g.items_out)
    assert any(i.qty not in (1.0, 2.0, 3.0) for i in g.items_out)  # fractional drop rates


def test_guide_with_variables_parses_without_crashing() -> None:
    g = fixture("mmg_uses_var.wiki")
    assert g.items_out
    assert g.kph is None or g.kph > 0


# --------------------------------------------------------------------------- quantities and flags
def test_quantity_forms_seen_on_the_wiki() -> None:
    g = parse(
        "|isperkill = y\n|kph = 10\n"
        "|Output1 = A\n|Output1num = {{#expr:1*(1/720)}}\n"
        "|Output2 = B\n|Output2num = 4*0.7852\n"
        "|Output3 = C\n|Output3num = .525\n"
        "|Output4 = D\n|Output4num = ((1/3)*4*(5/6)) + 2\n"
        "|Output5 = E\n"  # no num: defaults to 1
    )
    assert [round(i.qty, 6) for i in g.items_out] == [
        round(1 / 720, 6),
        round(4 * 0.7852, 6),
        0.525,
        round(1 / 3 * 4 * 5 / 6 + 2, 6),
        1.0,
    ]
    assert g.items_out[0].qty_per_hour == pytest.approx(10 / 720)


def test_variables_via_vardefine_var_and_vardefineecho() -> None:
    g = parse(
        "|isperkill = y\n|kph = {{#vardefineecho:catches|287}}\n"
        "{{#vardefine:reg|{{#expr:0.5*2}}}}\n"
        "|Output1 = A\n|Output1num = {{#expr:4*{{#var:reg}}}}\n"
    )
    assert g.kph == 287 and g.items_out[0].qty == 4.0


def test_unresolved_variable_is_unsupported() -> None:
    g = parse("|Output1 = A\n|Output1num = {{#var:never_defined}}\n")
    assert not g.rankable and any("never_defined" in u for u in g.unsupported)


@pytest.mark.parametrize(
    ("flag", "per_hour"), [("y", True), ("Yes", True), ("true", True), ("n", False), ("", False)]
)
def test_isph_flag_variants(flag: str, per_hour: bool) -> None:
    g = parse(f"|isperkill = y\n|kph = 100\n|Output1 = A\n|Output1num = 2\n|Output1isph = {flag}\n")
    item = g.items_out[0]
    assert item.is_per_hour is per_hour
    assert item.qty_per_hour == (2.0 if per_hour else 200.0)


def test_kph_without_a_number_defaults_to_one_like_the_wiki_and_says_so() -> None:
    g = parse("|isperkill = y\n|Output1 = A\n|Output1num = 3\n")
    assert g.kph == 1.0 and g.items_out[0].qty_per_hour == 3.0
    assert any("defaults to 1" in i for i in g.issues)


def test_comments_in_values_are_ignored() -> None:
    g = parse(
        "|isperkill = y<!-- see talk -->\n"
        "|kph = 4000<!--\n26 tiles from the bank-->\n"
        "|Output1 = A\n"
    )
    assert g.is_per_kill and g.kph == 4000.0


def test_items_stop_at_the_first_missing_entry_like_the_module() -> None:
    g = parse("|Output1 = A\n|Output2 = B\n|Output4 = D\n")  # no Output3: D is ignored
    assert [i.name for i in g.items_out] == ["A", "B"]


# --------------------------------------------------------------------------- values
def test_value_overrides() -> None:
    g = parse(
        "|Input1 = Coins\n|Input1num = 72000\n|Input1value = 1\n"
        "|Input2 = Fish offcuts\n|Input2value = {{GEP|Fish offcuts}}\n"  # itself: not an override
        "|Output1 = Bundle\n"
        "|Output1value = {{GEP|Fire rune|3}} + {{GEP|Blood rune}} + {{HA|Rune scimitar}}\n"
        "|Output2 = Half\n|Output2value = {{#expr:{{GEP|Magus ring}} - 3*{{GEP|Chromium ingot}}}}\n"
    )
    coins, offcuts = g.items_in
    assert coins.fixed_value == 1.0 and coins.value_expr is None
    assert offcuts.value_expr is None and offcuts.fixed_value is None
    assert (
        g.items_out[0].value_expr
        == 'gep("Fire rune", 3) + gep("Blood rune", 1) + ha("Rune scimitar")'
    )
    assert g.items_out[1].value_expr == '(gep("Magus ring", 1) - 3*gep("Chromium ingot", 1))'
    assert g.rankable


@pytest.mark.parametrize(
    "template",
    [
        "{{GDTValue}}",
        "{{RDTValue}}",
        "{{SeedPackValue|1|spiritseed=Yes}}",
        "{{Average drop value|mob=X|raw=y}}",
    ],
)
def test_values_computed_by_lua_are_unsupported(template: str) -> None:
    g = parse(f"|Output1 = A\n|Output1value = {template}\n")
    assert not g.rankable and g.unsupported


def test_cheap_food_is_a_choice_among_named_items() -> None:
    g = parse("|Input1 = {{Cheap food}}\n|Input1num = 1.5\n|Output1 = A\n")
    food = g.items_in[0]
    assert food.cheapest_of == ["Manta ray", "Tuna potato", "Anglerfish", "Dark crab", "Bluefin"]
    assert g.rankable


def test_computed_item_names_are_flagged_not_silently_dropped() -> None:
    g = parse("|Input1 = {{#var:best_input}}\n|Output1 = A\n")
    assert not g.rankable and any("Input1 name is computed" in u for u in g.unsupported)
    g2 = parse("|Output1 = {{#var:MaxProfit_item}}\n")
    assert not g2.rankable and any("Output1 name is computed" in u for u in g2.unsupported)
    assert not any("no outputs" in u for u in g2.unsupported)  # the real reason is reported


def test_multiline_parser_functions_do_not_truncate_the_template() -> None:
    # Regression: a multi-line {{#expr:}} used to terminate {{Mmgtable}} early, dropping every
    # output after it (and their requirements).
    body = (
        "|Input1 = A\n|Input1num = 1\n\n"
        "{{#vardefine:total|{{#expr:\n  {{GEP|X|1}}*(1/5) +\n  {{GEP|Y|100}}*(1/5)\n}}}}\n\n\n"
        "|Output1 = Magus ring\n|Output1num = {{#expr:1*(1/720)}}\n|Output2 = Eye\n"
    )
    g = parse(body)
    assert [i.name for i in g.items_out] == ["Magus ring", "Eye"]


def test_flatten_parser_functions() -> None:
    src = "a {{#expr:\n 1 +\n 2}} b\nkeep\nlines {{Plain|x\ny}} {{#if:\n{{#expr:1\n+1}}|t}} end"
    out = flatten_parser_functions(src)
    assert out == "a {{#expr: 1 + 2}} b\nkeep\nlines {{Plain|x\ny}} {{#if: {{#expr:1 +1}}|t}} end"
    assert flatten_parser_functions("no functions\nhere") == "no functions\nhere"
    assert isinstance(flatten_parser_functions("{{#expr:\nunterminated"), str)


# --------------------------------------------------------------------------- requirements
def test_skill_requirements_hard_vs_soft() -> None:
    g = parse(
        "|Skill =\n"
        "* {{SCP|Fishing|43}} (99 recommended)\n"
        "* {{SCP|Magic|49}} highly recommended for enchanting\n"
        "* {{SCP|Mining|60}} or optional extras\n"
        "* {{plink|Ironmen}}: {{SCP|construction|25}} to make a table\n"
        "* {{SCP|Slayer|94+}}\n"
        "* {{SCP|Quest|150}} and {{SCP|Combat|100}} and {{SCP|Total|1500}}\n"
        "* {{SCP|crafting|5-90}} if you wish to make your own\n"
    )
    assert g.skill_requirements == {"Fishing": 43, "Slayer": 94}
    assert (g.quest_points, g.combat_level, g.total_level) == (150, 100, 1500)
    assert any("49 Magic" in line for line in g.requirements_text)  # still shown as text


def test_requirements_packed_in_one_line_with_br_are_judged_one_by_one() -> None:
    # Seen on "Collecting steel platebodies (High Alchemy)": the hard Magic 55 sits next to a
    # recommendation in the same bullet and used to be dropped with it.
    g = parse(
        "|Skill = {{SCP|Magic|55}} for [[High Level Alchemy]]<br/>Decent {{SCP|Combat}}<br />"
        "{{SCP|Prayer|43}}+ recommended for [[Protect from Melee]]<BR>{{SCP|Defence|40}}\n"
    )
    assert g.skill_requirements == {"Magic": 55, "Defence": 40}  # Prayer is only recommended


def test_quest_requirements_are_the_linked_quests() -> None:
    g = parse(
        "|Quest =\n* [[Dragon Slayer II]]\n* Started [[Regicide|the quest]]\n"
        "* [[Quest points]]\n* None\n"
    )
    assert g.quest_requirements == ["Dragon Slayer II", "Regicide"]
    assert parse("|Quest = None\n").quest_requirements == []


def test_members_and_metadata_fields() -> None:
    g = parse(
        "|Members = Yes\n|Category = Combat/High\n|Intensity = Moderate\n"
        "|Location = [[Zulrah]]'s shrine\n"
    )
    assert (g.members_only, g.category, g.intensity, g.location) == (
        True,
        "Combat/High",
        "Moderate",
        "Zulrah's shrine",
    )
    assert parse("|Members = no\n").members_only is False
    assert parse("").members_only is None  # unknown stays unknown


def test_exclude_marks_a_guide_obsolete_like_the_wiki_does() -> None:
    # Module:Mmgtable: Exclude = yes keeps the guide out of the MMG lists and files it under
    # "Obsolete money making guides".
    assert parse("|Exclude = Yes\n").obsolete is True
    assert parse("|Exclude = no\n").obsolete is False
    assert parse("").obsolete is False


def test_capital_is_the_largest_coins_amount_in_the_item_line() -> None:
    assert parse("|Item = {{Coins|10000000}}+ recommended\n").capital == 10_000_000
    assert parse("|Item =\n* {{Coins|840,000}}\n* {{Coins|1500}}\n* [[Rune axe]]\n").capital == (
        840_000
    )
    assert parse("|Item = Decent [[armour]]\n").capital is None
    # amounts computed from live prices are not a fixed bankroll: ignored, never garbled
    formula = "|Item = {{Coins|{{#expr:2500*({{GEP|Plain pizza}}+1)/30000 round 0}}*10000}}+ rec.\n"
    assert parse(formula).capital is None
    assert parse("").capital is None


def test_only_guides_titled_ironman_are_known_to_suit_ironmen() -> None:
    body = "{{Mmgtable\n|Activity = Testing\n}}"
    green = parse_money_making_guide("Money making guide/Killing green dragons (Ironman)", body)
    plain = parse_money_making_guide("Money making guide/Killing green dragons", body)
    assert green is not None and green.ironman_friendly is True
    assert plain is not None and plain.ironman_friendly is None  # unknown, never False


def test_experience_lines_follow_the_same_per_hour_rules() -> None:
    g = parse(
        "|isperkill = y\n|kph = 100\n|Experience1 = Fishing\n|Experience1num = 5\n"
        "|Experience2 = Hunter\n|Experience2num = 800\n|Experience2isph = y\n"
    )
    fishing, hunter = g.experience
    assert fishing.qty_per_hour == 500 and hunter.qty_per_hour == 800


# --------------------------------------------------------------------------- page-level
def test_pages_without_a_guide_table_return_none() -> None:
    assert parse_money_making_guide("Money making guide", "{{MMG list}}\nJust an index.") is None


def test_table_inside_a_warning_wrapper_is_found() -> None:
    wikitext = (
        "{{Mmgwarning|This guide is outdated}}\n{{Mmgtable\n|Activity = Old thing\n|Output1 = X\n}}"
    )
    guide = parse_money_making_guide("Money making guide/Old thing", wikitext)
    assert guide is not None and guide.activity == "Old thing"


def test_activity_falls_back_to_the_page_title() -> None:
    guide = parse_money_making_guide(
        "Money making guide/Smelting gold", "{{Mmgtable\n|Output1 = X\n}}"
    )
    assert guide is not None and guide.activity == "Smelting gold"


# --------------------------------------------------------------------------- post-parse steps
def test_reference_profits_come_from_the_rendered_list_table() -> None:
    from app.ingest.mmg import attach_reference_profits, parse_reference_profits

    profits = parse_reference_profits((FIXTURES / "mmg_list_excerpt.html").read_text())
    assert profits["Money making guide/Killing the Doom of Mokhaiotl (Delve 1-16)"] == 13_925_040
    assert (
        profits["Money making guide/Killing Phosani's Nightmare"] == 11_922_148
    )  # &#39; unescaped
    assert len(profits) == 4 and all(isinstance(v, int) for v in profits.values())

    guide = parse("|Output1 = A\n")
    guide.page_title = "Money making guide/Killing Nex (Duo)"
    assert (
        attach_reference_profits([guide], profits) == 1 and guide.source_hourly_profit == 10_909_290
    )
    other = parse("|Output1 = A\n")
    other.page_title = "Money making guide/Not in the list"
    assert attach_reference_profits([other], profits) == 0 and other.source_hourly_profit is None


def test_resolve_item_ids_flags_items_that_cannot_be_priced() -> None:
    from app.clients.prices import ItemMeta
    from app.ingest.mmg import resolve_item_ids
    from app.tools.items import ItemIndex

    index = ItemIndex(
        [
            ItemMeta(id=1, name="Shark"),
            ItemMeta(id=2, name="Manta ray"),
            ItemMeta(id=3, name="Bluefin"),
        ]
    )
    guide = parse(
        "|Input1 = Shark\n|Input2 = {{Cheap food}}\n|Input3 = Resurrect Greater Ghost\n"
        "|Input3value = {{GEP|Shark|5}}\n|Input4 = Coins\n|Input4value = 1\n"
        "|Output1 = Shark\n|Output2 = Mystery drop\n"
    )
    resolve_item_ids(guide, index)
    assert guide.items_in[0].item_id == 1 and guide.items_out[0].item_id == 1
    assert guide.items_in[3].item_id == 995  # Coins is a special item, not in /mapping
    assert guide.items_in[2].item_id is None  # a spell: priced by its formula, no id needed
    assert guide.unsupported == ["no Grand Exchange price for 'Mystery drop' (not in /mapping)"]
    assert not guide.rankable


def test_cheap_food_with_no_priced_choice_is_unsupported() -> None:
    from app.clients.prices import ItemMeta
    from app.ingest.mmg import resolve_item_ids
    from app.tools.items import ItemIndex

    guide = parse("|Input1 = {{Cheap food}}\n|Output1 = Shark\n")
    resolve_item_ids(guide, ItemIndex([ItemMeta(id=1, name="Shark")]))
    assert any("Manta ray" in u for u in guide.unsupported)
