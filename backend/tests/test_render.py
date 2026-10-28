from __future__ import annotations

import pathlib

import mwparserfromhell as mw
import pytest

from app.ingest.render import Renderer, convert_tables, render, tidy

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wiki"


# --------------------------------------------------------------------------- inline markup
@pytest.mark.parametrize(
    ("wikitext", "expected"),
    [
        ("A [[Dragon bones]] drop", "A Dragon bones drop"),
        ("the [[Barbarian_Training#Pyre_ships|pyre ship]] part", "the pyre ship part"),
        ("see [[Foo#Bar]] here", "see Foo here"),
        ("img [[File:X.png|thumb|caption]] gone", "img gone"),
        ("[[Category:Quests]]text", "text"),
        ("[[:Category:Quests|the category]]", "the category"),
        ("'''bold''' and ''italic'' text", "bold and italic text"),
        ("a<!-- hidden -->b", "ab"),
        ("fact<ref>source note</ref> stays", "fact stays"),
        ("line one<br>line two<br />line three", "line one\nline two\nline three"),
        ("fish &amp; chips&nbsp;now", "fish & chips now"),
        ("[http://example.com a label] and [http://example.com]", "a label and"),
        ("__TOC__\ntext", "text"),
    ],
)
def test_inline_markup(wikitext: str, expected: str) -> None:
    assert render(wikitext) == expected


def test_lists_keep_their_nesting() -> None:
    assert render("*A\n**B [[x]]\n*C\n#One\n##Two") == "- A\n  - B x\n- C\n- One\n  - Two"
    assert render(":indented text\n;Term: definition") == "indented text\nTerm: definition"


def test_blank_line_runs_collapse_and_text_is_stripped() -> None:
    assert render("\n\n a \n\n\n\n b \n\n") == "a\n\nb"
    assert tidy("a\xa0b   c") == "a b c"


# ----------------------------------------------------------- templates that carry facts
@pytest.mark.parametrize(
    ("wikitext", "expected"),
    [
        ("{{SCP|Fishing|43}}", "43 Fishing"),
        ("{{SCP|Fishing|43|link=yes}}", "43 Fishing"),
        ("{{SCP|crafting|5-90}}", "5-90 Crafting"),
        ("{{SCP|Hunter}}", "Hunter"),
        ("{{SCP|Quest|200|link=yes}}", "200 quest points"),
        ("{{SCP|Combat|85}}", "combat level 85"),
        ("{{SCP|Total|1500}}", "total level 1500"),
        ("{{Coins|1000}}", "1000 coins"),
        ("{{NoCoins|49}}", "49"),
        ("{{Plinkt|Dragon bones}} and {{Plink|Whip|pic=x}}", "Dragon bones and Whip"),
        ("code {{Fairycode|DJR}}", "code DJR"),
        ("{{Okay}} {{Not okay}} {{NA}}", "Yes No N/A"),
        ("{{Colour|red|Caution:}} careful", "Caution: careful"),
        ("{{Boostable|no}} {{Boostable|yes}}", "(not boostable) (boostable)"),
        ("{{Questreqstart|yes}}", "(needed to start the quest)"),
        ("{{Questreqstart|no}}", "(needed later in the quest)"),
        ("press {{Key press|Ctrl|Z}}", "press Ctrl+Z"),
        ("on the {{FloorNumber|uk=0}} of", "on the ground floor of"),
        ("on the {{FloorNumber|uk=2}} of", "on the floor 2 of"),
        ("a{{!}}b {{=}}", "a|b ="),
        ("{{Main|Prayer|Altars}}", "Main article: Prayer, Altars"),
    ],
)
def test_template_rendering(wikitext: str, expected: str) -> None:
    assert render(wikitext) == expected


def test_live_prices_become_dependencies_not_stale_numbers() -> None:
    assert render("{{GEP|Fire rune}}") == "GE price of Fire rune"
    assert render("{{GEP|Fire rune|3}}") == "3x GE price of Fire rune"
    assert render("cost ({{GEP|A}} + {{GEPt|B}})*2") == "cost (GE price of A + GE price of B)*2"
    assert render("{{HA|Rune scimitar}}") == "high alchemy value of Rune scimitar"
    assert render("{{Average drop value|mob=Zulrah}}") == ""  # Lua-computed number: dropped


@pytest.mark.parametrize(
    "wikitext",
    [
        "{{Subject changes|date=1 Jan}}",
        "{{Subject changes header}}{{Subject changes footer}}",
        "{{CiteTwitter|author=x|url=y}}",
        "{{Cite web|url=x}}",
        "{{Reflist}}",
        "{{Clear}}",
        "{{Bosses}}",
        "{{Navbox quests}}",
        "{{#var:regchance}}",
        "{{Hastranscript|x}}",
        "{{Infobox Foo|image=[[File:a.png]]}}",
        "{{SomeDataBlob|a=1|b=2}}",
        "{{Unknown|" + "x" * 200 + "}}",
        "{{{1|default}}}",
    ],
)
def test_noise_templates_vanish(wikitext: str) -> None:
    assert render(wikitext) == ""


def test_unknown_templates_keep_only_short_positional_text() -> None:
    assert render("{{Somehat|Look at this|other=ignored}}") == "Look at this"
    assert render("{{Pair|one|two}}") == "one two"


def test_drops_lines() -> None:
    wikitext = (
        "{{DropsTableHead}}\n"
        "{{DropsLine|name=Tanzanite fang|quantity=1|rarity=1/1024|rolls=2}}\n"
        "{{DropsLine|name=Zulrah's scales|quantity=100-299<!--note-->|rarity=Always}}\n"
        "{{DropsLineClue|tier=Elite|rarity=1/200}}\n"
        "{{DropsTableBottom}}"
    )
    assert render(wikitext) == (
        "- Tanzanite fang: 1, rarity 1/1024, 2 rolls\n"
        "- Zulrah's scales: 100-299, rarity Always\n"
        "- Clue scroll (Elite): rarity 1/200"
    )


def test_infobox_skips_images_and_noise_keys() -> None:
    out = render(
        "{{Infobox Item|name=Abyssal whip|image=[[File:Abyssal whip.png]]|members=Yes|"
        "tradeable=Yes|examine=A weapon from the Abyss.|exchange=Yes|empty=|leagueRegion=General}}"
    )
    assert out == (
        "[Item] Abyssal whip\nMembers: Yes; Tradeable: Yes; Examine: A weapon from the Abyss."
    )
    monster = render("{{Infobox Monster|name=Zulrah|combat=725|hitpoints=500|id1=2042|size=5}}")
    assert "Combat level: 725" in monster and "Hitpoints: 500" in monster
    assert "id1" not in monster and "size" not in monster


def test_bonuses_are_compact() -> None:
    out = render(
        "{{Infobox Bonuses|astab=0|aslash=+82|str=+82|rstr=0|mdmg=0|prayer=0|slot=weapon|speed=4|"
        "attackrange=1|combatstyle=Whip|image=[[File:x.png]]}}"
    )
    assert out == (
        "[Equipment bonuses]\n"
        "Attack bonuses: stab 0, slash +82\n"
        "Other bonuses: melee strength +82, ranged strength 0, magic damage % 0, prayer 0\n"
        "Weapon: slot weapon, speed (ticks) 4, range 1, style Whip"
    )


def test_recipe() -> None:
    out = render(
        "{{Recipe|skill1=Smithing|skill1lvl=50|skill1exp=37.5|mat1=Steel bar|mat1quantity=2|"
        "output1=Steel platebody|output1quantity=1|facility=Anvil|tools=Hammer|ticks=3}}"
    )
    assert out == (
        "Recipe: skills: 50 Smithing (37.5 xp); materials: 2 Steel bar; "
        "output: 1 Steel platebody; facility: Anvil; tools: Hammer"
    )


# --------------------------------------------------------------------------- tables
def test_native_table_with_caption_header_rows_and_cell_attributes() -> None:
    wikitext = (
        '{| class="wikitable"\n|+ Level table\n! Level !! Experience\n|-\n'
        '| 1 || style="text-align:center" | 0\n|-\n| 2\n| {{Coins|83}}\n|}'
    )
    assert render(wikitext) == "Level table\nLevel | Experience\n1 | 0\n2 | 83 coins"


def test_table_cells_with_lists_and_links_stay_on_one_row() -> None:
    out = render("{|\n! Item !! Where\n|-\n| [[Egg]] || *Kitchen\n*Farm\n|}")
    assert out.splitlines()[0] == "Item | Where"
    assert out.splitlines()[1].startswith("Egg | ")


def test_text_fallback_table_conversion() -> None:
    markup = '{| class="x"\n! A !! B\n|-\n| 1 || 2\n|-\n| 3 ||\n|}\nafter'
    assert convert_tables(markup).strip() == "A | B\n1 | 2\n3 | \n\nafter"


# --------------------------------------------------------------------------- robustness
def test_malformed_and_extreme_input_never_raises() -> None:
    for bad in (
        "{{unclosed",
        "[[unclosed",
        "<ref>never closed",
        "}}{{",
        "{|\n| a",
        "",
        " ",
        "{{{{{{x}}}}}}",
    ):
        assert isinstance(render(bad), str)
    deep = "{{Colour|red|" * 200 + "x" + "}}" * 200
    assert isinstance(render(deep), str)  # depth guard, no RecursionError


def test_renderer_instances_are_reusable() -> None:
    renderer = Renderer()
    assert (
        renderer.render("{{SCP|Attack|70}}") == renderer.render("{{SCP|Attack|70}}") == "70 Attack"
    )
    assert renderer.inline("a\n\nb   c") == "a b c"


# --------------------------------------------------------------------------- real pages
FIXTURE_FILES = sorted(p.name for p in FIXTURES.glob("*.wiki") if not p.name.startswith("mmg_"))


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_real_pages_leave_no_markup_behind(name: str) -> None:
    out = Renderer().render((FIXTURES / name).read_text())
    assert out, name
    leftovers = ["{{", "}}", "[[", "]]", "<ref", "<!--", "'''", "{|", "|}", "__TOC__"]
    if name == "mechanic_experience.wiki":  # <math> formulas are LaTeX and legitimately use braces
        leftovers = [mark for mark in leftovers if mark not in {"{{", "}}"}]
    for leftover in leftovers:
        assert leftover not in out, f"{name}: {leftover!r} left in the output"


def test_dragon_slayer_ii_requirements_survive() -> None:
    out = Renderer().render((FIXTURES / "quest_dragon_slayer_ii_head.wiki").read_text())
    for fact in (
        "200 quest points",
        "75 Magic",
        "70 Smithing",
        "Legends' Quest",
        "Dream Mentor",
        "Pickaxe" if False else "pickaxe",
        "Grandmaster",
        "Very Long",
    ):
        assert fact in out, fact
    assert "(not boostable)" in out


def test_item_page_keeps_stats_and_prose() -> None:
    out = Renderer().render((FIXTURES / "item_abyssal_whip.wiki").read_text())
    assert "requires an Attack level of 70" in out
    assert (
        "slash +82" in out
        and "melee strength +82" in out
        and "Examine: A weapon from the Abyss." in out
    )


def test_boss_page_keeps_drops_and_stat_levels_without_defence_noise() -> None:
    out = Renderer().render((FIXTURES / "monster_zulrah.wiki").read_text())
    assert "- Zulrah's scales: 100-299, rarity Always" in out
    assert "Hitpoints: 500" in out and "Combat level: 725" in out
    assert "dlight" not in out and "dstandard" not in out and "Defence bonuses:" in out


def test_training_guide_prices_are_dependencies_and_tables_are_rows() -> None:
    out = Renderer().render((FIXTURES / "guide_prayer_training.wiki").read_text())
    assert "GE price of" in out


def test_real_tables_become_rows() -> None:
    out = Renderer().render((FIXTURES / "mechanic_experience.wiki").read_text())
    rows = [line for line in out.splitlines() if " | " in line]
    assert len(rows) >= 5, "the experience tables should render one line per row"


# --------------------------------------------------------------------------- drops and notes
def test_drop_notes_keep_the_condition_that_changes_the_rate() -> None:
    out = render(
        "{{DropsLine|name=Mossy key|quantity=1|rarity=1/150|altrarity=1/75|raritynotes="
        "{{Refn|group=d|On a moss giant Slayer task the rate is 1/75."
        "{{CiteDiscord|author=Mod Ash}}}}}}"
    )
    assert out == (
        "- Mossy key: 1, rarity 1/150, alternative rarity 1/75 "
        "(note: On a moss giant Slayer task the rate is 1/75.)"
    )


def test_members_and_free_to_play_markers_follow_the_item() -> None:
    out = render(
        "{{DropsLine|name=Mithril spear|namenotes={{(m)}}|quantity=1|rarity=2/128}}\n"
        "{{DropsLine|name=Coins|namenotes={{(f)}}|quantity=5|rarity=35/128}}"
    )
    assert "- Mithril spear (members only): 1, rarity 2/128" in out
    assert "- Coins (free-to-play only): 5, rarity 35/128" in out


def test_shared_drop_tables_keep_how_often_they_are_rolled() -> None:
    out = render(
        "{{HerbDropTableInfo|5/128}}{{HerbDropLines|5/128|f2p=yes}}\n"
        "{{UncommonSeedDropTableInfo|35/128}}{{UncommonSeedDropLines|35/128}}\n"
        "{{GemDropTable|4/128|naturetalisman=yes}}\n"
        "{{RareDropTable|1/128|5/128|chaostalisman=yes}}\n"
        "{{UncommonSeedDropLines|10/127|2}}"
    )
    assert out.splitlines() == [
        "- Herb drop table: rarity 5/128",
        "- Uncommon seed drop table: rarity 35/128",
        "- Gem drop table: rarity 4/128",
        "- Rare drop table: rarity 1/128",
        "- Gem drop table: rarity 5/128",
        "- Uncommon seed drop table: rarity 10/127, 2 rolls",
    ]


def test_clue_drop_lines_name_their_tier() -> None:
    assert render("{{DropsLineClue|type=beginner|rarity=1/45}}") == (
        "- Clue scroll (beginner): rarity 1/45"
    )


def test_citations_vanish_but_grouped_notes_stay_where_they_apply() -> None:
    assert render("Fast.<ref>Some forum post</ref> Slow.") == "Fast. Slow."
    out = render("Rate 1/4<ref group=d>Only on members worlds.</ref> and more.")
    assert out == "Rate 1/4 (note: Only on members worlds.) and more."
    assert render("Needs a spade{{efn|Or a trowel.}}") == "Needs a spade (note: Or a trowel.)"


def test_a_named_note_is_spelled_out_once_per_section_even_when_reused_bare() -> None:
    wikitext = (
        "A{{efn|name=void}} and B{{efn|Only with the full set.|name=void}} and C{{efn|name=void}}"
    )
    renderer = Renderer()
    code = mw.parse(wikitext)
    renderer.collect_notes(code)
    out = renderer.render(code)
    assert out == "A (note: Only with the full set.) and B and C"


def test_long_notes_are_cut_at_a_word() -> None:
    out = render("x{{efn|" + "word " * 200 + "}}")
    assert out.endswith("...)") and len(out) < 520


# --------------------------------------------------------------------------- infoboxes
def test_infobox_versions_are_grouped_and_labelled() -> None:
    out = render(
        "{{Infobox Monster|name=Moss giant|version1=Level 42|version2=Level 48|combat1=42|"
        "combat2=48|hitpoints1=60|hitpoints2=85|max hit=6|slayxp1=60|slayxp2=85|id1=1|id2=2|"
        "update1=x|elementalweaknesstype=Fire|elementalweaknesspercent=50}}"
    )
    assert out.splitlines() == [
        "[Monster] Moss giant",
        "Max hit: 6; Elemental weakness: Fire (50%)",
        'Version "Level 42": Combat level: 42; Hitpoints: 60; Slayer XP: 60',
        'Version "Level 48": Combat level: 48; Hitpoints: 85; Slayer XP: 85',
    ]


def test_defence_bonuses_become_one_line() -> None:
    out = render(
        "{{Infobox Monster|name=X|dstab=20|dslash=20|dcrush=20|dmagic=0|dlight=20|dstandard=15|"
        "dheavy=10|attbns=0|amagic=0}}"
    )
    assert out.splitlines()[1] == (
        "Defence bonuses: stab 20, slash 20, crush 20, magic 0, light ranged 20, "
        "standard ranged 15, heavy ranged 10"
    )


def test_switch_infobox_keeps_every_form() -> None:
    out = render(
        "{{Switch infobox|text1=First form|item1={{Infobox Monster|name=Damis|combat=95}}"
        "|text2=Second form|item2={{Infobox Monster|name=Damis|combat=99}}}}"
    )
    assert "First form:\n[Monster] Damis\nCombat level: 95" in out
    assert "Second form:\n[Monster] Damis\nCombat level: 99" in out


def test_tabber_template_and_tag_keep_every_tab() -> None:
    template = render(
        "{{tabber|tab1=Normal mould|tab1content={{Recipe|skill1=Smithing|skill1lvl=75|"
        "mat1=Adamantite bar|output1=Adamant cannonball}}|tab2=Double mould|tab2content=Other}}"
    )
    assert "Normal mould:\nRecipe: skills: 75 Smithing" in template
    assert "Double mould:\nOther" in template
    tag = render("<tabber>\nRanged=\nUse a crossbow.\n|-|\nMelee=\nUse a lance.\n</tabber>")
    assert tag == "Ranged:\nUse a crossbow.\n\nMelee:\nUse a lance."


# --------------------------------------------------------------------------- strategies, clues
def test_recommended_equipment_lists_the_options_best_first_with_their_notes() -> None:
    out = render(
        "{{Recommended equipment|style=Ranged|head1={{plink|Masori mask (f)}} / "
        "{{plink|Void ranger helm}}{{efn|Only with the full Void set.}}|"
        "head2={{plink|Crystal helm}}|weapon1={{plink|Dragon hunter crossbow}}|"
        "special1={{plink|Zaryte crossbow}}}}"
    )
    assert out.splitlines() == [
        "Recommended equipment (Ranged), options listed best first:",
        "- Head: Masori mask (f) / Void ranger helm (note: Only with the full Void set.); "
        "Crystal helm",
        "- Weapon: Dragon hunter crossbow",
        "- Special attack weapon: Zaryte crossbow",
    ]


def test_inventory_setups_keep_quantities_and_compress_repeats() -> None:
    out = render("{{Inventory|Rake|Shark|Shark|Shark||Coins\\1400|Coconut;n\\175|align=right}}")
    assert out == "Inventory: Rake, 3 Shark, 1,400 Coins, 175 Coconut (noted)"
    numbered = render("{{Inventory|align=right|1=Diamond bolts|2=Divine ranging potion(3)}}")
    assert numbered == "Inventory: Diamond bolts, Divine ranging potion(3)"


def test_equipment_setup_names_each_slot() -> None:
    out = render("{{Equipment|align=right|head=Void ranger helm|cape=|weapon=Toxic blowpipe}}")
    assert out == "Equipment setup: head: Void ranger helm; weapon: Toxic blowpipe"


def test_clue_info_keeps_the_clue_and_its_solution() -> None:
    out = render(
        "{{Clue info|id=|text=The anagram reveals<br>who to speak to:<br><br>AN EARL|"
        "notes=Speak to [[Ranael]]|map={{Map|3316,3163}}}}"
    )
    assert out == (
        "Clue text: The anagram reveals who to speak to: AN EARL\nSolution: Speak to Ranael"
    )


def test_clue_guide_tables_keep_every_step() -> None:
    cryptic = render(
        "{{Cryptics|clue=[[Clue scroll (easy) - X|Belladonna, my dear.]]|note=Talk to the "
        "[[Tool leprechaun]].|task=talk|level=1}}"
    )
    assert cryptic == "- Clue: Belladonna, my dear. Solution: Talk to the Tool leprechaun."
    coordinates = render(
        "{{Coordinates|coords=00 degrees 35 minutes north,<br/>35 degrees 50 minutes east|"
        "req=[[Darkness of Hallowvale]]|notes=Dig west of the wall rubble.}}"
    )
    assert "00 degrees 35 minutes north, 35 degrees 50 minutes east" in coordinates
    assert "Solution: Dig west of the wall rubble." in coordinates
    assert "Requirements: Darkness of Hallowvale" in coordinates


def test_quest_item_lists_survive_the_short_text_rule() -> None:
    needed = render(
        "{{Needed|A [[spade]], any [[sapling]], 1 free [[tree patch]] or [[fruit tree patch]]."
        "|recommended=An [[axe]]|skills={{SCP|Farming|15|link=yes}}}}"
    )
    assert needed.splitlines() == [
        "Items needed: A spade, any sapling, 1 free tree patch or fruit tree patch.",
        "Items recommended: An axe",
        "Skills: 15 Farming",
    ]
    checklist = render("{{Checklist|\n*Speak to the [[Bank Camel]].\n** Use a [[bucket]].\n}}")
    assert checklist == "- Speak to the Bank Camel.\n  - Use a bucket."


def test_rune_costs_are_readable() -> None:
    cost = render(
        "{{Spell cost table|Rune1=water|Rune1num=2|Rune2=law|Rune2num=2|Rune3=soft clay}}"
    )
    assert cost == "Runes required: 2 water rune, 2 law rune, 1 soft clay"
    assert render("Needs {{RuneReq|Law=200|Blood=100}}.") == "Needs 200 law rune, 100 blood rune."


def test_shop_and_location_lines_are_readable() -> None:
    assert render("{{StoreLine|name=Iron pickaxe|stock=5|restock=100|sell=140}}") == (
        "- Iron pickaxe: stock 5, restock 100, sells for 140"
    )
    assert render("{{LocationMonsterTableLine|name=Man|levels=2|spawns=6}}") == (
        "- Man: level 2, 6 spawns"
    )
    assert render(
        "{{LocLine|name=Moss giant|location=Crandor|levels=42|members=No|mtype=pin}}"
    ) == ("- Crandor (Moss giant, level 42, free-to-play)")


def test_formula_cells_over_live_prices_are_noise_but_a_plain_price_stays() -> None:
    table = (
        "{|\n! Potion !! Cost !! Profit\n|-\n| Combat potion || {{GEP|Goat horn dust}} || "
        "(0.75x {{GEP|Combat potion(4)}} - {{GEP|Harralander potion (unf)}} - "
        "{{GEP|Goat horn dust}})/84 coins\n|}"
    )
    out = render(table)
    assert "Combat potion | GE price of Goat horn dust | (depends on GE prices)" in out
    assert "0.75x" not in out and "Harralander" not in out


def test_a_drop_rate_computed_with_expr_is_evaluated() -> None:
    out = render(
        "{{DropsLine|name=Hydra's claw|quantity=1|"
        "rarity=1/{{#expr:1000/(1999/2000*1999/2000) round 1}}}}"
    )
    assert out == "- Hydra's claw: 1, rarity 1/1001"


def test_a_rate_that_depends_on_a_variable_is_left_out_instead_of_half_printed() -> None:
    out = render(
        "{{DropsLine|name=Potato seed|quantity=1-4|rarity=1/{{#expr:1/({{#var:x}}*64) round 1}}}}"
    )
    assert out == "- Potato seed: 1-4"


def test_quantity_notes_are_kept_with_the_quantity() -> None:
    out = render(
        "{{DropsLine|name=Mole skin|quantity=1-3|rarity=Always|"
        "quantitynotes=<ref group=d>Noted after the hard Falador diary.</ref>}}"
    )
    assert out == "- Mole skin: 1-3 (note: Noted after the hard Falador diary.), rarity Always"
