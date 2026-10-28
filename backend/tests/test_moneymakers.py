"""Money-making ranking (SPEC §8): eligibility from the player's snapshot and live-price valuation.

Every expected figure is worked out by hand in the comments. Market used throughout (/1h averages):

  Yew logs   sells 400 (5,000/h)   buys 380 (4,000/h)      tax: floor(400 * 2%) = 8
  Logs       sells  56 (25,000/h)  buys  50 (30,000/h)
  Plank      sells 160 (10,000/h)  buys 150 (9,000/h)      tax: floor(160 * 2%) = 3
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.api.schemas import SKILL_NAMES, PlayerState
from app.clients.prices import AveragePrice, ItemMeta, LatestPrice, Snapshot
from app.config import Settings
from app.ingest.mmg import GuideItem, MoneyMakingGuide
from app.tools.context import ToolContext
from app.tools.items import ItemIndex
from app.tools.moneymakers import (
    GuideStore,
    Market,
    MoneyConfig,
    Unpriced,
    check_requirements,
    ironman_list,
    rank_guides,
    value_guide,
)
from app.tools.registry import run_tool
from app.tools.tax import TaxPolicy

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
NOW_TS = int(NOW.timestamp())
FETCHED = datetime(2026, 10, 14, 11, 59, 30, tzinfo=UTC)

ITEMS = [
    ItemMeta(id=1515, name="Yew logs", members=True, limit=15_000),
    ItemMeta(id=1511, name="Logs", limit=15_000),
    ItemMeta(id=960, name="Plank", limit=10_000),
    ItemMeta(id=20997, name="Twisted bow", members=True, limit=8),
    ItemMeta(id=13190, name="Old school bond", limit=100),
    ItemMeta(id=4151, name="Abyssal whip", members=True, highalch=72_000),
    ItemMeta(id=391, name="Manta ray", members=True, limit=10_000),
    ItemMeta(id=13441, name="Anglerfish", members=True, limit=10_000),
    ItemMeta(id=9999, name="Rare thing", members=True, limit=10),
    ItemMeta(id=9998, name="Dead stock", limit=10),
]


def avg(high: float | None, high_volume: int, low: float | None, low_volume: int) -> AveragePrice:
    return AveragePrice(avg_high=high, high_volume=high_volume, avg_low=low, low_volume=low_volume)


HOURLY = {
    1515: avg(400.0, 5_000, 380.0, 4_000),
    1511: avg(56.0, 25_000, 50.0, 30_000),
    960: avg(160.0, 10_000, 150.0, 9_000),
    20997: avg(1_200_000_000.0, 3, 1_150_000_000.0, 3),
    13190: avg(11_700_000.0, 300, 11_668_000.0, 300),
    391: avg(1_900.0, 5_000, 1_800.0, 5_000),
    13441: avg(1_700.0, 3_000, 1_600.0, 3_000),
}
LATEST = {
    9999: LatestPrice(high=5_000, high_time=NOW_TS - 600, low=4_800, low_time=NOW_TS - 600),
    9998: LatestPrice(high=700, high_time=NOW_TS - 4_000, low=650, low_time=NOW_TS - 4_000),
}


def market(basis: str = "patient", **cfg) -> Market:
    settings = Settings(_env_file=None, prices_user_agent="osrs-agent-tests - tests@example.com")
    tax, _ = TaxPolicy.from_settings(settings, ITEMS)
    return Market(
        latest=LATEST,
        hourly=HOURLY,
        index=ItemIndex(ITEMS),
        tax=tax,
        now=NOW,
        cfg=MoneyConfig(basis=basis, **cfg),  # type: ignore[arg-type]
    )


def item(name: str, per_hour: float, item_id: int | None = None, **kw) -> GuideItem:
    return GuideItem(
        name=name, item_id=item_id, qty=per_hour, is_per_hour=True, qty_per_hour=per_hour, **kw
    )


def guide(title: str, outs: list[GuideItem], ins: list[GuideItem] | None = None, **kw):
    return MoneyMakingGuide(
        page_title=f"Money making guide/{title}",
        activity=title,
        items_out=outs,
        items_in=ins or [],
        **kw,
    )


def yew(per_hour: float = 300) -> GuideItem:
    return item("Yew logs", per_hour, 1515)


def player(members: bool | None = True, **overrides) -> PlayerState:
    state = {
        "account_type": "main",
        "members": members,
        "skills": dict.fromkeys(SKILL_NAMES, 70),
        "quests": {},
        "gp": 5_000_000,
    }
    return PlayerState.model_validate(state | overrides)


# =========================================================================== valuation
def test_profit_is_sell_price_net_of_tax_minus_buy_price_per_hour() -> None:
    # 100 Logs/h bought at 50 = 5,000; 100 Planks/h sold at 160, tax 3 -> 157 net = 15,700.
    g = guide("Planks", [item("Plank", 100, 960)], [item("Logs", 100, 1511)])
    v = value_guide(g, market())
    assert v.revenue_per_hour == pytest.approx(15_700)
    assert v.cost_per_hour == pytest.approx(5_000)
    assert v.profit_per_hour == pytest.approx(10_700)
    out = v.outputs[0]
    assert (out.unit_price, out.value_per_hour, out.price_source) == (160, 15_700, "1h average")


def test_price_basis_changes_the_prices_but_not_the_rules() -> None:
    g = guide("Yew", [yew()])
    # patient: sell at the 400 high side, tax 8 -> 392 * 300
    assert value_guide(g, market("patient")).profit_per_hour == pytest.approx(117_600)
    # mid: (400 + 380) / 2 = 390, tax floor(7.8) = 7 -> 383 * 300
    assert value_guide(g, market("mid")).profit_per_hour == pytest.approx(114_900)
    # instant: sell at the 380 low side, tax floor(7.6) = 7 -> 373 * 300
    assert value_guide(g, market("instant")).profit_per_hour == pytest.approx(111_900)


def test_inputs_are_bought_at_the_low_side_when_patient_and_the_high_side_when_instant() -> None:
    g = guide("Burn logs", [item("Plank", 100, 960)], [item("Logs", 100, 1511)])
    assert value_guide(g, market("patient")).cost_per_hour == pytest.approx(100 * 50)
    assert value_guide(g, market("instant")).cost_per_hour == pytest.approx(100 * 56)


def test_tax_is_capped_per_item_and_exempt_items_pay_none() -> None:
    # Twisted bow at 1.2e9: 2% = 24,000,000 is capped at 5,000,000 -> 1,195,000,000 net each.
    bow = guide("Bow", [item("Twisted bow", 0.001, 20997)])
    assert value_guide(bow, market()).revenue_per_hour == pytest.approx(1_195_000)
    # Old school bonds are exempt: the full 11,700,000.
    bond = guide("Bond", [item("Old school bond", 1, 13190)])
    assert value_guide(bond, market()).revenue_per_hour == pytest.approx(11_700_000)


def test_value_overrides_are_never_taxed() -> None:
    coins = guide("Loot", [item("Coins", 600_000, 995, fixed_value=1)])
    assert value_guide(coins, market()).profit_per_hour == pytest.approx(600_000)
    # a price formula: gep("Yew logs", 2) = 2 * 400 = 800 (sell side), ha = 72,000 -> 72,800 each
    formula = item("Mix", 10, None, value_expr='gep("Yew logs", 2) + ha("Abyssal whip")')
    assert value_guide(guide("Mix", [formula]), market()).revenue_per_hour == pytest.approx(728_000)


def test_coins_and_platinum_tokens_are_worth_their_face_value() -> None:
    g = guide("Face", [item("Platinum token", 2, 13204)], [item("Coins", 500, 995)])
    v = value_guide(g, market())
    assert v.revenue_per_hour == pytest.approx(2_000)  # 2 * 1,000, no tax on face value
    assert v.cost_per_hour == pytest.approx(500)


def test_cheapest_of_picks_the_cheapest_option_at_todays_prices() -> None:
    food = item("Cheap food", 24, None, cheapest_of=["Manta ray", "Anglerfish"])
    v = value_guide(guide("Food", [yew()], [food]), market())
    # Manta ray buys at 1,800, Anglerfish at 1,600 -> 24 * 1,600
    assert v.cost_per_hour == pytest.approx(38_400)
    assert "Anglerfish" in (v.inputs[0].note or "")


def test_a_guide_faster_than_the_market_is_scaled_to_the_pace_it_allows() -> None:
    # 6,000 Yew logs/h but 5,000/h trade on the high side; one player may take 25% = 1,250/h.
    # The whole guide scales by 1,250 / 6,000 and the profit is 1,250 * 392 = 490,000.
    v = value_guide(guide("Too much", [yew(6_000)]), market())
    assert v.pace == pytest.approx(1_250 / 6_000)
    assert v.profit_per_hour == pytest.approx(490_000)
    assert v.outputs[0].per_hour == pytest.approx(1_250)
    assert len(v.warnings) == 1 and "limited by the market" in v.warnings[0]
    assert "21%" in v.warnings[0] and "5,000/h trade" in v.warnings[0]


def test_the_scarcest_item_sets_the_pace_for_inputs_and_outputs_together() -> None:
    # Plank: 100/h of 10,000 sold -> fine. Logs: 10,000/h wanted, 30,000 bought, cap 7,500 -> 0.75
    g = guide("Burn", [item("Plank", 100, 960)], [item("Logs", 10_000, 1511)])
    v = value_guide(g, market())
    assert v.pace == pytest.approx(0.75)
    # outputs 75 * 157 = 11,775 ; inputs 7,500 * 50 = 375,000 : both scaled by the same 0.75
    assert v.revenue_per_hour == pytest.approx(11_775)
    assert v.cost_per_hour == pytest.approx(375_000)


def test_a_guide_within_the_market_is_untouched() -> None:
    v = value_guide(guide("Fine", [yew(300)]), market())  # 6% of the volume
    assert v.pace == 1.0 and v.warnings == []
    edge = value_guide(guide("Edge", [yew(1_250)]), market())  # exactly 25%
    assert edge.pace == 1.0 and edge.warnings == []


def test_the_market_share_is_configurable() -> None:
    v = value_guide(guide("Greedy", [yew(300)]), market(market_share=0.01))  # cap 50/h
    assert v.pace == pytest.approx(50 / 300)


def test_items_that_barely_matter_are_not_checked_for_liquidity() -> None:
    # "Rare thing" has no trades in the last hour (last-trade quote only): 0.004% of the value.
    tiny = item("Rare thing", 0.001, 9999)
    v = value_guide(guide("Mostly yew", [yew(), tiny]), market())
    assert v.revenue_per_hour == pytest.approx(117_600 + 0.001 * (5_000 - 100))  # tax 100
    assert any(line.price_source == "last trade (10 min ago)" for line in v.outputs)
    # ...but when it IS the guide, a last-trade-only quote is not enough to promise a profit.
    with pytest.raises(Unpriced) as raised:
        value_guide(guide("Only rare", [item("Rare thing", 1, 9999)]), market())
    assert raised.value.reason == "no_volume"


def test_stale_last_trades_are_not_used_as_prices() -> None:
    with pytest.raises(Unpriced) as raised:  # last traded 4,000 s ago, limit is 3,600 s
        value_guide(guide("Dead", [item("Dead stock", 1, 9998)]), market())
    assert raised.value.reason == "no_price"
    ok = guide("Dead", [item("Dead stock", 0.001, 9998), yew()])
    with pytest.raises(Unpriced):  # a stale item cannot be skipped: that would misstate profit
        value_guide(ok, market())
    assert value_guide(ok, market(max_price_age_s=5_000)).profit_per_hour > 0


def test_a_formula_with_an_unknown_item_makes_the_guide_unpriceable() -> None:
    bad = item("Mystery", 1, None, value_expr='gep("Nonexistent thing", 1)')
    with pytest.raises(Unpriced) as raised:
        value_guide(guide("Mystery", [bad]), market())
    assert raised.value.reason in {"formula", "unknown_item"}


def test_recurring_guides_are_valued_per_hour_of_active_time_like_the_wiki_does() -> None:
    # 60 Yew logs per hour of active time -> 60 * 392 = 23,520 per active hour.
    farm = guide("Farm", [yew(60)], kind="recurring", recurrence="90 minutes", activity_minutes=6)
    assert value_guide(farm, market()).profit_per_hour == pytest.approx(23_520)


# =========================================================================== eligibility
def with_skills(**levels: int) -> PlayerState:
    return player(skills=dict.fromkeys(SKILL_NAMES, 70) | levels)


def needs(**skills: int) -> MoneyMakingGuide:
    return guide("Req", [yew()], skill_requirements={k.capitalize(): v for k, v in skills.items()})


def test_meeting_every_requirement_is_eligible() -> None:
    e = check_requirements(needs(slayer=69, crafting=70), with_skills())
    assert not e.gaps and e.status(MoneyConfig()) == "eligible"


@pytest.mark.parametrize(
    ("have", "expected"),
    [(66, "almost"), (65, "almost"), (64, "ineligible")],  # need 70: a gap of 5 is "a few levels"
)
def test_a_few_levels_short_is_almost_eligible(have: int, expected: str) -> None:
    e = check_requirements(needs(slayer=70), with_skills(slayer=have))
    assert e.gaps[0].have == have and e.gaps[0].need == 70
    assert e.status(MoneyConfig()) == expected


def test_at_most_two_skills_may_be_short_for_almost() -> None:
    cfg = MoneyConfig()
    two = check_requirements(needs(slayer=72, crafting=72), with_skills())
    three = check_requirements(needs(slayer=72, crafting=72, hunter=72), with_skills())
    assert two.status(cfg) == "almost" and three.status(cfg) == "ineligible"


def test_quests_must_be_finished_and_one_missing_quest_is_almost() -> None:
    g = guide("Quests", [yew()], quest_requirements=["Ghosts Ahoy", "Dream Mentor"])
    done = player(quests={"Ghosts Ahoy": "FINISHED", "Dream Mentor": "FINISHED"})
    one = player(quests={"Ghosts Ahoy": "FINISHED", "Dream Mentor": "IN_PROGRESS"})
    none = player(quests={"Ghosts Ahoy": "NOT_STARTED", "Dream Mentor": "NOT_STARTED"})
    cfg = MoneyConfig()
    assert check_requirements(g, done).status(cfg) == "eligible"
    e = check_requirements(g, one)
    assert e.quests_missing == ["Dream Mentor"] and e.status(cfg) == "almost"
    assert check_requirements(g, none).status(cfg) == "ineligible"


def test_what_the_snapshot_does_not_report_is_a_caveat_not_a_failure() -> None:
    g = guide(
        "Quests", [yew()], quest_requirements=["Ghosts Ahoy"], skill_requirements={"Slayer": 50}
    )
    e = check_requirements(g, PlayerState(account_type="main", members=True))  # type: ignore[arg-type]
    assert e.status(MoneyConfig()) == "eligible"
    assert e.caveats == ["Slayer level not reported", "status of 'Ghosts Ahoy' not reported"]


def test_quest_points_combat_and_total_level_are_checked_too() -> None:
    cfg = MoneyConfig()
    qp = guide("QP", [yew()], quest_points=200)
    assert check_requirements(qp, player(quest_points=150)).status(cfg) == "ineligible"
    assert check_requirements(qp, player(quest_points=200)).status(cfg) == "eligible"
    assert check_requirements(qp, player()).caveats == ["Quest points not available"]
    combat = guide("Combat", [yew()], combat_level=100)
    low = player(skills=dict.fromkeys(SKILL_NAMES, 70))  # combat level 89
    assert low.combat_level == 89
    e = check_requirements(combat, low)
    assert (e.gaps[0].have, e.gaps[0].need) == (89, 100) and e.status(cfg) == "ineligible"
    near = player(skills=dict.fromkeys(SKILL_NAMES, 75))  # combat level 95
    assert near.combat_level == 95
    assert check_requirements(combat, near).status(cfg) == "almost"
    assert check_requirements(guide("T", [yew()], total_level=2_000), player()).status(cfg) == (
        "ineligible"
    )  # 24 skills at 70 = 1,680


# =========================================================================== ranking
def catalogue() -> list[MoneyMakingGuide]:
    return [
        guide("Yew cutting", [yew()], category="Skilling", members_only=True),  # 117,600
        guide(
            "Planks", [item("Plank", 100, 960)], [item("Logs", 100, 1511)], category="Processing"
        ),  # 10,700
        guide(
            "Looting", [item("Coins", 600_000, 995, fixed_value=1)], category="Combat/High"
        ),  # 600,000
    ]


def test_guides_are_ranked_by_live_profit_per_hour() -> None:
    result = rank_guides(catalogue(), player(), market())
    assert [(r.title, r.profit_per_hour) for r in result.ranked] == [
        ("Looting", 600_000),
        ("Yew cutting", 117_600),
        ("Planks", 10_700),
    ]
    top = result.ranked[0]
    assert top.url == "https://oldschool.runescape.wiki/w/Money_making_guide/Looting"
    assert result.eligible_total == 3 and result.excluded == {}


def farm(**kw) -> MoneyMakingGuide:
    base = {"kind": "recurring", "category": "Recurring", "recurrence": "90 minutes"}
    return guide("Farm", [yew(60)], activity_minutes=6, **(base | kw))


def test_recurring_activities_are_listed_apart_and_valued_per_run() -> None:
    result = rank_guides([*catalogue(), farm()], player(), market())
    assert "Farm" not in [r.title for r in result.ranked]  # not comparable with profit per hour
    (row,) = result.recurring
    # 60/h of active time over a 6-minute run: 23,520 * 6/60 = 2,352 per run
    assert row.profit_per_run == 2_352 and row.active_minutes == 6
    assert row.repeats_every == "90 minutes" and row.profit_per_hour is None
    assert row.main_outputs[0].qty == pytest.approx(6.0)  # 60/h * 0.1 h per run
    assert row.main_outputs[0].value == pytest.approx(6 * 392)


def test_the_wikis_reference_for_a_recurring_guide_is_per_run_not_per_hour() -> None:
    (row,) = rank_guides([farm(source_hourly_profit=2_400)], player(), market()).recurring
    assert row.reference_profit_per_run == 2_400 and row.reference_profit_per_hour is None


def test_recurring_guides_can_be_filtered_by_category_or_left_out() -> None:
    guides = [*catalogue(), farm()]
    assert [
        r.title for r in rank_guides(guides, player(), market(), category="recurring").recurring
    ] == ["Farm"]
    assert rank_guides(guides, player(), market(), category="recurring").ranked == []
    assert rank_guides(guides, player(), market(), include_recurring=False).recurring == []


def test_recurring_activities_are_ranked_by_profit_per_run() -> None:
    big = guide(
        "Big farm",
        [yew(600)],
        kind="recurring",
        activity_minutes=6,
        category="Recurring",
        recurrence="1 day",
    )
    assert [r.title for r in rank_guides([farm(), big], player(), market()).recurring] == [
        "Big farm",
        "Farm",
    ]


def test_top_n_and_category_filters() -> None:
    assert len(rank_guides(catalogue(), player(), market(), top_n=2).ranked) == 2
    combat = rank_guides(catalogue(), player(), market(), category="combat")  # prefix, any case
    assert [r.title for r in combat.ranked] == ["Looting"]


def test_a_scaled_guide_says_how_much_of_its_pace_the_market_allows() -> None:
    (row,) = rank_guides([guide("Too much", [yew(6_000)])], player(), market()).ranked
    assert row.pace == pytest.approx(0.208, abs=1e-3) and row.profit_per_hour == 490_000
    (fine,) = rank_guides([guide("Fine", [yew(300)])], player(), market()).ranked
    assert fine.pace is None


def test_exclusions_are_counted_by_reason() -> None:
    obsolete = guide("Old", [yew()], obsolete=True)
    unsupported = guide("Lua", [yew()], unsupported=["computed by the wiki"])
    members = guide("Members", [yew()], members_only=True)
    hard = needs(slayer=99)
    unpriced = guide("Dead", [item("Dead stock", 1, 9998)])
    result = rank_guides([obsolete, unsupported, members, hard, unpriced], player(False), market())
    assert result.excluded == {
        "obsolete_on_wiki": 1,
        "not_computable_from_the_wiki": 1,
        "members_only": 1,
        "requirements_not_met": 1,
        "unpriced:no_price": 1,
    }
    assert result.ranked == [] and result.unpriced[0].startswith("Dead: ")


def test_unknown_membership_keeps_members_guides_and_a_free_player_gets_a_caveat() -> None:
    members = guide("Members", [yew()], members_only=True)
    unsure = guide("Unsure", [yew()])  # the wiki does not say
    result = rank_guides([members, unsure], player(None), market())
    assert {r.title for r in result.ranked} == {"Members", "Unsure"}
    free = rank_guides([members, unsure], player(False), market())
    assert [r.title for r in free.ranked] == ["Unsure"]
    assert "membership" in free.ranked[0].caveats[0]


def test_almost_eligible_guides_are_reported_apart_with_what_is_missing() -> None:
    near = needs(slayer=73)  # the player has 70
    result = rank_guides([near, *catalogue()], player(), market())
    assert [r.title for r in result.almost] == ["Req"]
    missing = result.almost[0].missing
    assert missing is not None and missing.skills == [{"skill": "Slayer", "have": 70, "need": 73}]
    assert "Req" not in [r.title for r in result.ranked]


def test_guides_needing_more_gp_than_the_player_has_are_listed_separately() -> None:
    rich = guide("Rich", [yew()], capital=50_000_000)
    result = rank_guides([rich, *catalogue()], player(gp=1_000_000), market())
    assert [r.title for r in result.needs_gp] == ["Rich"]
    assert "Rich" not in [r.title for r in result.ranked]
    unknown = rank_guides([rich], player(gp=None), market())  # no GP reported: not checked
    assert [r.title for r in unknown.ranked] == ["Rich"]


def test_the_wikis_reference_profit_is_context_not_the_ranking_key() -> None:
    stale = guide("Yew", [yew()], source_hourly_profit=9_999_999)
    row = rank_guides([stale], player(), market()).ranked[0]
    assert row.profit_per_hour == 117_600 and row.reference_profit_per_hour == 9_999_999


def test_ironmen_get_only_the_guides_written_for_them_and_no_profit_figures() -> None:
    cfg = MoneyConfig()
    guides = [
        guide("Yew cutting", [yew()]),  # a Grand Exchange guide: counted, not listed
        guide(
            "Green dragons (Ironman)", [yew()], ironman_friendly=True, source_hourly_profit=1_000
        ),
        guide("Elemental staves (Ironman)", [yew()], ironman_friendly=True, capital=5_000_000),
        guide("Hard (Ironman)", [yew()], ironman_friendly=True, skill_requirements={"Slayer": 99}),
        needs(slayer=99),
    ]
    rows, total, other = ironman_list(guides, player(account_type="ironman"), cfg)
    assert (total, other) == (2, 1)
    assert [r.title for r in rows] == ["Elemental staves (Ironman)", "Green dragons (Ironman)"]
    for row in rows:
        assert row.profit_per_hour is None and row.reference_profit_per_hour is None
        assert row.capital_recommended is None


# =========================================================================== the tool
class FakePrices:
    async def mapping(self) -> Snapshot[ItemMeta]:
        return Snapshot({i.id: i for i in ITEMS}, FETCHED)

    async def latest(self) -> Snapshot[LatestPrice]:
        return Snapshot(LATEST, FETCHED)

    async def one_hour(self) -> Snapshot[AveragePrice]:
        return Snapshot(HOURLY, FETCHED, api_timestamp=NOW_TS - 1_800)


def tool_ctx(guides: list[MoneyMakingGuide] | None) -> ToolContext:
    settings = Settings(_env_file=None, prices_user_agent="osrs-agent-tests - tests@example.com")
    store = GuideStore(guides) if guides is not None else None
    return ToolContext(settings, FakePrices(), now=lambda: NOW, guides=store)  # type: ignore[arg-type]


async def test_the_tool_returns_a_ranking_with_provenance() -> None:
    out = await run_tool(tool_ctx(catalogue()), "rank_moneymakers", {"top_n": 2}, player=player())
    assert out["status"] == "ok" and out["eligible_total"] == 3
    assert out["recurring"] == [] and out["almost_eligible"] == []
    assert [r["title"] for r in out["ranking"]] == ["Looting", "Yew cutting"]
    # default basis is mid: (400 + 380) / 2 = 390, tax floor(7.8) = 7 -> 383 * 300
    assert out["ranking"][1]["profit_per_hour"] == 114_900
    assert out["prices_fetched_at"] == "2026-10-14T11:59:30Z" and out["stale"] is False
    assert "midpoint" in out["price_basis"]
    assert any("reference_profit_per_hour" in note for note in out["notes"])
    assert out["account"] == {
        "type": "main",
        "members": True,
        "members_known": True,
        "gp": 5_000_000,
    }


async def test_the_tool_asks_for_the_snapshot_instead_of_guessing_levels() -> None:
    out = await run_tool(tool_ctx(catalogue()), "rank_moneymakers", {})
    assert out["error"] == "player_state_missing"


async def test_the_tool_never_gives_an_ironman_a_profit_figure() -> None:
    guides = [*catalogue(), guide("Dragons (Ironman)", [yew()], ironman_friendly=True)]
    out = await run_tool(
        tool_ctx(guides),
        "rank_moneymakers",
        {},
        player=player(account_type="hardcore_ironman"),
    )
    assert out["status"] == "not_applicable_ironman" and out["eligible_total"] == 1
    assert out["other_eligible_guides_assuming_ge"] == 3 and "3 guides" in out["message"]
    assert "ranking" not in out and "advice_pages" in out
    assert [g["title"] for g in out["eligible_guides"]] == ["Dragons (Ironman)"]
    assert "profit" not in str(out["eligible_guides"]) and "reference" not in str(out)


async def test_the_tool_reports_missing_data_instead_of_inventing_a_ranking() -> None:
    out = await run_tool(tool_ctx(None), "rank_moneymakers", {}, player=player())
    assert out["error"] == "guides_unavailable"


async def test_the_model_cannot_pass_the_player_itself() -> None:
    out = await run_tool(
        tool_ctx(catalogue()), "rank_moneymakers", {"player": {"skills": {}}}, player=player()
    )
    assert out["error"] == "invalid_arguments"


# =========================================================================== what is verified
def test_the_requirements_a_player_meets_are_listed_with_what_they_have() -> None:
    g = guide(
        "Req", [yew()], skill_requirements={"Smithing": 30}, quest_requirements=["The Giant Dwarf"]
    )
    state = player(
        skills=dict.fromkeys(SKILL_NAMES, 70) | {"smithing": 45},
        quests={"The Giant Dwarf": "FINISHED"},
    )
    eligibility = check_requirements(g, state)
    assert eligibility.met == ["Smithing 30 (you have 45)", "The Giant Dwarf (quest finished)"]
    assert eligibility.gaps == [] and eligibility.quests_missing == []


def test_a_quest_the_wiki_only_asks_to_have_started_counts_when_it_is_in_progress() -> None:
    g = guide(
        "Req",
        [yew()],
        quest_requirements=["The Giant Dwarf"],
        requirements_text=["Quest: Started The Giant Dwarf"],
    )
    started = player(quests={"The Giant Dwarf": "IN_PROGRESS"})
    not_started = player(quests={"The Giant Dwarf": "NOT_STARTED"})
    assert check_requirements(g, started).quests_missing == []
    assert check_requirements(g, not_started).quests_missing == ["The Giant Dwarf"]
    finished_only = guide("Req2", [yew()], quest_requirements=["The Giant Dwarf"])
    assert check_requirements(finished_only, started).quests_missing == ["The Giant Dwarf"]


def test_rows_carry_what_the_player_meets_and_the_wikis_own_requirement_lines() -> None:
    g = guide(
        "Boss",
        [item("Coins", 100_000, 995, fixed_value=1)],
        category="Combat/High",
        skill_requirements={"Prayer": 43},
        requirements_text=["Skill: 43+ Prayer", "Item: Torva platebody"],
    )
    result = rank_guides([g], player(skills=dict.fromkeys(SKILL_NAMES, 70)), market())
    (row,) = result.ranked
    assert row.you_meet == ["Prayer 43 (you have 70)"]
    assert row.wiki_requirements == ["Skill: 43+ Prayer", "Item: Torva platebody"]
    assert row.requirements is None


def test_close_to_says_in_plain_words_what_each_almost_guide_still_needs() -> None:
    from app.tools.moneymakers import close_to_lines

    g = guide(
        "Boss",
        [item("Coins", 100_000, 995, fixed_value=1)],
        skill_requirements={"Defence": 75},
        quest_requirements=["Perilous Moons"],
    )
    state = player(skills=dict.fromkeys(SKILL_NAMES, 70) | {"defence": 74}, quests={})
    state = player(
        skills=dict.fromkeys(SKILL_NAMES, 70) | {"defence": 74},
        quests={"Perilous Moons": "NOT_STARTED"},
    )
    result = rank_guides([g], state, market())
    assert close_to_lines(result.almost) == [
        "Boss: ~100,000 GP/h once you have Defence 75 (you have 74) and the quest Perilous Moons"
    ]
