"""The tool layer end to end (registry -> wrappers -> pure maths) over a fake prices client."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.clients.prices import (
    AveragePrice,
    ItemMeta,
    LatestPrice,
    Snapshot,
    Timeseries,
    TimeseriesPoint,
)
from app.config import Settings
from app.errors import UpstreamError
from app.tools.context import ToolContext
from app.tools.registry import TOOLS, run_tool, tool_specs

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
NOW_TS = int(NOW.timestamp())
FETCHED = datetime(2026, 10, 14, 11, 59, 30, tzinfo=UTC)

ITEMS = [
    ItemMeta(
        id=20997,
        name="Twisted bow",
        members=True,
        limit=8,
        highalch=2_400_000,
        lowalch=1_600_000,
        value=4_000_000,
    ),
    ItemMeta(
        id=4151,
        name="Abyssal whip",
        members=True,
        limit=70,
        highalch=72_000,
        lowalch=48_000,
        value=120_001,
    ),
    ItemMeta(id=13190, name="Old school bond", limit=100, value=2_000_000),
    ItemMeta(id=561, name="Nature rune", limit=18_000, highalch=None),
    ItemMeta(id=999, name="Never traded", limit=10),
    ItemMeta(id=1333, name="Rune scimitar", members=False, limit=70),
    ItemMeta(id=1359, name="Rune axe", members=False, limit=70),
]


def q(low: int | None, high: int | None, age: int = 90) -> LatestPrice:
    return LatestPrice(
        high=high,
        high_time=NOW_TS - age if high else None,
        low=low,
        low_time=NOW_TS - age if low else None,
    )


def hv(low_volume: int, high_volume: int) -> AveragePrice:
    return AveragePrice(
        avg_high=1_000.5, high_volume=high_volume, avg_low=990.25, low_volume=low_volume
    )


class FakePrices:
    """Duck-typed stand-in for PricesClient (the real one is covered in test_prices_client)."""

    def __init__(self) -> None:
        self.latest_data = {
            4151: q(775_464, 791_060),
            20997: q(1_000_000_000, 1_100_000_000),
            13190: q(11_668_000, 11_700_000),
            561: q(100, 110),
            999: LatestPrice(),
            1333: q(15_000, 15_500),
        }
        self.hourly_data = {
            4151: hv(100, 80),
            13190: hv(300, 300),
            561: hv(9_000, 8_000),
            1333: hv(400, 400),
        }
        self.fail = False
        self.flat_history = False  # True: 7d/30d series built from the current quote and volumes
        self.requested_timeseries: list[tuple[int, str]] = []

    async def mapping(self) -> Snapshot[ItemMeta]:
        return Snapshot({i.id: i for i in ITEMS}, FETCHED)

    async def latest(self) -> Snapshot[LatestPrice]:
        self._maybe_fail()
        return Snapshot(self.latest_data, FETCHED)

    async def latest_item(self, item_id: int):
        self._maybe_fail()
        return self.latest_data.get(item_id), FETCHED, False

    async def one_hour(self) -> Snapshot[AveragePrice]:
        self._maybe_fail()
        return Snapshot(self.hourly_data, FETCHED, api_timestamp=NOW_TS - 1_800)

    async def timeseries(self, item_id: int, lookback: str) -> Timeseries:
        self._maybe_fail()
        self.requested_timeseries.append((item_id, lookback))
        quote, volume = self.latest_data.get(item_id), self.hourly_data.get(item_id)
        if self.flat_history and lookback in ("7d", "30d") and quote and volume:
            assert quote.high and quote.low
            # a flat, well-traded week: the current quote and hourly volumes, every hour
            n, step = (168, 3_600) if lookback == "7d" else (120, 21_600)
            points = [
                TimeseriesPoint(
                    timestamp=NOW_TS - (n - 1 - i) * step,
                    avg_high=float(quote.high),
                    avg_low=float(quote.low),
                    high_volume=volume.high_volume,
                    low_volume=volume.low_volume,
                )
                for i in range(n)
            ]
            return Timeseries(
                itemId=item_id, lookback=lookback, timestep=step, points=points, fetched_at=FETCHED
            )
        points = [
            TimeseriesPoint(
                timestamp=NOW_TS - (5 - i) * 3_600,
                avg_high=100 + i,
                avg_low=90 + i,
                high_volume=2,
                low_volume=1,
            )
            for i in range(6)
        ]
        return Timeseries(
            itemId=item_id, lookback=lookback, timestep=3_600, points=points, fetched_at=FETCHED
        )

    def _maybe_fail(self) -> None:
        if self.fail:
            raise UpstreamError("prices API returned HTTP 503")


@pytest.fixture
def prices() -> FakePrices:
    return FakePrices()


@pytest.fixture
def ctx(prices: FakePrices) -> ToolContext:
    settings = Settings(_env_file=None, prices_user_agent="osrs-agent-tests - tests@example.com")
    return ToolContext(settings, prices, now=lambda: NOW)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- schemas
def test_specs_cover_the_spec_tools_in_openai_function_format() -> None:
    specs = tool_specs()
    assert [s["function"]["name"] for s in specs] == [
        "get_item_price",
        "get_price_history",
        "calc_flip",
        "analyze_flip",
        "rank_flips",
        "rank_moneymakers",
    ]
    for spec in specs:
        assert spec["type"] == "function"
        params = spec["function"]["parameters"]
        assert params["type"] == "object" and params["additionalProperties"] is False
        assert "title" not in params and all(
            "title" not in p for p in params["properties"].values()
        )
    by_name = {s["function"]["name"]: s["function"]["parameters"] for s in specs}
    # The player's snapshot is injected by the runtime: the model gets no argument for it.
    assert set(by_name["rank_moneymakers"]["properties"]) == {
        "top_n",
        "category",
        "include_recurring",
        "stated_combat_level",
    }
    assert "required" not in by_name["rank_moneymakers"]
    assert by_name["rank_flips"]["required"] == ["max_gp"]
    assert by_name["get_item_price"]["required"] == ["item"]
    assert by_name["get_price_history"]["properties"]["lookback"]["enum"] == [
        "6h",
        "24h",
        "7d",
        "30d",
        "6m",
        "1y",
    ]
    assert [s["function"]["name"] for s in tool_specs(["calc_flip"])] == ["calc_flip"]
    # the model is told what high/low mean, in the words the tool output uses
    assert "instant_buy_price" in TOOLS["get_item_price"].description


# --------------------------------------------------------------------------- dispatch errors
async def test_unknown_tool_and_invalid_arguments_come_back_as_data(ctx) -> None:
    unknown = await run_tool(ctx, "make_money", {})
    assert unknown["error"] == "unknown_tool" and "rank_flips" in unknown["available"]

    for tool, args in [
        ("rank_flips", {}),  # missing max_gp
        ("rank_flips", {"max_gp": 0}),
        ("rank_flips", {"max_gp": 1_000, "top_n": 500}),
        ("rank_flips", {"max_gp": 1_000, "strategy": "yolo"}),
        ("calc_flip", {"item": "Abyssal whip", "qty": 0}),
        ("calc_flip", {"item": "Abyssal whip", "extra": 1}),  # hallucinated argument
        ("get_price_history", {"item": "Abyssal whip", "lookback": "1w"}),
    ]:
        result = await run_tool(ctx, tool, args)
        assert result["error"] == "invalid_arguments" and result["details"], (tool, args)


async def test_price_api_outage_is_reported_without_inventing_numbers(ctx, prices) -> None:
    prices.fail = True
    for tool, args in [
        ("get_item_price", {"item": "Abyssal whip"}),
        ("calc_flip", {"item": "Abyssal whip"}),
        ("rank_flips", {"max_gp": 1_000_000}),
        ("analyze_flip", {"item": "Abyssal whip"}),
        ("get_price_history", {"item": "Abyssal whip"}),
    ]:
        result = await run_tool(ctx, tool, args)
        assert result["error"] == "price_data_unavailable" and "do not guess" in result["message"]


# --------------------------------------------------------------------------- get_item_price
async def test_get_item_price_resolves_nicknames_and_reports_timestamps(ctx) -> None:
    out = await run_tool(ctx, "get_item_price", {"item": "tbow"})
    assert out["item"] == "Twisted bow" and out["item_id"] == 20997
    assert out["match"]["status"] == "exact" and "nickname" in out["match"]["note"]
    assert out["instant_buy_price"] == {
        "price": 1_100_000_000,  # above int32, untouched
        "as_of": "2026-10-14T11:58:30Z",
        "age_seconds": 90,
    }
    assert out["instant_sell_price"]["price"] == 1_000_000_000
    assert out["spread"] == 100_000_000 and out["buy_limit_per_4h"] == 8
    assert out["high_alchemy_value"] == 2_400_000
    assert out["prices_fetched_at"] == "2026-10-14T11:59:30Z" and out["stale"] is False


async def test_get_item_price_includes_last_hour_volume(ctx) -> None:
    out = await run_tool(ctx, "get_item_price", {"item": "Abyssal whip"})
    assert out["last_hour"] == {
        "window_start": "2026-10-14T11:30:00Z",
        "avg_instant_buy_price": 1_000.5,
        "avg_instant_sell_price": 990.25,
        "units_bought_instantly": 80,
        "units_sold_instantly": 100,
    }
    missing = await run_tool(ctx, "get_item_price", {"item": "Twisted bow"})
    assert missing["last_hour"] is None  # no /1h row: reported, not zero-filled


async def test_get_item_price_ambiguous_name_returns_candidates(ctx) -> None:
    out = await run_tool(ctx, "get_item_price", {"item": "rune"})
    assert out["error"] == "item_ambiguous"
    assert {c["name"] for c in out["candidates"]} >= {"Rune scimitar", "Rune axe"}


async def test_get_item_price_edge_cases(ctx) -> None:
    coins = await run_tool(ctx, "get_item_price", {"item": "coins"})
    assert coins["face_value_gp"] == 1 and "Not tradeable" in coins["note"]
    never = await run_tool(ctx, "get_item_price", {"item": "Never traded"})
    assert never["no_price_data"] is True
    nothing = await run_tool(ctx, "get_item_price", {"item": "qwertyuiop"})
    assert nothing["error"] == "item_not_found"


# --------------------------------------------------------------------------- get_price_history
async def test_get_price_history_summarises_and_passes_the_lookback(ctx, prices) -> None:
    out = await run_tool(ctx, "get_price_history", {"item": "whip", "lookback": "30d"})
    assert prices.requested_timeseries == [(4151, "30d")]
    assert out["item"] == "Abyssal whip" and out["lookback"] == "30d"
    assert out["trend"]["direction"] == "up" and out["n_points"] == 6
    default = await run_tool(ctx, "get_price_history", {"item": "Abyssal whip"})
    assert default["lookback"] == "7d"
    coins = await run_tool(ctx, "get_price_history", {"item": "coins"})
    assert coins["error"] == "not_tradeable"


# --------------------------------------------------------------------------- calc_flip
async def test_calc_flip_can_use_a_sell_price_the_player_states(ctx: ToolContext) -> None:
    # Selling an abyssal whip at 1,500,000: tax floor(1,500,000 * 2%) = 30,000, net 1,470,000;
    # the buy side stays the market's 775,464 -> margin 694,536.
    out = await run_tool(ctx, "calc_flip", {"item": "abyssal whip", "sell_price": 1_500_000})
    assert out["sell_price"] == 1_500_000 and out["buy_price"] == 775_464
    assert out["tax_per_unit"] == 30_000 and out["net_sell_price"] == 1_470_000
    assert out["margin_per_unit"] == 694_536
    assert out["price_source"] == {"buy": "market", "sell": "player"}
    assert any("player gave" in w for w in out["warnings"])


async def test_calc_flip_with_both_prices_does_not_need_a_market_quote(ctx: ToolContext) -> None:
    out = await run_tool(
        ctx, "calc_flip", {"item": "Never traded", "buy_price": 1_000, "sell_price": 1_200}
    )
    assert out["tax_per_unit"] == 24 and out["margin_per_unit"] == 176  # 1,200 - 24 - 1,000
    assert out["price_source"] == {"buy": "player", "sell": "player"}


async def test_calc_flip_without_overrides_reports_no_price_source(ctx: ToolContext) -> None:
    out = await run_tool(ctx, "calc_flip", {"item": "abyssal whip"})
    assert "price_source" not in out


async def test_calc_flip_end_to_end_matches_the_hand_calculation(ctx) -> None:
    out = await run_tool(ctx, "calc_flip", {"item": "Abyssal whip", "qty": 100})
    # buy 775,464 / sell 791,060: tax floor(2%) = 15,821 -> net 775,239 -> margin -225 (a loss!)
    assert out["tax_per_unit"] == 15_821 and out["net_sell_price"] == 775_239
    assert out["margin_per_unit"] == -225 and out["roi_pct"] == -0.03
    assert out["qty_this_window"] == 70 and out["windows_needed"] == 2
    assert out["profit_this_window"] == -225 * 70
    assert any("No profit after tax" in w for w in out["warnings"])
    assert out["prices_fetched_at"] == "2026-10-14T11:59:30Z" and out["stale"] is False
    assert "volume" in out and out["volume"]["bottleneck_units_per_hour"] == 80


async def test_calc_flip_applies_the_default_exempt_list(ctx) -> None:
    out = await run_tool(ctx, "calc_flip", {"item": "bond"})
    assert (
        out["tax_exempt"] is True and out["tax_per_unit"] == 0 and out["margin_per_unit"] == 32_000
    )


async def test_calc_flip_error_payloads(ctx) -> None:
    assert (await run_tool(ctx, "calc_flip", {"item": "Never traded"}))["error"] == "one_sided"
    assert (await run_tool(ctx, "calc_flip", {"item": "rune"}))["error"] == "item_ambiguous"
    assert (await run_tool(ctx, "calc_flip", {"item": "platinum token"}))[
        "error"
    ] == "not_tradeable"


# --------------------------------------------------------------------------- rank_flips
async def test_rank_flips_screens_the_market_then_judges_each_candidate_by_its_history(
    ctx, prices
) -> None:
    prices.flat_history = True
    out: dict[str, Any] = await run_tool(ctx, "rank_flips", {"max_gp": 50_000_000, "top_n": 5})
    assert out["strategy"] == "volume" and out["offer_slots"] == 8
    # The volume strategy asks for 200 units/hour on the slower side and a margin of at least 1%
    # after tax: the whip (80/hour), the 1B bow (beyond the budget) and the tax-exempt bond (a 0.27%
    # margin) are out before any history is fetched; nature rune and scimitar are analysed.
    assert sorted(i for i, lb in prices.requested_timeseries if lb == "7d") == [561, 1333]
    assert out["items_with_history_analysed"] == 2
    rows = {r["item"]: r for r in out["rows"]}
    assert set(rows) == {"Nature rune", "Rune scimitar"}
    # nature rune: 110 sold -> tax 2 -> 108 - 100 = 8 per unit; 8,000/h x 4 h x 25% = 8,000 units
    rune = rows["Nature rune"]
    assert (rune["buy_at"], rune["sell_at"]) == (100, 110)
    assert (rune["margin_typical_after_tax"], rune["units"]) == (8, 8_000)
    assert rune["limiting_factor"] == "market_volume"
    assert rune["profit_per_window_typical"] == 64_000
    # scimitar: 15,500 sold -> tax 310 -> 15,190 - 15,000 = 190; 400/h x 4 h x 25% = 400, limit 70
    scimitar = rows["Rune scimitar"]
    assert (scimitar["margin_typical_after_tax"], scimitar["units"]) == (190, 70)
    assert scimitar["limiting_factor"] == "buy_limit"
    assert scimitar["profit_per_window_conservative"] == 13_300
    # every row carries facts to quote and a worked example that adds up
    for row in out["rows"]:
        assert any("positive in 100%" in line for line in row["evidence"])
        example = row["worked_example"]
        assert example["capital_gp"] == example["units"] * example["buy_at"]
    assert out["rows"][0]["item"] == "Nature rune"  # 8,000 x 8 gp beats 70 x 190 gp
    assert out["excluded"]["low_volume"] == 1 and out["excluded"]["unaffordable"] == 1
    assert out["excluded"]["margin_too_thin"] == 1
    assert out["tax_rate_pct"] == 2.0 and out["prices_fetched_at"] == "2026-10-14T11:59:30Z"
    assert any("manipulation" in note for note in out["notes"])


async def test_rank_flips_free_to_play_gets_three_offer_slots_and_no_members_items(
    ctx, prices
) -> None:
    prices.flat_history = True
    out = await run_tool(ctx, "rank_flips", {"max_gp": 50_000_000, "members": False})
    assert out["offer_slots"] == 3
    assert {r["item"] for r in out["rows"]} <= {"Nature rune", "Rune scimitar"}
    assert out["excluded"]["members_only"] >= 1


async def test_rank_flips_says_so_instead_of_filling_the_list_with_weak_items(ctx, prices) -> None:
    prices.hourly_data = {i: hv(5, 5) for i in prices.hourly_data}  # nothing trades much
    out = await run_tool(ctx, "rank_flips", {"max_gp": 50_000_000})
    assert out["rows"] == [] and out["excluded"]["low_volume"] >= 3


async def test_rank_flips_with_no_history_at_all_is_a_data_error(ctx, prices) -> None:
    async def broken(item_id: int, lookback: str):
        raise UpstreamError("prices API returned HTTP 503")

    prices.timeseries = broken  # latest and hourly work, the history does not
    out = await run_tool(ctx, "rank_flips", {"max_gp": 50_000_000})
    assert out["error"] == "price_data_unavailable" and "do not give" in out["message"]


async def test_analyze_flip_judges_one_item_and_shows_its_evidence(ctx, prices) -> None:
    prices.flat_history = True
    out = await run_tool(ctx, "analyze_flip", {"item": "nature rune", "budget": 1_000_000})
    assert out["item"] == "Nature rune" and out["verdict"] in ("solid", "reasonable")
    # 1,000,000 / 100 gp = 10,000 units, but the market share allows 8,000
    assert (out["units"], out["limiting_factor"]) == (8_000, "market_volume")
    assert out["metrics"]["volume_typical_per_hour"] == 8_000
    assert out["metrics"]["buy_sell_ratio"] == 0.89  # 8,000 instant buys vs 9,000 instant sells
    assert out["history"]["readings"] == 168 and out["tax_rate_pct"] == 2.0
    assert out["evidence"] and out["worked_example"]["units"] == 8_000


async def test_analyze_flip_error_payloads(ctx, prices) -> None:
    prices.flat_history = True
    never = await run_tool(ctx, "analyze_flip", {"item": "Never traded"})
    assert never["error"] == "no_two_sided_quote"
    assert (await run_tool(ctx, "analyze_flip", {"item": "rune"}))["error"] == "item_ambiguous"
    plat = await run_tool(ctx, "analyze_flip", {"item": "platinum token"})
    assert plat["error"] == "not_tradeable"
