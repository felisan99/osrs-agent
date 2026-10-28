"""Money-making ranking (SPEC §8): which guides the player can do, and what each pays right now.

Guides are stored as structured data (quantities per hour plus price formulas), never as frozen
prices (``app/ingest/mmg.py``). Here a guide is (1) checked against what the player meets and
(2) valued with live prices:

    profit/h = sum(out qty/h * (sell price - GE tax)) - sum(in qty/h * buy price)

* Only outputs sold on the Grand Exchange pay tax. A guide's own ``value`` override is never taxed
  (that is how the wiki's ``Module:Mmgtable`` does it, and we reproduce the wiki's numbers).
* Prices come from ``/1h`` averages when that side of the market traded in the last hour, else
  from ``/latest`` if the last trade is fresh; otherwise the item cannot be priced and the guide is
  skipped (and counted), never valued with a guess. ``MM_PRICE_BASIS`` selects mid (default: the
  wiki's own convention), patient (sell at the high side, buy at the low side: the flipper's
  view, SPEC §6) or instant prices.
* A guide states how fast it can be run (items per hour), not whether the market can absorb it.
  An item that dominates the guide's value is checked against the hourly volume it trades: the
  guide is scaled down (profit scales linearly with pace) so that no item exceeds
  ``MM_MARKET_SHARE`` of the market, and the result says so. An item with no trades in the last
  hour cannot be valued, so such a guide is skipped.
* Recurring activities (farming runs, daily purchases) are listed apart, valued PER RUN: the wiki
  quotes "profit per hour of active time" for a few minutes of work every so often, which would
  put them above every continuous method.
* Ironmen cannot use the Grand Exchange, so there is no GP/hour for them (user decision, SPEC §8
  step 3): the result lists the guides they qualify for, without any profit figure. Only guides
  written for ironmen ("(Ironman)" pages) are listed: every other guide is a Grand Exchange
  trade or a loot valuation, and is merely counted.
* "Almost eligible" guides (a few levels or one quest short) and guides that need more GP than
  the player has are reported apart from the ranking.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import psycopg
from pydantic import BaseModel, Field

from app.api.schemas import PlayerState
from app.clients.prices import AveragePrice, LatestPrice
from app.clients.wiki import page_url
from app.config import Settings
from app.expr import ExprError, evaluate
from app.ingest.mmg import GuideItem, MoneyMakingGuide
from app.tools.items import FACE_VALUE, ItemIndex
from app.tools.tax import TaxPolicy

if TYPE_CHECKING:
    from app.tools.context import ToolContext

Side = Literal["sell", "buy"]
Status = Literal["eligible", "almost", "ineligible"]
GapKind = Literal["skill", "combat", "quest_points", "total_level"]
ConnectionFactory = Callable[[], AbstractAsyncContextManager[psycopg.AsyncConnection]]

GUIDE_PREFIX = "Money making guide/"
IRONMAN_LIST_LIMIT = 15
ALMOST_LIST_LIMIT = 5
RECURRING_LIST_LIMIT = 5
NEEDS_GP_LIST_LIMIT = 3
WIKI_LINES = 10  # requirement lines of the guide's own text kept per row
WIKI_LINE_CHARS = 200
MAIN_LINES = 3  # items shown per side in a ranking row
IRONMAN_ADVICE_PAGES = ["Ironman money making guide", "Ironman guide"]


# --------------------------------------------------------------------------- store
class GuideStore:
    """The structured guides, loaded once from Postgres (637 small records)."""

    def __init__(self, guides: Iterable[MoneyMakingGuide]) -> None:
        self.guides = list(guides)

    def __len__(self) -> int:
        return len(self.guides)

    @classmethod
    async def load(cls, connect: ConnectionFactory) -> GuideStore:
        async with connect() as conn:
            cursor = await conn.execute("select data from moneymaking_guides order by page_title")
            rows = await cursor.fetchall()
        return cls(MoneyMakingGuide.model_validate(row[0]) for row in rows)


@dataclass(frozen=True)
class MoneyConfig:
    basis: Literal["patient", "mid", "instant"] = "mid"
    max_price_age_s: int = 3600
    liquidity_min_share: float = 0.05
    market_share: float = 0.25
    almost_max_level_gap: int = 5
    almost_max_skills: int = 2
    almost_max_quests: int = 1
    wiki_base: str = "https://oldschool.runescape.wiki/w/"

    @classmethod
    def from_settings(cls, settings: Settings) -> MoneyConfig:
        return cls(
            basis=settings.mm_price_basis,
            max_price_age_s=settings.mm_max_price_age_s,
            liquidity_min_share=settings.mm_liquidity_min_share,
            market_share=settings.mm_market_share,
            almost_max_level_gap=settings.mm_almost_max_level_gap,
            almost_max_skills=settings.mm_almost_max_skills,
            almost_max_quests=settings.mm_almost_max_quests,
            wiki_base=settings.wiki_page_base_url,
        )


# --------------------------------------------------------------------------- eligibility
@dataclass(frozen=True)
class Gap:
    kind: GapKind
    name: str
    have: int
    need: int


@dataclass
class Eligibility:
    gaps: list[Gap] = field(default_factory=list)
    quests_missing: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)  # requirements the snapshot cannot confirm
    met: list[str] = field(default_factory=list)  # requirements the snapshot confirms are met

    def status(self, cfg: MoneyConfig) -> Status:
        if not self.gaps and not self.quests_missing:
            return "eligible"
        # A few levels (skills, combat) or one quest short; point/total-level shortfalls are far.
        if any(g.kind in ("quest_points", "total_level") for g in self.gaps):
            return "ineligible"
        small = all(g.need - g.have <= cfg.almost_max_level_gap for g in self.gaps)
        if (
            small
            and len(self.gaps) <= cfg.almost_max_skills
            and len(self.quests_missing) <= cfg.almost_max_quests
        ):
            return "almost"
        return "ineligible"


def check_requirements(
    guide: MoneyMakingGuide, player: PlayerState, *, stated_combat: int | None = None
) -> Eligibility:
    """Compare a guide's hard requirements with the snapshot. Unknowns are caveats, not failures.

    ``stated_combat`` is a combat level the player *said* in the message when no snapshot could
    give one: it is used only where the snapshot has none.
    """
    result = Eligibility()
    for skill, need in sorted(guide.skill_requirements.items()):
        have = player.level(skill)
        if have is None:
            result.caveats.append(f"{skill} level not reported")
        elif have < need:
            result.gaps.append(Gap("skill", skill, have, need))
        else:
            result.met.append(f"{skill} {need} (you have {have})")
    for quest in guide.quest_requirements:
        status = player.quest_status(quest)
        # "Quest: Started The Giant Dwarf": the wiki only asks for the quest to be started
        started_is_enough = any(
            f"started {quest}".casefold() in line.casefold() for line in guide.requirements_text
        )
        if status is None:
            result.caveats.append(f"status of '{quest}' not reported")
        elif status.value == "FINISHED":
            result.met.append(f"{quest} (quest finished)")
        elif started_is_enough and status.value == "IN_PROGRESS":
            result.met.append(f"{quest} (quest started, which is all the guide asks)")
        else:
            result.quests_missing.append(quest)
    numeric: list[tuple[GapKind, str, int | None, int | None]] = [
        ("quest_points", "Quest points", player.quest_points, guide.quest_points),
        ("combat", "Combat level", player.combat_level or stated_combat, guide.combat_level),
        ("total_level", "Total level", player.total_level, guide.total_level),
    ]
    for kind, name, have, need in numeric:
        if need is None:
            continue
        if have is None:
            result.caveats.append(f"{name} not available")
        elif have < need:
            result.gaps.append(Gap(kind, name, have, need))
        else:
            result.met.append(f"{name} {need} (you have {have})")
    return result


# --------------------------------------------------------------------------- prices
class Unpriced(Exception):
    """A guide cannot be valued exactly right now; ``reason`` is a stable slug for the counts."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Quote:
    price: float
    volume: int  # units traded per hour on the side(s) the price comes from (/1h)
    source: Literal["1h average", "last trade"]
    age_s: int | None  # of the last trade, when the price comes from /latest


class Market:
    """Live prices for valuing guides: one instance per call, built from three snapshots."""

    def __init__(
        self,
        *,
        latest: Mapping[int, LatestPrice],
        hourly: Mapping[int, AveragePrice],
        index: ItemIndex,
        tax: TaxPolicy,
        now: datetime,
        cfg: MoneyConfig,
    ) -> None:
        self.latest = latest
        self.hourly = hourly
        self.index = index
        self.tax = tax
        self.now = now
        self.cfg = cfg

    def _side(self, item_id: int, which: Literal["high", "low"]) -> Quote | None:
        hourly = self.hourly.get(item_id)
        latest = self.latest.get(item_id)
        if hourly is not None:
            avg = hourly.avg_high if which == "high" else hourly.avg_low
            volume = hourly.high_volume if which == "high" else hourly.low_volume
            if avg and avg > 0 and volume > 0:
                return Quote(float(avg), volume, "1h average", None)
        if latest is not None:
            price = latest.high if which == "high" else latest.low
            stamp = latest.high_time if which == "high" else latest.low_time
            if price and price > 0 and stamp:
                age = max(int(self.now.timestamp() - stamp), 0)
                if age <= self.cfg.max_price_age_s:
                    return Quote(float(price), 0, "last trade", age)
        return None

    def quote(self, item_id: int, side: Side) -> Quote:
        high, low = self._side(item_id, "high"), self._side(item_id, "low")
        basis = self.cfg.basis
        if basis == "mid":
            chosen = [high, low]
        elif (basis == "patient") == (side == "sell"):
            chosen = [high]  # patient seller / instant buyer deal at the high side
        else:
            chosen = [low]
        if any(q is None for q in chosen):
            name = self.index.name_of(item_id)
            raise Unpriced("no_price", f"no recent price for {name}")
        quotes: list[Quote] = [q for q in chosen if q is not None]
        if len(quotes) == 1:
            return quotes[0]
        ages = [q.age_s for q in quotes if q.age_s is not None]
        return Quote(
            sum(q.price for q in quotes) / len(quotes),
            min(q.volume for q in quotes),
            "last trade" if ages else "1h average",
            max(ages) if ages else None,
        )

    def exact_id(self, name: str) -> int:
        resolution = self.index.resolve(name)
        if resolution.status != "exact" or resolution.item is None:
            raise Unpriced("unknown_item", f"'{name}' is not an exact item name")
        return resolution.item.id


# --------------------------------------------------------------------------- valuation
class Line(BaseModel):
    item: str
    per_hour: float
    unit_price: float
    value_per_hour: float = Field(description="after GE tax for sold items")
    price_source: str | None = None
    market_volume_per_hour: int | None = None
    note: str | None = None


@dataclass
class Valuation:
    revenue_per_hour: float = 0.0
    cost_per_hour: float = 0.0
    outputs: list[Line] = field(default_factory=list)
    inputs: list[Line] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pace: float = 1.0  # share of the guide's own pace the market allows (1.0 = all of it)

    @property
    def profit_per_hour(self) -> float:
        return self.revenue_per_hour - self.cost_per_hour


def _expr_functions(market: Market, side: Side) -> dict[str, Callable[..., float]]:
    def gep(name: Any, qty: Any = 1.0) -> float:
        if not isinstance(name, str):
            raise ExprError("gep() needs an item name")
        item_id = market.exact_id(name)
        unit = FACE_VALUE.get(item_id)
        if unit is None:
            unit = market.quote(item_id, side).price
        return float(unit) * float(qty)

    def ha(name: Any) -> float:
        if not isinstance(name, str):
            raise ExprError("ha() needs an item name")
        meta = market.index.by_id(market.exact_id(name))
        if meta is None or meta.highalch is None:
            raise ExprError(f"no high alchemy value for {name!r}")
        return float(meta.highalch)

    return {"gep": gep, "ha": ha}


@dataclass(frozen=True)
class _Priced:
    unit: float  # price per item before tax
    tax: float  # per item; 0 for overrides and purchases
    quote: Quote | None  # None for fixed values and formulas (no market to check)
    note: str | None = None


def _price_item(item: GuideItem, side: Side, market: Market) -> _Priced:
    if item.fixed_value is not None:
        return _Priced(item.fixed_value, 0.0, None)
    if item.value_expr:
        try:
            value = evaluate(item.value_expr, funcs=_expr_functions(market, side))
        except ExprError as exc:
            raise Unpriced("formula", f"{item.name}: {exc}") from exc
        return _Priced(value, 0.0, None, "valued with the guide's own price formula")
    if item.cheapest_of:
        options: list[tuple[str, Quote]] = []
        for name in item.cheapest_of:
            try:
                options.append((name, market.quote(market.exact_id(name), side)))
            except Unpriced:
                continue
        if not options:
            raise Unpriced("no_price", f"none of {item.cheapest_of} has a recent price")
        name, quote = min(options, key=lambda option: option[1].price)
        return _Priced(quote.price, 0.0, quote, f"cheapest of the listed options today: {name}")
    if item.item_id is None:
        raise Unpriced("unknown_item", f"'{item.name}' has no item id")
    if item.item_id in FACE_VALUE:
        return _Priced(float(FACE_VALUE[item.item_id]), 0.0, None)
    quote = market.quote(item.item_id, side)
    tax = float(market.tax.tax_estimate(item.item_id, quote.price)) if side == "sell" else 0.0
    return _Priced(quote.price, tax, quote)


def _source_text(quote: Quote | None) -> str | None:
    if quote is None:
        return None
    return quote.source + (f" ({quote.age_s // 60} min ago)" if quote.age_s else "")


def _line(item: GuideItem, per_hour: float, priced: _Priced, side: Side) -> Line:
    net = priced.unit - priced.tax if side == "sell" else priced.unit
    quote = priced.quote
    return Line(
        item=item.name,
        per_hour=round(per_hour, 2),
        unit_price=round(priced.unit, 2),
        value_per_hour=round(per_hour * net, 2),
        price_source=_source_text(quote),
        market_volume_per_hour=quote.volume if quote and quote.volume else None,
        note=priced.note or item.note,
    )


def value_guide(guide: MoneyMakingGuide, market: Market) -> Valuation:
    """Live profit per hour of one guide; raises ``Unpriced`` when it cannot be valued exactly.

    For recurring guides the hour is an hour of ACTIVE time (the wiki's convention).
    """
    cfg = market.cfg
    gross: dict[Side, float] = {"sell": 0.0, "buy": 0.0}
    priced_lines: list[tuple[GuideItem, float, _Priced, Side]] = []
    sides: tuple[tuple[Side, list[GuideItem]], ...] = (
        ("buy", guide.items_in),
        ("sell", guide.items_out),
    )
    for side, items in sides:
        for item in items:
            if item.qty_per_hour is None:
                raise Unpriced("no_quantity", f"{item.name}: quantity per hour unknown")
            priced = _price_item(item, side, market)
            gross[side] += item.qty_per_hour * priced.unit
            priced_lines.append((item, item.qty_per_hour, priced, side))

    # How much of the guide's pace can the market absorb? Only items that make up a real share of
    # the value matter; one that is capped limits the whole activity (everything scales together).
    pace = 1.0
    limiting: tuple[str, str, float, int] | None = None
    for item, per_hour, priced, side in priced_lines:
        quote = priced.quote
        if quote is None or per_hour <= 0 or gross[side] <= 0:
            continue
        if per_hour * priced.unit / gross[side] < cfg.liquidity_min_share:
            continue
        verb = "sells" if side == "sell" else "buys"
        if quote.volume <= 0:
            raise Unpriced(
                "no_volume",
                f"{item.name}: no trades in the last hour, only a last-trade quote "
                f"(the guide {verb} {per_hour:,.0f}/h)",
            )
        allowed = cfg.market_share * quote.volume
        if per_hour > allowed and allowed / per_hour < pace:
            pace = allowed / per_hour
            limiting = (item.name, verb, per_hour, quote.volume)

    valuation = Valuation(pace=pace)
    for item, per_hour, priced, side in priced_lines:
        line = _line(item, per_hour * pace, priced, side)
        if side == "sell":
            valuation.revenue_per_hour += line.value_per_hour
            valuation.outputs.append(line)
        else:
            valuation.cost_per_hour += line.value_per_hour
            valuation.inputs.append(line)
    if limiting is not None:
        name, verb, per_hour, volume = limiting
        valuation.warnings.append(
            f"limited by the market: the guide {verb} {per_hour:,.0f} {name}/h but only "
            f"{volume:,}/h trade; at a {cfg.market_share:.0%} share of that you can run about "
            f"{pace:.0%} of its pace, and the figures below are scaled to that"
        )
    return valuation


# --------------------------------------------------------------------------- ranking
class Missing(BaseModel):
    skills: list[dict[str, Any]] = Field(default_factory=list)
    quests: list[str] = Field(default_factory=list)
    other: list[dict[str, Any]] = Field(default_factory=list)


class RowLine(BaseModel):
    """One item of a guide, per hour (continuous guides) or per run (recurring ones)."""

    item: str
    qty: float
    unit_price: float
    value: float = Field(description="after Grand Exchange tax for sold items")
    price_source: str | None = None
    market_volume_per_hour: int | None = None
    note: str | None = None


class Row(BaseModel):
    title: str
    page: str
    url: str
    category: str | None = None
    intensity: str | None = None
    kind: str
    members_only: bool | None = None
    # Continuous guides: figures per hour.
    profit_per_hour: int | None = None
    revenue_per_hour: int | None = None
    cost_per_hour: int | None = None
    reference_profit_per_hour: int | None = Field(
        None, description="the Wiki's own figure from an earlier date: context only"
    )
    # Recurring guides: figures per run (the Wiki's list quotes per run too).
    profit_per_run: int | None = None
    active_minutes: float | None = None
    repeats_every: str | None = None
    reference_profit_per_run: int | None = Field(
        None, description="the Wiki's own per-run figure from an earlier date: context only"
    )
    pace: float | None = Field(
        None,
        description="share of the guide's own pace the market allows; omitted when it is all of it",
    )
    capital_recommended: int | None = None
    main_outputs: list[RowLine] = Field(default_factory=list)
    main_inputs: list[RowLine] = Field(default_factory=list)
    requirements: dict[str, Any] | None = Field(
        None, description="what the guide needs; only listed when the character is unknown"
    )
    you_meet: list[str] = Field(
        default_factory=list,
        description="hard requirements already checked against the character and met",
    )
    wiki_requirements: list[str] = Field(
        default_factory=list,
        description="the wiki's own requirement lines: they mix hard requirements with "
        "recommended levels and the gear (Item: ...) the method uses",
    )
    missing: Missing | None = None
    caveats: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _short_title(guide: MoneyMakingGuide) -> str:
    return guide.page_title.removeprefix(GUIDE_PREFIX)


def _row(guide: MoneyMakingGuide, cfg: MoneyConfig) -> Row:
    recurring = guide.kind == "recurring"
    return Row(
        title=_short_title(guide),
        page=guide.page_title,
        url=page_url(guide.page_title, cfg.wiki_base),
        category=guide.category,
        intensity=guide.intensity,
        kind=guide.kind,
        members_only=guide.members_only,
        reference_profit_per_hour=None if recurring else guide.source_hourly_profit,
        reference_profit_per_run=guide.source_hourly_profit if recurring else None,
        capital_recommended=guide.capital,
    )


def _requirements(guide: MoneyMakingGuide) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if guide.skill_requirements:
        out["skills"] = dict(sorted(guide.skill_requirements.items()))
    if guide.quest_requirements:
        out["quests"] = list(guide.quest_requirements)
    if guide.combat_level:
        out["combat_level"] = guide.combat_level
    if guide.quest_points:
        out["quest_points"] = guide.quest_points
    if guide.total_level:
        out["total_level"] = guide.total_level
    return out


def _row_lines(lines: list[Line], scale: float) -> list[RowLine]:
    top = sorted(lines, key=lambda line: -abs(line.value_per_hour))[:MAIN_LINES]
    return [
        RowLine(
            item=line.item,
            qty=round(line.per_hour * scale, 2),
            unit_price=line.unit_price,
            value=round(line.value_per_hour * scale, 2),
            price_source=line.price_source,
            market_volume_per_hour=line.market_volume_per_hour,
            note=line.note,
        )
        for line in top
    ]


def _fill_values(row: Row, guide: MoneyMakingGuide, valuation: Valuation) -> None:
    """Put the live figures on a row: per hour for continuous guides, per run for recurring."""
    if guide.kind == "recurring":
        minutes = guide.activity_minutes or 0.0
        scale = minutes / 60  # lines are per hour of active time; a run lasts `minutes`
        row.profit_per_run = round(valuation.profit_per_hour * scale)
        row.active_minutes = minutes
        row.repeats_every = guide.recurrence
    else:
        scale = 1.0
        row.profit_per_hour = round(valuation.profit_per_hour)
        row.revenue_per_hour = round(valuation.revenue_per_hour)
        row.cost_per_hour = round(valuation.cost_per_hour)
    row.main_outputs = _row_lines(valuation.outputs, scale)
    row.main_inputs = _row_lines(valuation.inputs, scale)
    row.warnings = list(valuation.warnings)
    if valuation.pace < 1.0:
        row.pace = round(max(valuation.pace, 1e-4), 4)  # never rounds a real limit down to 0


def _missing(e: Eligibility) -> Missing:
    return Missing(
        skills=[
            {"skill": g.name, "have": g.have, "need": g.need} for g in e.gaps if g.kind == "skill"
        ],
        quests=list(e.quests_missing),
        other=[
            {"requirement": g.name, "have": g.have, "need": g.need}
            for g in e.gaps
            if g.kind != "skill"
        ],
    )


@dataclass
class Ranking:
    ranked: list[Row] = field(default_factory=list)  # continuous guides, best profit/h first
    recurring: list[Row] = field(default_factory=list)  # best profit per run first
    almost: list[Row] = field(default_factory=list)
    needs_gp: list[Row] = field(default_factory=list)
    eligible_total: int = 0
    excluded: dict[str, int] = field(default_factory=dict)
    unpriced: list[str] = field(default_factory=list)  # a few examples of why guides were skipped

    def skip(self, reason: str) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + 1


def _category_matches(guide: MoneyMakingGuide, wanted: str | None) -> bool:
    if not wanted:
        return True
    return (guide.category or "").split("/")[0].strip().casefold() == wanted.strip().casefold()


def _top(rows: list[tuple[float, Row]], n: int) -> list[Row]:
    return [row for _, row in sorted(rows, key=lambda pair: (-pair[0], pair[1].title))[:n]]


def rank_guides(
    guides: Iterable[MoneyMakingGuide],
    player: PlayerState,
    market: Market,
    *,
    top_n: int = 8,
    category: str | None = None,
    include_recurring: bool = True,
    stated_combat: int | None = None,
    show_requirements: bool = False,
) -> Ranking:
    """Filter by what the player meets, value with live prices, order by profit per hour."""
    cfg = market.cfg
    result = Ranking()
    valued: list[tuple[float, Row]] = []
    recurring: list[tuple[float, Row]] = []
    almost: list[tuple[float, Row]] = []
    needs_gp: list[tuple[float, Row]] = []
    for guide in guides:
        if guide.obsolete:
            result.skip("obsolete_on_wiki")
            continue
        if guide.unsupported or not guide.items_out:
            result.skip("not_computable_from_the_wiki")
            continue
        is_recurring = guide.kind == "recurring"
        if not _category_matches(guide, category) or (is_recurring and not include_recurring):
            continue
        if guide.members_only and player.members is False:
            result.skip("members_only")
            continue
        eligibility = check_requirements(guide, player, stated_combat=stated_combat)
        status = eligibility.status(cfg)
        if status == "ineligible" or (is_recurring and status == "almost"):
            result.skip("requirements_not_met")
            continue
        try:
            valuation = value_guide(guide, market)
        except Unpriced as exc:
            result.skip(f"unpriced:{exc.reason}")
            if len(result.unpriced) < 5:
                result.unpriced.append(f"{_short_title(guide)}: {exc}")
            continue
        if is_recurring and not guide.activity_minutes:
            result.skip("unpriced:no_quantity")
            continue
        if not math.isfinite(valuation.profit_per_hour):
            result.skip("unpriced:not_finite")
            continue
        row = _row(guide, cfg)
        _fill_values(row, guide, valuation)
        row.caveats = [] if show_requirements else list(eligibility.caveats)
        row.wiki_requirements = [line[:WIKI_LINE_CHARS] for line in guide.requirements_text][
            :WIKI_LINES
        ]
        if show_requirements:
            row.requirements = _requirements(guide)
        else:
            row.you_meet = list(eligibility.met)
        if guide.members_only is None and player.members is False:
            row.caveats.append("the Wiki does not say whether this guide needs membership")
        # Continuous guides compete on profit per hour; recurring ones on profit per run.
        key = float(row.profit_per_run if is_recurring else row.profit_per_hour or 0)
        if status == "almost":
            row.missing = _missing(eligibility)
            almost.append((key, row))
        elif player.gp is not None and guide.capital and guide.capital > player.gp:
            if not is_recurring:
                needs_gp.append((key, row))
            else:
                result.skip("recurring_needs_more_gp")
        elif is_recurring:
            recurring.append((key, row))
        else:
            valued.append((key, row))

    result.eligible_total = len(valued)
    result.ranked = _top(valued, top_n)
    result.recurring = _top(recurring, RECURRING_LIST_LIMIT)
    result.almost = _top(almost, ALMOST_LIST_LIMIT)
    result.needs_gp = _top(needs_gp, NEEDS_GP_LIST_LIMIT)
    return result


def ironman_list(
    guides: Iterable[MoneyMakingGuide],
    player: PlayerState,
    cfg: MoneyConfig,
    *,
    category: str | None = None,
) -> tuple[list[Row], int, int]:
    """``(rows, ironman_guides, other_guides)``: the guides written for ironmen that the player
    qualifies for, with no profit figure at all, and how many ordinary guides also qualify."""
    friendly: list[MoneyMakingGuide] = []
    other = 0
    for guide in guides:
        if guide.obsolete or not _category_matches(guide, category):
            continue
        if guide.members_only and player.members is False:
            continue
        if check_requirements(guide, player).status(cfg) != "eligible":
            continue
        if guide.ironman_friendly:
            friendly.append(guide)
        else:
            other += 1
    friendly.sort(key=_short_title)
    rows = []
    for guide in friendly[:IRONMAN_LIST_LIMIT]:
        row = _row(guide, cfg)
        # Grand Exchange figures do not apply to ironmen: no number of any kind goes out.
        row.reference_profit_per_hour = row.reference_profit_per_run = None
        row.capital_recommended = None
        rows.append(row)
    return rows, len(friendly), other


# --------------------------------------------------------------------------- the tool
def _stamp(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _account(player: PlayerState) -> dict[str, Any]:
    return {
        "type": player.account_type.value,
        "members": player.members,
        "members_known": player.members is not None,
        "gp": player.gp,
    }


def close_to_lines(rows: list[Row]) -> list[str]:
    """One plain sentence per almost-eligible guide: what is missing, worded by the code."""
    lines = []
    for row in rows:
        missing = row.missing
        if missing is None:
            continue
        parts = [f"{m['skill']} {m['need']} (you have {m['have']})" for m in missing.skills]
        parts += [f"the quest {quest}" for quest in missing.quests]
        parts += [f"{m['requirement']} {m['need']} (you have {m['have']})" for m in missing.other]
        worth = f" ~{row.profit_per_hour:,} GP/h" if row.profit_per_hour else ""
        lines.append(f"{row.title}:{worth} once you have " + " and ".join(parts))
    return lines


def _dump(rows: list[Row]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json", exclude_none=True) for row in rows]


PRICE_BASIS_TEXT = {
    "patient": "outputs valued at the average instant-buy (high) price, inputs at the "
    "average instant-sell (low) price, like a flip",
    "mid": "outputs and inputs valued at the midpoint of the high and low prices",
    "instant": "outputs valued at the instant-sell price, inputs at the instant-buy price",
}


async def rank_moneymakers(
    ctx: ToolContext,
    player: PlayerState,
    top_n: int = 8,
    category: str | None = None,
    include_recurring: bool = True,
    stated_combat_level: int | None = None,
) -> dict[str, Any]:
    """SPEC §8 ``rank_moneymakers(player_state)``: ranking with live profit per hour.

    Without any character data (an empty snapshot: the plugin could not read one) requirements
    cannot be checked, so every guide stays in the ranking and lists what it needs; a combat
    level the player stated in words still filters on combat.
    """
    if ctx.guides is None or not len(ctx.guides):
        return {
            "error": "guides_unavailable",
            "message": "The structured money-making guides are not loaded; do not guess figures.",
        }
    cfg = MoneyConfig.from_settings(ctx.settings)
    guides = ctx.guides.guides
    account = _account(player)
    if player.is_ironman:
        rows, total, other = ironman_list(guides, player, cfg, category=category)
        return {
            "status": "not_applicable_ironman",
            "account": account,
            "message": (
                "Ironman accounts cannot trade on the Grand Exchange, so GP per hour priced at "
                "Grand Exchange values does not apply to them. No profit figures are given. "
                "Only the guides written for ironmen that the player meets the requirements for "
                f"are listed; the other {other} guides that the player qualifies for assume "
                "buying and selling on the Grand Exchange."
            ),
            "eligible_total": total,
            "eligible_guides": _dump(rows),
            "other_eligible_guides_assuming_ge": other,
            "advice_pages": IRONMAN_ADVICE_PAGES,
        }

    bundle = await ctx.bundle()
    latest, hourly = await asyncio.gather(ctx.prices.latest(), ctx.prices.one_hour())
    anonymous = not player.skills and not player.quests
    market = Market(
        latest=latest.data,
        hourly=hourly.data,
        index=bundle.index,
        tax=bundle.tax,
        now=ctx.now(),
        cfg=cfg,
    )
    ranking = rank_guides(
        guides,
        player,
        market,
        top_n=top_n,
        category=category,
        include_recurring=include_recurring,
        stated_combat=stated_combat_level,
        show_requirements=anonymous,
    )
    notes = [
        "profit_per_hour is recomputed now with live Grand Exchange prices "
        f"(basis: {cfg.basis}; tax on sales; see price_basis).",
        "reference_profit_per_hour / reference_profit_per_run is the Wiki's own figure from an "
        "earlier date: context only, never today's number.",
        "`recurring` activities (farming runs, daily purchases) are valued per run and are not "
        "comparable with profit per hour: mention them as side activities.",
    ]
    if anonymous:
        notes.append(
            "No character data: requirements were NOT checked against the player. Each row lists "
            "its `requirements`; recommend only guides that fit what the player said about "
            "themself (a stated combat level was already applied"
            + (f": {stated_combat_level}" if stated_combat_level else ": none was stated")
            + "), name what to verify, and offer to check precisely when the player is logged in."
        )
    if player.members is None:
        notes.append("Membership is not reported: members-only guides are included.")
    if player.gp is None:
        notes.append("The player's GP is unknown, so bankroll needs were not checked.")
    return {
        "status": "ok",
        "requirements_checked": not anonymous,
        "account": account,
        "price_basis": PRICE_BASIS_TEXT[cfg.basis],
        "tax_rate_pct": float(bundle.tax.rate * 100),
        "prices_fetched_at": _stamp(latest.fetched_at),
        "stale": latest.stale or hourly.stale,
        "eligible_total": ranking.eligible_total,
        "close_to": close_to_lines(ranking.almost),
        "ranking": _dump(ranking.ranked),
        "recurring": _dump(ranking.recurring),
        "almost_eligible": _dump(ranking.almost),
        "needs_more_gp": _dump(ranking.needs_gp),
        "excluded": dict(sorted(ranking.excluded.items())),
        "skipped_examples": ranking.unpriced,
        "notes": notes,
    }
