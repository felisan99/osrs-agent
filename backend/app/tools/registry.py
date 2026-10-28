"""Tool registry: function-calling schemas + validated dispatch (SPEC §6, §7).

Every tool returns structured JSON and the LLM never calculates (principle 1). Failures that the
model can act on (ambiguous item, invalid argument, price API down) are returned as
``{"error": ...}`` payloads instead of raised, so a tool loop can recover or explain.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.api.schemas import PlayerState
from app.errors import UpstreamError
from app.tools import flips, moneymakers, prices
from app.tools.context import ToolContext

Lookback = Literal["6h", "24h", "7d", "30d", "6m", "1y"]

_PRICE_SEMANTICS = (
    "Price semantics: `instant_buy_price` (the Wiki API's `high`) is what a buyer pays right now; "
    "`instant_sell_price` (`low`) is what a seller receives right now."
)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetItemPriceArgs(_Args):
    item: str = Field(
        description="Item name as shown in game, e.g. 'Abyssal whip'. Common nicknames "
        "('tbow', 'bgs') also work."
    )


class GetPriceHistoryArgs(_Args):
    item: str = Field(description="Item name as shown in game.")
    lookback: Lookback = Field("7d", description="Window to summarise.")


class CalcFlipArgs(_Args):
    item: str = Field(description="Item name as shown in game.")
    qty: int | None = Field(
        None, ge=1, description="Units the player wants to flip; omit for per-unit numbers only."
    )
    buy_price: int | None = Field(
        None, gt=0, description="ONLY if the player states the price they would buy at."
    )
    sell_price: int | None = Field(
        None, gt=0, description="ONLY if the player states the price they would sell at."
    )


class AnalyzeFlipArgs(_Args):
    item: str = Field(description="Item name as shown in game.")
    budget: int | None = Field(
        None, gt=0, description="GP the player can spend, if known; omit otherwise."
    )


class RankFlipsArgs(_Args):
    max_gp: int = Field(gt=0, description="GP the player can spend on the buy side.")
    members: bool = Field(True, description="False restricts the ranking to free-to-play items.")
    min_volume: int | None = Field(
        None,
        ge=0,
        description="Minimum units traded per hour on BOTH sides; omit to use the default.",
    )
    top_n: int = Field(5, ge=1, le=10, description="How many rows to return.")
    strategy: Literal["volume", "margin", "overnight", "dip"] = Field(
        "volume",
        description="volume (default): liquid items, many repeats; margin: bigger margin on "
        "slower items; overnight: long offers left for hours; dip: speculative, below its usual "
        "price.",
    )


class RankMoneymakersArgs(_Args):
    top_n: int = Field(8, ge=1, le=15, description="How many ranked guides to return.")
    category: str | None = Field(
        None,
        description="Restrict to one guide category: Combat, Skilling, Processing, Collecting, "
        "Recurring... Omit for all.",
    )
    include_recurring: bool = Field(
        True, description="False drops recurring activities (farming runs, daily purchases)."
    )
    stated_combat_level: int | None = Field(
        None,
        ge=3,
        le=126,
        description="Only when the system has no character data: the combat level the player "
        "stated in their own words. Never guess it.",
    )


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    args: type[BaseModel]
    handler: Callable[..., Awaitable[dict[str, Any]]]
    # The player's snapshot comes from the request, never from the model: such tools receive it
    # from the runtime and expose no argument for it.
    needs_player: bool = False


TOOLS: dict[str, ToolDef] = {
    t.name: t
    for t in (
        ToolDef(
            "get_item_price",
            "Current Grand Exchange prices for ONE item, with timestamps, last-hour volume, the "
            "4-hour buy limit and alchemy values. If the name is ambiguous it returns candidates: "
            f"call again with an exact name. Never state a price that did not come from a tool. "
            f"{_PRICE_SEMANTICS}",
            GetItemPriceArgs,
            prices.get_item_price,
        ),
        ToolDef(
            "get_price_history",
            "Summary of one item's price history over a lookback window: min/max/median, trend "
            "(direction, % change, slope, volatility), traded volume and a downsampled series. "
            f"{_PRICE_SEMANTICS}",
            GetPriceHistoryArgs,
            prices.get_price_history,
        ),
        ToolDef(
            "calc_flip",
            "Deterministic flip economics for ONE item: buy near the instant-sell price, sell near "
            "the instant-buy price, after the Grand Exchange tax (rate, cap and exempt items are "
            "applied by the code). Returns margin, ROI, 4-hour buy limit, max profit per limit "
            "window, totals for `qty` if given, liquidity and warnings to relay to the player.",
            CalcFlipArgs,
            flips.calc_flip,
        ),
        ToolDef(
            "analyze_flip",
            "Is ONE item a good flip? Uses its last 7 and 30 days of prices and volume: how often "
            "the margin after tax was positive, typical and conservative margin, volume per hour, "
            "buy/sell balance, price against its recent median and range, how long an offer takes "
            "to fill. Returns a verdict, `evidence` (facts to quote), `flags` (risks) and a worked "
            "example. Use it for 'is X worth flipping?'.",
            AnalyzeFlipArgs,
            flips.analyze_flip,
        ),
        ToolDef(
            "rank_flips",
            "Find the best flips within a GP budget, justified with history. A snapshot of every "
            "item picks candidates; then each one's last 7 days (and 30 for the best) decide "
            "whether the margin is repeatable and tradeable. `strategy`: volume (default), margin, "
            "overnight, dip. Every row has `evidence`, `flags`, prices to use and a worked "
            "example. Use it for 'what should I flip' questions. Not available to ironmen.",
            RankFlipsArgs,
            flips.rank_flips,
        ),
        ToolDef(
            "rank_moneymakers",
            "Rank the money-making guides the CURRENT PLAYER can do, with profit per hour "
            "recomputed from live Grand Exchange prices (tax and market volume applied by the "
            "code). The player's stats, quests, membership and GP are supplied by the system: do "
            "not pass them. Also returns guides the player is almost able to do, guides that need "
            "more GP, and why others were excluded. For ironman accounts it returns the eligible "
            "guides WITHOUT profit figures (the Grand Exchange does not apply to them).",
            RankMoneymakersArgs,
            moneymakers.rank_moneymakers,
            needs_player=True,
        ),
    )
}


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated ``title`` keys: noise (and tokens) for function calling."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" and isinstance(value, str):
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {name: _strip_titles(sub) for name, sub in value.items()}
            else:
                out[key] = _strip_titles(value)
        return out
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


def tool_specs(names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """OpenAI-format ``tools`` array for the requested tools (default: all)."""
    selected = [TOOLS[n] for n in (names if names is not None else TOOLS)]
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": _strip_titles(t.args.model_json_schema()),
            },
        }
        for t in selected
    ]


async def run_tool(
    ctx: ToolContext,
    name: str,
    arguments: dict[str, Any],
    *,
    player: PlayerState | None = None,
) -> dict[str, Any]:
    """Validate ``arguments`` and run a tool; failures come back as ``{"error": ...}`` data."""
    tool = TOOLS.get(name)
    if tool is None:
        return {
            "error": "unknown_tool",
            "message": f"No tool named {name!r}",
            "available": sorted(TOOLS),
        }
    try:
        args = tool.args.model_validate(arguments)
    except ValidationError as exc:
        return {
            "error": "invalid_arguments",
            "details": [
                {"field": ".".join(str(part) for part in e["loc"]), "problem": e["msg"]}
                for e in exc.errors()
            ],
        }
    extra: dict[str, Any] = {}
    if tool.needs_player:
        if player is None:
            return {
                "error": "player_state_missing",
                "message": "This tool needs the player's snapshot (stats, quests), which the "
                "plugin did not send. Tell the player you need it; do not assume their levels.",
            }
        extra["player"] = player
    try:
        return await tool.handler(ctx, **extra, **args.model_dump())
    except UpstreamError as exc:
        return {
            "error": "price_data_unavailable",
            "message": "The Wiki prices API could not be reached; do not guess prices.",
            "detail": exc.message,
        }
