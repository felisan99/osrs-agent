"""Tool-calling workflows: ``flipping_workflow`` and ``price_lookup`` (SPEC §7).

The model chooses which tool to call and with what arguments (that is what it is good at); the
tools do all the arithmetic and the prices, and the model explains the result. Around that loop,
code enforces the principles the prompt can only request:

* the first round must call a tool (no answering prices from memory);
* every number of 1,000 or more in the answer must come from a tool result, the player's words or
  the wiki (``grounding``): one rewrite is requested, then the offending sentences are dropped;
* the answer always ends with when the prices were fetched (SPEC §7 prompt rules);
* if the price API is down, a fixed message replaces the answer.
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from app.graph.footer import with_price_footer
from app.graph.format import clean_answer
from app.graph.grounding import drop_sentences_with, ungrounded_numbers
from app.graph.nodes.crag import history_messages
from app.graph.prompts import fixed, system_prompt
from app.graph.state import AgentState, RunContext
from app.graph.summary import player_summary
from app.graph.usage import add_llm
from app.tools.registry import run_tool, tool_specs

MAX_ROUNDS = 6
MAX_TOOL_MESSAGE_CHARS = 12_000
FLIP_TOOLS = ("calc_flip", "analyze_flip", "rank_flips", "get_price_history")
PRICE_TOOLS = ("get_item_price", "get_price_history")
PRICES_SOURCE = {"title": "OSRS Wiki real-time prices", "url": "https://prices.runescape.wiki/"}

FLIP_GUIDANCE = (
    "You help the player flip on the Grand Exchange with tools, and every recommendation must be "
    "JUSTIFIED with data from them. Tools: `rank_flips` for 'what should I flip' (the player's "
    "GP is max_gp when they give no budget and it is known in the player summary; if neither is "
    "known, ask for the budget instead of guessing; members=false for a free-to-play player; "
    "strategy `volume` by default, because liquid items whose price one trader cannot move are "
    "the safest, and `margin`, `overnight` or `dip` only when the player asks for bigger "
    "margins, long offers, or buying dips); `analyze_flip` for 'is X worth flipping?'; "
    "`calc_flip` for the maths of one trade (pass buy_price/sell_price only when the player "
    "states a price); `get_price_history` for a trend the player asks about.\n"
    "HOW TO ANSWER: pick the best 2-3 rows (not all) and talk like a flipper friend, in flowing "
    "text with no labels like 'Evidence:' or 'Flags:'. For each, say the item and the prices "
    "to use (buy offer at `buy_at`, sell offer at `sell_at`), how many units, and the profit "
    "per window (lead with `profit_per_window_conservative`; say `typical` is the usual). Then "
    "CONVINCE with two or three of its `evidence` facts quoted as given (the share of hours the "
    "margin was positive, the volume, the buy/sell balance): that is the point of this answer. "
    "If it has `flags`, relay them honestly in plain words (say nothing when it has none). "
    "Say which strategy you used and that "
    "the others exist (margin, overnight, dip) in one line. Close with practical steps in two "
    "or three sentences: place the buy offer, wait for it to fill (`est_hours_to_fill`), then "
    "list at the sell price; mention `offer_slots` and not putting the whole bank in one item. "
    "If no row qualifies, say so and why (`excluded`); never fill the list with weak items. "
    "Prices move, so flipping is never guaranteed, and history cannot see manipulation or "
    "upcoming updates.\n"
    "PRINCIPLES (from flipping guides): liquid, cheap items with a buy/sell ratio near 1 are safer "
    "than thinly traded ones; a price far above its usual level or at a 30-day high may be a "
    "bubble; expect about 3% spread before tax to make a profit after it; limit offers beat "
    "instant trades. Use them to explain, never as a source of numbers. Quote the tax rate only "
    "as the tool gives it (`tax_rate_pct`). Never state a figure that is not in a tool result."
)
PRICE_GUIDANCE = (
    "You answer questions about Grand Exchange prices with tools: `get_item_price` for current "
    "prices (also the buy limit and alchemy values) and `get_price_history` for trends. If the "
    "item name is ambiguous the tool returns candidates: pick the obvious one or ask. Quote the "
    "prices exactly as the tools give them, name which side is which (instant buy / instant "
    "sell) and always name the exact item the figures are for (the tool's `item`, doses included)."
)


def _all_failed(results: list[dict[str, Any]]) -> bool:
    return bool(results) and all(
        isinstance(r["result"], dict) and r["result"].get("error") == "price_data_unavailable"
        for r in results
    )


def _grounding_sources(state: AgentState, results: list[dict[str, Any]]) -> list[str]:
    texts = [m["content"] for m in state["messages"]]  # the player's numbers (and our earlier ones)
    texts += [json.dumps(r["result"], ensure_ascii=False) for r in results]
    return texts


async def _tool_loop(
    state: AgentState, ctx: RunContext, tool_names: tuple[str, ...], guidance: str
) -> dict:
    language = state.get("language", "es")
    system = f"{system_prompt(language, guidance)}\n\nPLAYER\n{player_summary(ctx.player)}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *history_messages(state, ctx),
        {"role": "user", "content": state["messages"][-1]["content"]},
    ]
    specs = tool_specs(tool_names)
    usage = state.get("usage")
    results: list[dict[str, Any]] = []
    called: list[str] = []
    answer = ""

    for round_number in range(MAX_ROUNDS):
        reply = await ctx.llm.chat(messages, tools=specs, tool_choice="auto")
        usage = add_llm(usage, reply)
        if reply.tool_calls:
            messages.append(reply.assistant_message)
            for call in reply.tool_calls:
                if call.name not in tool_names:
                    payload: dict[str, Any] = {
                        "error": "unknown_tool",
                        "available": list(tool_names),
                    }
                elif call.parse_error:
                    payload = {"error": "invalid_arguments", "message": call.parse_error}
                else:
                    payload = await run_tool(
                        ctx.deps.tools, call.name, call.arguments, player=ctx.player
                    )
                called.append(call.name)
                results.append({"tool": call.name, "arguments": call.arguments, "result": payload})
                text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": text[:MAX_TOOL_MESSAGE_CHARS],
                    }
                )
            continue
        if not results and round_number == 0:
            # No prices from memory: the first answer must rest on a tool call.
            messages.append(reply.assistant_message or {"role": "assistant", "content": ""})
            messages.append(
                {"role": "user", "content": "Call the tools to get real data before answering."}
            )
            continue
        answer = (reply.content or "").strip()
        break
    else:
        messages.append(
            {"role": "user", "content": "Answer now using only what the tools returned."}
        )
        reply = await ctx.llm.chat(messages)
        usage = add_llm(usage, reply)
        answer = (reply.content or "").strip()

    if _all_failed(results) or not answer:
        return {
            "answer": fixed("tool_failed", language),
            "sources": [],
            "tool_results": results,
            "tools_called": called,
            "usage": usage,
        }

    sources = _grounding_sources(state, results)
    bad = ungrounded_numbers(answer, sources)
    if bad:
        messages.append({"role": "assistant", "content": answer})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your answer states figures that are not in the tool results: {bad}. "
                    "Rewrite it using only figures from the tool results, quoted as given, with "
                    "no new calculations. Keep the same language. Reply with the corrected "
                    "answer only: do not mention this correction or apologise."
                ),
            }
        )
        reply = await ctx.llm.chat(messages)
        usage = add_llm(usage, reply)
        retried = (reply.content or "").strip()
        answer = retried or answer
        bad = ungrounded_numbers(answer, sources)
        if bad:
            answer = drop_sentences_with(answer, bad)

    answer = with_price_footer(clean_answer(answer), language, results)
    return {
        "answer": answer or fixed("no_evidence", language),
        "sources": [PRICES_SOURCE] if results else [],
        "tool_results": results,
        "tools_called": called,
        "usage": usage,
    }


async def flipping_workflow(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    if ctx.player is not None and ctx.player.is_ironman:
        return {"answer": fixed("flip_ironman", state.get("language", "es")), "sources": []}
    return await _tool_loop(state, ctx, FLIP_TOOLS, FLIP_GUIDANCE)


async def price_workflow(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    return await _tool_loop(state, runtime.context, PRICE_TOOLS, PRICE_GUIDANCE)
