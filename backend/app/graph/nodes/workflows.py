"""Preparation of the CRAG-backed workflows (SPEC §7 ``quest_workflow``, ``moneymaking_workflow``).

Both do their deterministic part first and then hand over to the shared CRAG nodes, which retrieve,
grade and generate; the deterministic result travels in ``tool_results`` and the generation prompt
treats it as the only source for requirement checks and profit figures.

* quest: find the quest the player means, compare its requirements with the snapshot (skills,
  quests in the order to do them, quest points, combat level, items), and make retrieval include
  that quest's pages.
* moneymaking: ``rank_moneymakers`` over the player's snapshot (when there is one) and retrieval
  of the best guides' pages so the answer can detail them.
"""

from __future__ import annotations

import re

from langgraph.runtime import Runtime

from app.api.schemas import PlayerState
from app.graph.state import AgentState, RunContext
from app.tools.quests import check_quest
from app.tools.registry import run_tool

MAX_QUESTS_CHECKED = 2
GUIDES_TO_DETAIL = 2


# Without a snapshot the player's own words are all there is: a category they ask for and a combat
# level they state. Both only narrow the ranking; the figures still come from the code.
_CATEGORY_WORDS = {
    "Combat": re.compile(r"\b(combat|pvm|bossing|bosses|slayer|combate|jefes)\b", re.IGNORECASE),
    "Skilling": re.compile(r"\b(skilling|skill training)\b", re.IGNORECASE),
    "Collecting": re.compile(r"\b(collecting|gathering)\b", re.IGNORECASE),
    "Processing": re.compile(r"\b(processing)\b", re.IGNORECASE),
}
_COMBAT_LEVEL = (
    re.compile(
        r"\b(?:lvl|lv|level|nivel)\.?\s*(\d{2,3})\s*(?:de\s+)?(?:combat|combate|cb)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:combat|combate|cb)(?:\s+(?:level|lvl|nivel))?\s*(?:of|de|is|es|:)?\s*(\d{2,3})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{2,3})\s*(?:combat|combate|cb)\b", re.IGNORECASE),
)


def stated_category(*texts: str) -> str | None:
    """The one guide category the player asked for ("combat only"), if exactly one is named."""
    text = " ".join(texts)
    found = [name for name, pattern in _CATEGORY_WORDS.items() if pattern.search(text)]
    return found[0] if len(found) == 1 else None


def stated_combat_level(*texts: str) -> int | None:
    """A combat level the player wrote ("level 80 combat", "combat 90"), when there is one."""
    for text in texts:
        for pattern in _COMBAT_LEVEL:
            match = pattern.search(text)
            if match and 3 <= int(match.group(1)) <= 126:
                return int(match.group(1))
    return None


async def quest_prepare(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    named = ctx.deps.quests.resolve(state["query"], state["messages"][-1]["content"])
    if not named:
        return {}  # no specific quest recognised: plain CRAG over the wiki
    records = named[:MAX_QUESTS_CHECKED]
    titles: list[str] = []
    for record in records:
        titles += [record.title, f"{record.title}/Quick guide"]
    # No page-type restriction: a follow-up about an item of the quest ("¿y si no tengo el m'speak
    # amulet?") needs the item's page too; the quest's own pages are forced into the pool anyway.
    update: dict = {"force_titles": titles}
    if ctx.player is not None:
        bundle = await ctx.deps.tools.bundle()
        update["tool_results"] = [
            {"tool": "check_quest_requirements", "result": check_quest(r, ctx.player, bundle.index)}
            for r in records
        ]
        update["tools_called"] = ["check_quest_requirements"]
    return update


async def moneymaking_prepare(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    arguments: dict = {"top_n": 8}
    player = ctx.player
    if player is None:
        # No snapshot (not logged in): rank every guide by live profit and list what each needs,
        # narrowed by what the player said (category, combat level), instead of giving up.
        player = PlayerState()
        words = (state["query"], state["messages"][-1]["content"])
        arguments["top_n"] = 12
        category = stated_category(*words)
        if category:
            arguments["category"] = category
        level = stated_combat_level(*words[::-1])
        if level:
            arguments["stated_combat_level"] = level
    result = await run_tool(ctx.deps.tools, "rank_moneymakers", arguments, player=player)
    if (
        arguments.get("category")
        and result.get("status") == "ok"
        and not result.get("ranking")
        and ctx.player is None
    ):
        arguments.pop("category")  # nothing in that category fits: say so with the full ranking
        result = await run_tool(ctx.deps.tools, "rank_moneymakers", arguments, player=player)
    update: dict = {
        "tool_results": [{"tool": "rank_moneymakers", "result": result}],
        "tools_called": ["rank_moneymakers"],
    }
    if result.get("status") == "ok":
        update["force_titles"] = [row["page"] for row in result["ranking"][:GUIDES_TO_DETAIL]]
    elif result.get("status") == "not_applicable_ironman":
        update["force_titles"] = list(result.get("advice_pages", []))
    return update
