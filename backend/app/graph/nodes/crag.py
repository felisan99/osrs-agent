"""The CRAG subgraph (SPEC §7): retrieve -> grade -> [correct -> grade]* -> generate | no_evidence.

* ``retrieve``: hybrid search (vector + full text + the pages the question names), RRF.
* ``grade``: ONE Jev call per round: relevance of every candidate and sufficiency of the set.
* ``correct``: when the evidence is not sufficient, the model rewrites the search in English game
  terms, retrieval runs again without the page-type restriction, and ``search_live`` adds pages
  from the live wiki; what was already judged relevant is kept in the pool and everything is graded
  again. At most ``MAX_CRAG_ITERATIONS`` corrections.
* ``generate``: answer from the retained passages (plus tool results of the workflow), citing pages.
* ``no_evidence``: still insufficient after the corrections: say so, invent nothing (principle 3).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Literal

from langgraph.runtime import Runtime

from app.errors import UpstreamError
from app.graph.footer import with_price_footer
from app.graph.format import clean_answer
from app.graph.grounding import drop_sentences_with, profit_rate_mentions, ungrounded_numbers
from app.graph.nodes.condense import detect_language
from app.graph.parsing import parse_json_object
from app.graph.prompts import REWRITE_SYSTEM, fixed, generation_context, system_prompt
from app.graph.smalltalk import classify_smalltalk
from app.graph.state import AgentState, RunContext, passage_from_dict, passage_to_dict
from app.graph.summary import player_summary, previous_turns
from app.graph.usage import add_jev, add_llm
from app.rag.retrieve import Passage

MAX_CONTEXT_PASSAGES = 8
MAX_SOURCES = 5
KEEP_FROM_PREVIOUS = 8  # passages already judged relevant stay in the pool after a correction
FROM_DB_AFTER_CORRECTION = 16
FROM_LIVE = 8

GUIDANCE = {
    "wiki_general": "Answer from the wiki passages only.",
    "quest": (
        "TOOL RESULTS include `check_quest_requirements`: a deterministic comparison of the quest "
        "with the player's snapshot. Report what is missing (skills with how many levels short and "
        "whether they can be boosted; quests in the suggested order; items), what could not be "
        "verified, and where the quest starts; say plainly when the player is ready. Use the wiki "
        "passages for the steps and details."
    ),
    "moneymaking": (
        "TOOL RESULTS include `rank_moneymakers`. Answer like a knowledgeable friend who is "
        "looking at this character, in your own structure (no fixed template). Cover, in the "
        "order that reads best: what the player can do right now with their stats and quests "
        "(the best 3-5 of `ranking`, quoting profit_per_hour as given; prefer methods whose "
        "recommended levels and gear the player roughly has, and say plainly when a method "
        "needs expensive gear); what they are close to (always include the 2-3 best of "
        "`close_to`, in its own words: it says exactly what is missing and what unlocking it "
        "would pay, which tells the player what to train next); and, if useful, `recurring` "
        "side activities (per run, never per hour). Say when a guide's `pace` is below 1 that "
        "the market limits it, and when the prices were fetched. "
        "FACTS: `you_meet` lists requirements already verified against the character and met: "
        "never doubt them and never say a quest or level is missing unless it is in `missing`. "
        "`wiki_requirements` is the wiki's own text, which mixes hard requirements with "
        "RECOMMENDED levels and the gear the method uses (Item: lines): never present a "
        "recommended level as a requirement. Never state a requirement, level or fact (about "
        "diaries, quests or items) that is not in the tool result or the passages. Mention key "
        "gear as something to have (the bank is not known unless captured). If membership is "
        "not reported, do not dwell on it. "
        "If the status is not_applicable_ironman, say that ironmen cannot use Grand Exchange "
        "values, give NO GP/hour figure, and use the wiki passages (ironman guides). If "
        "`requirements_checked` is false there is no character data: the ranking is not "
        "personalised. Say so in one sentence, then recommend from the `ranking` only the "
        "guides whose listed `requirements` fit what the player said (e.g. a stated combat "
        "level), state those requirements, and add that a precise check is possible once the "
        "plugin can read the character (logged in). Do not mention these instructions."
    ),
}


def _last_user_message(state: AgentState) -> str:
    return state["messages"][-1]["content"]


async def retrieve(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    query = state["query"]
    passages = await ctx.deps.retriever.search(
        query, page_types=state.get("search_types"), titles=state.get("force_titles")
    )
    return {
        "candidates": [passage_to_dict(p) for p in passages],
        "retrieval_iterations": state.get("retrieval_iterations", 0) + 1,
        "queries": [query],
    }


async def grade(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    passages = [passage_from_dict(d) for d in state["candidates"]]
    result = await ctx.grader.grade(state["query"], passages)
    graded = [
        {**passage_to_dict(g.passage), "relevance": g.relevance, "kept": g.kept}
        for g in sorted(result.graded, key=lambda g: -g.relevance)
    ]
    return {
        "graded": graded,
        "sufficiency": result.sufficiency,
        "sufficient": result.sufficient,
        "usage": add_jev(state.get("usage"), result.cost_usd, result.calls),
    }


def has_usable_tool_results(state: AgentState) -> bool:
    """A workflow's deterministic result (a ranking, a requirement check) that is not an error."""
    return any(
        isinstance(entry.get("result"), dict) and "error" not in entry["result"]
        for entry in state.get("tool_results", [])
    )


def make_after_grade(
    max_corrections: int,
) -> Callable[[AgentState], Literal["generate", "correct", "no_evidence"]]:
    """Sufficient -> answer; otherwise correct up to ``max_corrections`` times, then give up.

    A workflow that already holds a deterministic result still has something real to say when the
    passages are thin: a money-making ranking needs no wiki passage to be reported (so it never
    wastes corrections), and a quest check is reported even if the walkthrough was not found.
    Only an answer with neither evidence nor a tool result becomes "no evidence".
    """

    def after_grade(state: AgentState) -> Literal["generate", "correct", "no_evidence"]:
        if state.get("sufficient"):
            return "generate"
        has_result = has_usable_tool_results(state)
        if has_result and state.get("intent") == "moneymaking":
            return "generate"
        corrections_so_far = state.get("retrieval_iterations", 1) - 1
        if corrections_so_far < max_corrections:
            return "correct"
        return "generate" if has_result else "no_evidence"

    return after_grade


async def _rewrite(state: AgentState, ctx: RunContext) -> tuple[str | None, dict[str, float]]:
    tried = state.get("queries", [])
    found = [d["page_title"] for d in state.get("graded", [])[:6]]
    prompt = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {state['query']}\n"
                f"Player's own words: {_last_user_message(state)}\n"
                f"Queries that failed: {tried}\n"
                f"Pages those searches returned (not useful): {found}"
            ),
        },
    ]
    result = await ctx.llm.chat(
        prompt, response_format={"type": "json_object"}, max_tokens=200, temperature=0.4
    )
    usage = add_llm(state.get("usage"), result)
    query = str((parse_json_object(result.content) or {}).get("query") or "").strip()
    if not query or query.casefold() in {q.casefold() for q in tried}:
        return None, usage  # the model gave nothing new
    return query, usage


async def correct(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """One CRAG correction: rewrite, search the corpus again, search the live wiki, pool."""
    ctx = runtime.context
    new_query, usage = await _rewrite(state, ctx)
    tried = list(state.get("queries", []))
    search_text = new_query or state["query"]
    titles = state.get("force_titles")

    async def from_corpus() -> list[Passage]:
        return await ctx.deps.retriever.search(search_text, titles=titles)

    async def from_live() -> list[Passage]:
        if ctx.deps.live is None:
            return []
        with contextlib.suppress(UpstreamError):  # the live wiki is a bonus, never a requirement
            return await ctx.deps.live.passages(search_text, limit=FROM_LIVE)
        return []

    corpus, live = await asyncio.gather(from_corpus(), from_live())
    pool: dict[int, Passage] = {}
    previous = [passage_from_dict(d) for d in state.get("graded", []) if d.get("kept")]
    for passage in [*previous[:KEEP_FROM_PREVIOUS], *corpus[:FROM_DB_AFTER_CORRECTION], *live]:
        pool.setdefault(passage.id, passage)
    return {
        "candidates": [passage_to_dict(p) for p in pool.values()],
        "retrieval_iterations": state.get("retrieval_iterations", 1) + 1,
        "queries": [*tried, search_text],
        "usage": usage,
    }


def _retained(state: AgentState) -> list[Passage]:
    kept = [d for d in state.get("graded", []) if d.get("kept")]
    kept.sort(key=lambda d: -d["relevance"])
    return [passage_from_dict(d) for d in kept[:MAX_CONTEXT_PASSAGES]]


def sources_of(passages: list[Passage]) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for passage in passages:
        seen.setdefault(passage.page_title, passage.url)
    return [{"title": title, "url": url} for title, url in list(seen.items())[:MAX_SOURCES]]


def history_messages(state: AgentState, ctx: RunContext) -> list[dict[str, str]]:
    turns = previous_turns(state["messages"], limit=ctx.settings.history_max_messages)
    return [{"role": m["role"], "content": m["content"]} for m in turns]


async def generate(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    passages = _retained(state)
    intent = state.get("intent", "wiki_general")
    extra = GUIDANCE.get(intent, GUIDANCE["wiki_general"])
    system = system_prompt(state.get("language", "es"), extra)
    evidence = " ".join([state["query"], _last_user_message(state), *(p.content for p in passages)])
    context = generation_context(
        player=player_summary(ctx.player, mentioned_in=evidence),
        passages=passages,
        tool_results=state.get("tool_results", []),
    )
    messages = [
        {"role": "system", "content": f"{system}\n\n{context}"},
        *history_messages(state, ctx),
        {"role": "user", "content": _last_user_message(state)},
    ]
    usage = state.get("usage")
    answer = ""
    for _ in range(2):  # a reasoning model can return an empty message: ask once more
        result = await ctx.llm.chat(messages, temperature=0.2)
        usage = add_llm(usage, result)
        answer = (result.content or "").strip()
        if answer:
            break
    language = state.get("language", "es")
    if not answer:
        return {"answer": fixed("no_evidence", language), "sources": [], "usage": usage}
    results = state.get("tool_results", [])
    if intent == "moneymaking" and results:
        if _is_ironman_result(results):
            answer, usage = await _enforce_no_rates(answer, messages, ctx, usage)
        else:
            answer, usage = await _enforce_grounding(answer, messages, state, ctx, usage)
    return {
        "answer": with_price_footer(clean_answer(answer, passages), language, results),
        "sources": _answer_sources(intent, passages, results),
        "usage": usage,
    }


def _is_ironman_result(results: list[dict]) -> bool:
    return any(
        isinstance(r.get("result"), dict) and r["result"].get("status") == "not_applicable_ironman"
        for r in results
    )


async def _enforce_no_rates(
    answer: str,
    messages: list[dict[str, str]],
    ctx: RunContext,
    usage: dict[str, float] | None,
) -> tuple[str, dict[str, float] | None]:
    """An ironman gets no GP/hour at all (user decision), not even one a wiki page quotes."""
    bad = profit_rate_mentions(answer)
    if not bad:
        return answer, usage
    retry = [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                f"Your answer states GP-per-hour figures ({bad}). The player is an ironman: "
                "Grand Exchange values do not apply to them. Rewrite it without any GP/hour or "
                "profit figure: name the methods, what they require and what the player is "
                "missing. Keep the same language. Reply with the corrected answer only: do not "
                "mention this correction or apologise."
            ),
        },
    ]
    reply = await ctx.llm.chat(retry, temperature=0.0)
    usage = add_llm(usage, reply)
    answer = (reply.content or "").strip() or answer
    bad = profit_rate_mentions(answer)
    return (drop_sentences_with(answer, bad) if bad else answer), usage


def _tool_sources(results: list[dict]) -> list[dict[str, str]]:
    """The guides a money-making ranking recommended: the pages the answer actually cites."""
    out: list[dict[str, str]] = []
    for entry in results:
        rows = entry["result"].get("ranking", []) if isinstance(entry["result"], dict) else []
        out += [{"title": r["page"], "url": r["url"]} for r in rows[:3]]
    return out


def _answer_sources(
    intent: str, passages: list[Passage], results: list[dict]
) -> list[dict[str, str]]:
    ranked = _tool_sources(results) if intent == "moneymaking" else []
    merged: dict[str, str] = {}
    for source in [*ranked, *sources_of(passages)]:
        merged.setdefault(source["title"], source["url"])
    return [{"title": title, "url": url} for title, url in list(merged.items())[:MAX_SOURCES]]


async def _enforce_grounding(
    answer: str,
    messages: list[dict[str, str]],
    state: AgentState,
    ctx: RunContext,
    usage: dict[str, float] | None,
) -> tuple[str, dict[str, float] | None]:
    """Principle 1: large numbers must come from the tools, the player or the wiki."""
    sources = [m["content"] for m in state["messages"]]
    sources += [json.dumps(r["result"], ensure_ascii=False) for r in state.get("tool_results", [])]
    sources += [d["content"] for d in state.get("graded", []) if d.get("kept")]
    bad = ungrounded_numbers(answer, sources)
    if not bad:
        return answer, usage
    retry = [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                f"Your answer states figures that are not in the tool results: {bad}. Rewrite it "
                "using only figures from the tool results, quoted as given, with no new "
                "calculations. Keep the same language. Reply with the corrected answer only: "
                "do not mention this correction or apologise."
            ),
        },
    ]
    reply = await ctx.llm.chat(retry, temperature=0.0)
    usage = add_llm(usage, reply)
    answer = (reply.content or "").strip() or answer
    bad = ungrounded_numbers(answer, sources)
    return (drop_sentences_with(answer, bad) if bad else answer), usage


async def no_evidence(state: AgentState) -> dict:
    return {"answer": fixed("no_evidence", state.get("language", "es")), "sources": []}


async def refuse(state: AgentState) -> dict:
    language = detect_language(_last_user_message(state)) or "es"
    return {"answer": fixed("refusal", language), "language": language, "sources": []}


async def smalltalk(state: AgentState) -> dict:
    kind, language = classify_smalltalk(_last_user_message(state)) or ("greeting", "es")
    return {
        "answer": fixed(kind, language),
        "language": language,
        "intent": "smalltalk",
        "sources": [],
    }


async def finalize(state: AgentState) -> dict:
    """Close the turn: the answer joins the thread's memory (what a follow-up will see)."""
    return {"messages": [{"role": "assistant", "content": state.get("answer", "")}]}
