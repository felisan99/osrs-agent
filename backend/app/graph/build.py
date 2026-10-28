"""Graph construction (SPEC §3 ``graph/build.py``).

    first message:  start_turn -> classify_intent -+-> refuse (out_of_domain) ------------+
                                                   +-> condense -+                         |
    follow-up:      start_turn -> condense -> classify_intent ---+  (routed by intent)    |
                                                                 v                        |
        quest_prepare / moneymaking_prepare / (wiki_general) -> retrieve -> grade -+      |
        flipping / price_lookup (tool loops) -------------------------------+     |      |
                                                                             |     v      v
        grade -+-> generate -----------------------------------------------> finalize -> END
               +-> no_evidence -----------------------------------------------> ^
               +-> correct (<= MAX_CRAG_ITERATIONS) -> grade

A first message is routed before it is condensed, so an out-of-domain one costs no model call; a
follow-up is condensed first (its condensing is needed anyway) so the router sees what it means.
A first message that is only a pronoun question ("where do I get it?") goes from condense straight
to ``no_evidence``, which asks for detail, without searching.

The checkpointer gives every ``thread_id`` its conversation memory. The user's key and the
player's snapshot reach nodes through the run-scoped ``RunContext``, never through the state or
the config, so neither is ever written to a checkpoint.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.graph.intents import Intent
from app.graph.nodes.condense import condense
from app.graph.nodes.crag import (
    correct,
    finalize,
    generate,
    grade,
    make_after_grade,
    no_evidence,
    refuse,
    retrieve,
    smalltalk,
)
from app.graph.nodes.router import classify_intent, start_turn
from app.graph.nodes.tools_loop import flipping_workflow, price_workflow
from app.graph.nodes.workflows import moneymaking_prepare, quest_prepare
from app.graph.smalltalk import classify_smalltalk
from app.graph.state import AgentState, RunContext

ENTRY_BY_INTENT = {
    Intent.QUEST.value: "quest_prepare",
    Intent.MONEYMAKING.value: "moneymaking_prepare",
    Intent.FLIPPING.value: "flipping",
    Intent.PRICE_LOOKUP.value: "price_lookup",
}


def route_start(state: AgentState) -> str:
    """A follow-up is condensed before it is routed (the model call is needed anyway); a first
    message is routed first, so an out-of-domain one costs no model call at all. A bare greeting
    or thanks is answered at once."""
    if classify_smalltalk(state["messages"][-1]["content"]):
        return "smalltalk"
    return "condense" if len(state["messages"]) > 1 else "classify_intent"


def route_by_intent(state: AgentState) -> str:
    if state.get("underspecified"):
        return "no_evidence"  # "where do I get it?": ask what "it" is instead of searching
    return ENTRY_BY_INTENT.get(state.get("intent", ""), "retrieve")


def route_after_classify(state: AgentState) -> str:
    if state.get("intent") == Intent.OUT_OF_DOMAIN.value:
        return "refuse"
    return route_by_intent(state) if state.get("query") else "condense"


def route_after_condense(state: AgentState) -> str:
    return route_by_intent(state) if state.get("routed") else "classify_intent"


def build_graph(settings: Settings, checkpointer: BaseCheckpointSaver | None = None) -> Any:
    graph = StateGraph(AgentState, context_schema=RunContext)
    nodes = {
        "start_turn": start_turn,
        "classify_intent": classify_intent,
        "refuse": refuse,
        "smalltalk": smalltalk,
        "condense": condense,
        "quest_prepare": quest_prepare,
        "moneymaking_prepare": moneymaking_prepare,
        "flipping": flipping_workflow,
        "price_lookup": price_workflow,
        "retrieve": retrieve,
        "grade": grade,
        "correct": correct,
        "generate": generate,
        "no_evidence": no_evidence,
        "finalize": finalize,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "start_turn")
    graph.add_conditional_edges(
        "start_turn",
        route_start,
        {"condense": "condense", "classify_intent": "classify_intent", "smalltalk": "smalltalk"},
    )
    targets = {
        "quest_prepare": "quest_prepare",
        "moneymaking_prepare": "moneymaking_prepare",
        "flipping": "flipping",
        "price_lookup": "price_lookup",
        "retrieve": "retrieve",
        "no_evidence": "no_evidence",
    }
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"refuse": "refuse", "condense": "condense", **targets},
    )
    graph.add_conditional_edges(
        "condense", route_after_condense, {"classify_intent": "classify_intent", **targets}
    )
    graph.add_edge("quest_prepare", "retrieve")
    graph.add_edge("moneymaking_prepare", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        make_after_grade(settings.max_crag_iterations),
        {"generate": "generate", "correct": "correct", "no_evidence": "no_evidence"},
    )
    graph.add_edge("correct", "grade")
    for terminal in ("refuse", "smalltalk", "flipping", "price_lookup", "generate", "no_evidence"):
        graph.add_edge(terminal, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
