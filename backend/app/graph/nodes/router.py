"""``classify_intent`` (SPEC §7): Jev Choice over the intents, with conservative fallbacks.

* Below ``ROUTE_MIN_CONFIDENCE`` the question goes to ``wiki_general`` (SPEC §7).
* ``out_of_domain`` needs real conviction (``OUT_OF_DOMAIN_MIN_PROB``): a false rejection turns a
  legitimate question away, whereas a false "in domain" only ends in a "no evidence" answer. Live,
  a Spanish Dragon Slayer II question got out_of_domain=0.35.
"""

from __future__ import annotations

from langgraph.runtime import Runtime

from app.clients.jev import ChoiceAnswer
from app.config import Settings
from app.graph.intents import INTENT_CRITERIA, ROUTER_INSTRUCTIONS, Intent
from app.graph.state import TURN_RESET, AgentState, RunContext
from app.graph.summary import conversation_summary, player_headline
from app.graph.usage import add_jev


async def start_turn(state: AgentState) -> dict:
    """Clear the previous turn's working data: the thread's checkpoint carries it over."""
    return dict(TURN_RESET)


def decide_intent(answer: ChoiceAnswer, settings: Settings) -> Intent:
    try:
        intent = Intent(answer.choice)
    except ValueError:  # a label the model invented: safest is the general path
        return Intent.WIKI_GENERAL
    probabilities = answer.probabilities
    if intent is Intent.OUT_OF_DOMAIN:
        sure = probabilities.get(Intent.OUT_OF_DOMAIN.value, 0.0) >= settings.out_of_domain_min_prob
        return Intent.OUT_OF_DOMAIN if sure else Intent.WIKI_GENERAL
    confidence = answer.confidence
    if confidence is None:
        confidence = probabilities.get(intent.value, 1.0)
    return intent if confidence >= settings.route_min_confidence else Intent.WIKI_GENERAL


async def classify_intent(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    messages = state["messages"]
    jev_state = {
        "user_message": messages[-1]["content"],
        # Follow-ups are condensed BEFORE routing, so "¿y cuál es su límite de compra?" reaches
        # Jev together with what it means ("what is the buy limit of the abyssal whip?").
        "standalone_question": state.get("query") or "(same as user_message)",
        "conversation_summary": conversation_summary(messages),
        "player_summary": player_headline(ctx.player),
    }
    answer, decision = await ctx.jev.choose(jev_state, ROUTER_INSTRUCTIONS, INTENT_CRITERIA)
    intent = decide_intent(answer, ctx.settings)
    return {
        "intent": intent.value,
        "routed": True,
        "route_probs": {k: round(v, 4) for k, v in answer.probabilities.items()},
        "usage": add_jev(state.get("usage"), decision.usage.cost),
    }
