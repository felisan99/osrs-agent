"""``condense``: the player's message as a standalone English search query, plus its language.

Two things the rest of the graph needs and a raw chat message cannot give it: follow-ups resolved
("¿y si no tengo ese ítem?" -> "what if I don't have <the item named earlier>"), and Spanish turned
into the English game terms the wiki is written in (the full-text side of retrieval is
English-only).

The model is only called when it is needed: a first message that is clearly English already is its
own query. Everything else, including every follow-up, takes one small JSON call.
"""

from __future__ import annotations

import re

from langgraph.runtime import Runtime

from app.graph.parsing import parse_json_object
from app.graph.prompts import CONDENSE_SYSTEM
from app.graph.state import AgentState, RunContext
from app.graph.summary import previous_turns
from app.graph.usage import add_llm
from app.rag.titles import TitleIndex

_WORD = re.compile(r"[a-záéíóúüñ']+")
_SPANISH_MARKS = re.compile(r"[¿¡ñáéíóúü]", re.IGNORECASE)


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


_SPANISH = _words(
    "el la los las un una unos unas de del en es son que qué como cómo cuanto cuánto cuántos "
    "cuánta donde dónde para por con sin mi mis tu tus necesito puedo quiero tengo tienen hacer "
    "hago empiezo cual cuál cuales y o pero si ya muy más mejor cuando cuándo hay esto eso ese "
    "esa yo vos me te se lo le les al"
)
_ENGLISH = _words(
    "the is are what how do does can i my to of for with and or a an in on it that this which "
    "where when why should best need want have has was will would me you your"
)


# A word both languages use ("me") says nothing about which one this is.
_SPANISH_ONLY = _SPANISH - _ENGLISH
_ENGLISH_ONLY = _ENGLISH - _SPANISH


def detect_language(text: str) -> str | None:
    """``"es"`` / ``"en"`` when the text is clearly one of them, ``None`` when unsure."""
    words = _WORD.findall(text.lower())
    if not words:
        return None
    spanish = sum(w in _SPANISH_ONLY for w in words)
    english = sum(w in _ENGLISH_ONLY for w in words)
    if _SPANISH_MARKS.search(text) or (spanish >= 2 and spanish > english):
        return "es"
    if english >= 1 and spanish == 0 and not _SPANISH_MARKS.search(text):
        return "en"
    return None


_REFERENT = re.compile(r"\b(?:it|its|it's|that|this|them|those|these)\b", re.IGNORECASE)
MAX_VAGUE_WORDS = (
    6  # "what if I don't have that?" is vague; "tell me about that weapon from X" is not
)


def is_underspecified(query: str, titles: TitleIndex | None, *, first_message: bool) -> bool:
    """A first message that is only a pronoun question ("where do I get it?"): nothing in it names
    a page and there is no earlier turn to say what "it" is.

    Searching for it finds whatever the wiki says about "getting" things, and a grader that is
    occasionally too generous turns that into a confident answer about the wrong item (seen: an
    antipoison for "where do I get it?"). Asking costs one fixed message and never invents.
    """
    if not first_message or len(query.split()) > MAX_VAGUE_WORDS or not _REFERENT.search(query):
        return False
    return not (titles and titles.match(query))


def _conversation_block(state: AgentState, limit: int) -> str:
    turns = previous_turns(state["messages"], limit=limit)
    if not turns:
        return "(none)"
    return "\n".join(f"{m['role']}: {' '.join(m['content'].split())[:600]}" for m in turns)


async def condense(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    messages = state["messages"]
    last = messages[-1]["content"].strip()
    guess = detect_language(last)
    follow_up = len(messages) > 1
    if not follow_up and guess == "en":
        return {
            "query": last,
            "language": "en",
            "underspecified": is_underspecified(last, ctx.deps.titles, first_message=True),
        }

    earlier = _conversation_block(state, ctx.settings.history_max_messages)
    prompt = [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation so far:\n{earlier}\n\nLast message: {last}",
        },
    ]
    result = await ctx.llm.chat(
        prompt, response_format={"type": "json_object"}, max_tokens=300, temperature=0.0
    )
    data = parse_json_object(result.content) or {}
    query = str(data.get("query") or "").strip() or last
    language = data.get("language")
    if language not in ("es", "en"):
        language = guess or "es"  # unknown languages get the default (Spanish)
    return {
        "query": query,
        "language": language,
        "underspecified": is_underspecified(query, ctx.deps.titles, first_message=not follow_up),
        "usage": add_llm(state.get("usage"), result),
    }
