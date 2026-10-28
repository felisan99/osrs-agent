"""Prompts and fixed messages (SPEC §7, "Reglas para los prompts del LLM").

Prompts are written in English (the model follows English instructions best); replies are in the
player's language, chosen by ``language``. Fixed messages (refusal, no evidence, ...) are templates,
not model output: they must not vary, cost nothing and never invent anything.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from app.rag.retrieve import Passage

LANGUAGE_NAMES = {"es": "Spanish (neutral, 'vos' is fine)", "en": "English"}
MAX_TOOL_JSON_CHARS = 14_000
MAX_PASSAGE_CHARS = 2_400

RULES = """\
You are the in-game assistant of an Old School RuneScape (OSRS, not RuneScape 3) player.

Rules, in order of importance:
1. Facts come ONLY from the wiki passages and tool results given below. If they do not contain \
what the question needs, say plainly what is missing; never fill a gap from memory, and never \
invent numbers, requirements, drop rates, locations or quest steps.
2. Never state a price, margin, GP/hour or profit that did not come from a tool result, and do \
no arithmetic on them yourself: quote the tool's figures. Recurring activities are valued per \
run, not per hour: never present them as GP/hour.
3. Use the player's data when it matters (levels, quests, GP, items), and say so when a datum \
you need is missing from the player summary; never assume it.
4. Answer in {language}. Keep item, quest, NPC and place names in English exactly as in the game.
5. Cite the wiki pages you used by their NAME in square brackets, for example [Abyssal whip] \
(never by passage number); do not cite pages that are not in the context.
6. This assistant is informational only: never suggest or describe bots, macros, auto-clickers \
or any automation, and do not give real-time combat instructions (attack-by-attack timings, \
prayer flicking, where to stand); general strategy and requirements are fine.
7. Be concise and direct: short paragraphs or flat lists, no headings, no tables, no preamble.\
"""


def system_prompt(language: str, extra: str = "") -> str:
    base = RULES.format(language=LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["es"]))
    return f"{base}\n\n{extra}" if extra else base


def format_passages(passages: Sequence[Passage]) -> str:
    if not passages:
        return "(no wiki passages)"
    blocks = []
    for number, p in enumerate(passages, 1):
        where = p.page_title + (f" > {p.section}" if p.section else "")
        body = p.content if len(p.content) <= MAX_PASSAGE_CHARS else p.content[:MAX_PASSAGE_CHARS]
        blocks.append(f"[{number}] {where} ({p.url})\n{body}")
    return "\n\n".join(blocks)


def format_tool_results(results: Sequence[dict[str, Any]]) -> str:
    if not results:
        return "(none)"
    blocks = []
    for entry in results:
        text = json.dumps(entry["result"], ensure_ascii=False, separators=(",", ":"))
        if len(text) > MAX_TOOL_JSON_CHARS:
            text = text[:MAX_TOOL_JSON_CHARS] + "...(truncated)"
        blocks.append(f"{entry['tool']}: {text}")
    return "\n\n".join(blocks)


def generation_context(
    *, player: str, passages: Sequence[Passage], tool_results: Sequence[dict[str, Any]]
) -> str:
    return (
        f"PLAYER\n{player}\n\n"
        f"TOOL RESULTS (deterministic code; the only source for prices, profits and requirement "
        f"checks)\n{format_tool_results(tool_results)}\n\n"
        f"WIKI PASSAGES\n{format_passages(passages)}"
    )


# --------------------------------------------------------------------------- query steps
CONDENSE_SYSTEM = """\
You prepare a player's chat message for a search over the Old School RuneScape wiki.

Write "query": ONE standalone English search query for the player's LAST message.
- Resolve references to the earlier conversation ("it", "that item", "and if I don't have it?") so \
the query is understandable on its own.
- Translate to English using the game's own terms; keep proper names exactly (items, quests, NPCs, \
places) in their English game spelling.
- Keep it short (at most 25 words) and drop filler.
Write "language": the language of the LAST message, "es", "en" or "other".

Return JSON only: {"query": "...", "language": "es|en|other"}\
"""

REWRITE_SYSTEM = """\
A search over the Old School RuneScape wiki did not find enough information to answer a question.
Write ONE different English search query more likely to hit the right wiki page: use exact \
page-style names (for example "Dragon bones", "Monkey Madness II", "Zulrah"), game terms, and no \
filler words. Do not repeat a query that already failed.

Return JSON only: {"query": "..."}\
"""

# --------------------------------------------------------------------------- fixed messages
FIXED: dict[str, dict[str, str]] = {
    "no_evidence": {
        "es": (
            "No encontré información confiable sobre eso en el Wiki de OSRS, así que prefiero no "
            "inventar una respuesta. ¿Podés darme más detalle (por ejemplo el nombre exacto del "
            "ítem, la quest o el jefe) o reformular la pregunta?"
        ),
        "en": (
            "I couldn't find reliable information about that in the OSRS Wiki, so I'd rather not "
            "guess. Could you give me more detail (for example the exact name of the item, quest "
            "or boss) or rephrase the question?"
        ),
    },
    "refusal": {
        "es": (
            "Eso queda fuera de lo que puedo hacer: soy un asistente informativo de Old School "
            "RuneScape (quests, precios, flips, formas de ganar GP y mecánicas del juego). No "
            "ayudo con otros temas ni con bots o automatización del juego. Si tenés una pregunta "
            "sobre OSRS, contame."
        ),
        "en": (
            "That is outside what I can do: I'm an informational assistant for Old School "
            "RuneScape (quests, prices, flips, ways to make GP and game mechanics). I don't help "
            "with other topics or with bots or automating the game. If you have an OSRS "
            "question, ask away."
        ),
    },
    "greeting": {
        "es": (
            "¡Hola! Soy un asistente de Old School RuneScape. Puedo ayudarte con quests (qué te "
            "falta y cómo hacerlas), formas de ganar GP con precios del momento, flipping en el "
            "Grand Exchange y dudas sobre el juego (jefes, drops, habilidades). ¿Qué querés saber?"
        ),
        "en": (
            "Hi! I'm an Old School RuneScape assistant. I can help with quests (what you're "
            "missing and how to do them), ways to make GP with live prices, Grand Exchange "
            "flipping and game questions (bosses, drops, skills). What would you like to know?"
        ),
    },
    "thanks": {
        "es": "¡De nada! Si tenés otra pregunta sobre OSRS, acá estoy.",
        "en": "You're welcome! If you have another OSRS question, I'm here.",
    },
    "bye": {
        "es": "¡Hasta luego! Suerte en Gielinor.",
        "en": "See you! Good luck out in Gielinor.",
    },
    "flip_ironman": {
        "es": (
            "Las cuentas ironman no pueden usar el Grand Exchange, así que el flipping no aplica "
            "a tu cuenta. Si querés, te cuento cómo conseguir GP con una cuenta ironman (sin "
            "precios del GE)."
        ),
        "en": (
            "Ironman accounts can't use the Grand Exchange, so flipping doesn't apply to your "
            "account. If you like, I can tell you how to get GP on an ironman (without Grand "
            "Exchange prices)."
        ),
    },
    "prices_footer": {
        "es": "_Precios de la API de precios del Wiki de OSRS, obtenidos {when} UTC._",
        "en": "_Prices from the OSRS Wiki prices API, fetched {when} UTC._",
    },
    "tool_failed": {
        "es": (
            "No pude obtener los precios en este momento (la API de precios del Wiki no "
            "respondió), y no quiero darte números sin fuente. Probá de nuevo en un rato."
        ),
        "en": (
            "I couldn't get prices right now (the Wiki prices API didn't answer), and I won't "
            "give you numbers without a source. Please try again in a moment."
        ),
    },
}


def fixed(kind: str, language: str, **fields: str) -> str:
    table = FIXED[kind]
    return table.get(language, table["es"]).format(**fields)
