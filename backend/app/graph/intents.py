"""Intent taxonomy for the router (SPEC §7, ``classify_intent``).

Single source of truth shared by the router node and the scripts/eval. Jev's Choice primitive
follows the option *names* as much as their descriptions, so names are self-explanatory and the
descriptions state boundaries explicitly.
"""

from __future__ import annotations

from enum import StrEnum


class Intent(StrEnum):
    QUEST = "quest"
    MONEYMAKING = "moneymaking"
    FLIPPING = "flipping"
    PRICE_LOOKUP = "price_lookup"
    WIKI_GENERAL = "wiki_general"
    OUT_OF_DOMAIN = "out_of_domain"


INTENT_CRITERIA: dict[str, str] = {
    Intent.QUEST: (
        "The user asks about a specific Old School RuneScape quest or miniquest: how to start it, "
        "its requirements, steps/walkthrough, items needed, rewards, or what they are missing."
    ),
    Intent.MONEYMAKING: (
        "The user asks how to make money / GP in Old School RuneScape: which money-making method "
        "or activity to do, GP per hour, or what profitable activity suits their stats and quests."
    ),
    Intent.FLIPPING: (
        "The user asks about Grand Exchange flipping or merching: price margins, which items to "
        "flip, how much profit a buy-low sell-high trade would make, or the exact tax and net "
        "proceeds of ONE specific sale at a price the player gives. Explaining how the tax or the "
        "Grand Exchange works in general is NOT this: that is a wiki question."
    ),
    Intent.PRICE_LOOKUP: (
        "The user only wants the current or historical Grand Exchange price (or buy limit / "
        "alchemy value) of one or more specific items, with no request to plan a profit."
    ),
    Intent.WIKI_GENERAL: (
        "Any other Old School RuneScape knowledge question: game mechanics (including how the "
        "Grand Exchange, its tax and its buy limits work in general), skill training, bosses, "
        "monsters, items, NPCs, locations, drop tables, combat, achievement diaries."
    ),
    Intent.OUT_OF_DOMAIN: (
        "Not about Old School RuneScape at all: other games, real-world topics, programming, "
        "general chit-chat, or requests to automate or bot the game."
    ),
}

ROUTER_INSTRUCTIONS = (
    "Classify what the player is asking for in `user_message`, a message sent to an Old School "
    "RuneScape (OSRS) assistant. `standalone_question` is the same message rewritten so it makes "
    "sense without the conversation: classify what it asks. Use `conversation_summary` and "
    "`player_summary` only as context to resolve what the message refers to. Pick the single "
    "best intent."
)
