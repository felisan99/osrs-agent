"""Short, factual summaries of the player and the conversation, for prompts and the router."""

from __future__ import annotations

from app.api.schemas import SKILL_NAMES, PlayerState
from app.graph.state import Message

SNIPPET_CHARS = 240


def compact_gp(value: int) -> str:
    """6,500,000 -> "6.5M" (the router and the logs do not need exact coins)."""
    for limit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= limit:
            return f"{value / limit:.1f}".rstrip("0").rstrip(".") + suffix
    return str(value)


def _clip(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def previous_turns(messages: list[Message], *, limit: int = 4) -> list[Message]:
    """The messages before the one being answered (the last message is the player's new one)."""
    return messages[:-1][-limit:]


def conversation_summary(messages: list[Message], *, limit: int = 4) -> str:
    """The last few exchanges, clipped: enough for Jev to resolve "and if I don't have it?"."""
    earlier = previous_turns(messages, limit=limit)
    if not earlier:
        return "(none: first message)"
    return "\n".join(f"{m['role']}: {_clip(m['content'])}" for m in earlier)


MEMBERS_ONLY_SKILLS = (
    "agility",
    "construction",
    "farming",
    "fletching",
    "herblore",
    "hunter",
    "slayer",
    "thieving",
)


def membership_text(player: PlayerState) -> str:
    if player.members is None:
        # Free accounts cannot train these skills: a level above 1 means the account has been a
        # member (not necessarily today), so say it is probably one instead of a bare "unknown".
        if any((player.level(skill) or 1) > 1 for skill in MEMBERS_ONLY_SKILLS):
            return "membership not reported (has trained members-only skills: probably a member)"
        return "membership unknown"
    return "members" if player.members else "free-to-play"


def player_headline(player: PlayerState | None) -> str:
    """One line for the router: account, membership, combat level, GP."""
    if player is None:
        return "(unknown: no snapshot from the plugin)"
    parts = [player.account_type.value.replace("_", " "), membership_text(player)]
    if player.combat_level is not None:
        parts.append(f"combat {player.combat_level}")
    if player.gp is not None:
        parts.append(f"{compact_gp(player.gp)} gp")
    return ", ".join(parts)


def player_summary(player: PlayerState | None, *, mentioned_in: str = "") -> str:
    """What the answer may rely on about the player, including what is NOT known.

    ``mentioned_in`` is the text of the question and of the evidence: the state of every quest
    named there is listed, so "do I have Monkey Madness I?" can be answered without sending the
    whole quest list (up to ~200 names) with every question.
    """
    if player is None:
        return (
            "UNKNOWN: the plugin sent no snapshot of the character, so nothing about their "
            "levels, quests, items or GP is known."
        )
    lines = [f"Account: {player.account_type.value.replace('_', ' ')}, {membership_text(player)}."]
    figures = []
    if player.combat_level is not None:
        figures.append(f"combat level {player.combat_level}")
    if player.total_level is not None:
        figures.append(f"total level {player.total_level:,}")
    if player.quest_points is not None:
        figures.append(f"{player.quest_points} quest points")
    figures.append(f"{player.gp:,} GP" if player.gp is not None else "GP not reported")
    line = ", ".join(figures)
    lines.append(line[0].upper() + line[1:] + ".")  # not str.capitalize(): it would turn GP into gp
    if player.skills:
        ordered = [f"{s} {player.skills[s]}" for s in SKILL_NAMES if s in player.skills]
        lines.append("Skills: " + ", ".join(ordered) + ".")
    else:
        lines.append("Skills: not reported.")
    if player.quests:
        finished = sum(1 for status in player.quests.values() if status.value == "FINISHED")
        lines.append(f"Quests: {finished} finished out of {len(player.quests)} reported.")
        haystack = mentioned_in.casefold()
        named = [
            f"{name} {status.value}"
            for name, status in player.quests.items()
            if haystack and name.casefold() in haystack
        ]
        if named:
            lines.append(
                "State of the quests named in this conversation: " + "; ".join(named) + "."
            )
    else:
        lines.append("Quests: not reported.")
    items = sum(1 for _ in player.inventory)
    bank = (
        f"bank captured {player.bank.captured_at:%Y-%m-%d %H:%M} UTC"
        if player.bank and player.bank.captured_at
        else "bank not captured (items may be in it)"
    )
    lines.append(f"Inventory: {items} stacks; {bank}.")
    return "\n".join(lines)
