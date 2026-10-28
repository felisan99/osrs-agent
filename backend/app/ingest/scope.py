"""What gets ingested (SPEC §9): categories of the OSRS Wiki, in priority order.

A page can be in several categories (a boss is also a monster); the first entry that lists it
decides its ``page_type``. Category sizes were measured on 2026-10-14.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeEntry:
    category: str
    page_type: str  # quest | mmg | skill_guide | item | npc | other
    namespace: int = 0

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() else "-" for c in self.category.lower()).strip("-")


_CORE: tuple[ScopeEntry, ...] = (
    ScopeEntry("Quests", "quest"),  # 228 pages
    ScopeEntry("Miniquests", "quest"),  # 35
    ScopeEntry("Quick guides", "quest"),  # 208: condensed step-by-step pages (X/Quick guide)
    ScopeEntry("Money making guides", "mmg"),  # 649
    ScopeEntry("Training guides", "skill_guide"),  # 93
    ScopeEntry("Guides", "skill_guide"),  # 98
    ScopeEntry("Skills", "skill_guide"),  # 35
    ScopeEntry("Bosses", "npc"),  # 173
    ScopeEntry("Slayer monsters", "npc"),  # 686
    ScopeEntry("Mechanics", "other"),  # 181
    ScopeEntry("Combat", "other"),  # 45
    ScopeEntry("Ironman Mode", "other"),  # 58
    ScopeEntry("Prayers", "other"),  # 62
    ScopeEntry("Transportation", "other"),  # 46
    ScopeEntry("Slayer", "other"),  # 182
    ScopeEntry("Minigames", "other"),  # 55
    ScopeEntry("Achievement diaries", "other"),  # 95: diary tiers, rewards, requirements
    ScopeEntry("Slayer Masters", "other"),  # 25: masters, assignment tables, task tips
    ScopeEntry("Raids", "other"),  # 5: Chambers of Xeric, Theatre of Blood, Tombs of Amascut...
    ScopeEntry("Grand Exchange", "other"),  # 6: GE rules, buying limits, guide prices
    ScopeEntry("Player-owned house", "other"),  # 45: POH rooms and mechanics
    ScopeEntry("Treasure Trails", "other"),  # 185: clue scroll mechanics and rewards
    # 113: "<Boss>/Strategies" pages: recommended gear, inventories, mechanics, phase by phase.
    # Their own category, not "Bosses": the main boss pages only hold the stats and drop tables.
    ScopeEntry("Strategies", "npc"),
)
_BULK: tuple[ScopeEntry, ...] = (
    ScopeEntry("Monsters", "npc"),  # 1,636
    ScopeEntry("Spells", "other"),  # 253
    ScopeEntry("Locations", "other"),  # 1,181
    ScopeEntry("Items", "item"),  # 12,498
)

SCOPES: dict[str, tuple[ScopeEntry, ...]] = {
    # Everything the v0 agent answers from (~17k pages).
    "v0": _CORE + _BULK,
    # Quests, guides and mechanics without the bulk item/monster/location pages (~2.7k pages).
    "core": _CORE,
    # A tiny slice of each page type for fast development loops.
    "sample": (
        ScopeEntry("Miniquests", "quest"),
        ScopeEntry("Money making guides", "mmg"),
        ScopeEntry("Training guides", "skill_guide"),
        ScopeEntry("Bosses", "npc"),
        ScopeEntry("Mechanics", "other"),
        ScopeEntry("Potions", "item"),
    ),
}
