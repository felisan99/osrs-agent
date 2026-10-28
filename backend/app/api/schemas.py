"""HTTP contract between the RuneLite plugin and the backend (SPEC §4, §5).

The plugin is a thin client: it sends a compact snapshot of the character on every turn and the
backend keeps nothing about the player (SPEC §5). Everything the plugin sends is untrusted input:
sizes are capped here and unknown fields are ignored (an older backend must keep working with a
newer plugin, and nothing unexpected is ever stored or logged).

``quest_points`` is an addition to the SPEC §5 snapshot: some money-making guides require a quest
point total, which cannot be derived from the quest list alone (the wiki does not give the points
per quest in a structured way). The plugin reads it from the client (a varp); when absent the
requirement is reported as unverified, never assumed.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SKILL_NAMES: tuple[str, ...] = (
    "attack",
    "defence",
    "strength",
    "hitpoints",
    "ranged",
    "prayer",
    "magic",
    "cooking",
    "woodcutting",
    "fletching",
    "fishing",
    "firemaking",
    "crafting",
    "smithing",
    "mining",
    "herblore",
    "agility",
    "thieving",
    "slayer",
    "farming",
    "runecraft",
    "hunter",
    "construction",
    "sailing",
)
MAX_LEVEL = 126  # virtual levels
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class AccountType(StrEnum):
    MAIN = "main"
    IRONMAN = "ironman"
    HARDCORE_IRONMAN = "hardcore_ironman"
    ULTIMATE_IRONMAN = "ultimate_ironman"
    GROUP_IRONMAN = "group_ironman"
    # RuneLite's AccountType has it; SPEC §5 lists no value for it (docs/verification.md §6).
    HARDCORE_GROUP_IRONMAN = "hardcore_group_ironman"
    UNKNOWN = "unknown"

    @property
    def is_ironman(self) -> bool:
        return self not in (AccountType.MAIN, AccountType.UNKNOWN)


class QuestStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    FINISHED = "FINISHED"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class ItemStack(_Model):
    id: Annotated[int, Field(ge=0, le=2_000_000)]
    qty: Annotated[int, Field(ge=0)]  # Python ints: stacks exceed int32


class Bank(_Model):
    """Readable only while the player has the bank open, so the plugin caches it with a date."""

    captured_at: datetime | None = None
    items: Annotated[list[ItemStack], Field(max_length=2_000)] = Field(default_factory=list)


class PlayerState(_Model):
    account_type: AccountType = AccountType.UNKNOWN
    members: bool | None = None  # None: unknown (the client cannot always tell)
    skills: Annotated[
        dict[str, Annotated[int, Field(ge=1, le=MAX_LEVEL)]], Field(max_length=40)
    ] = Field(default_factory=dict)
    quests: Annotated[dict[str, QuestStatus], Field(max_length=500)] = Field(default_factory=dict)
    quest_points: Annotated[int, Field(ge=0, le=5_000)] | None = None
    gp: Annotated[int, Field(ge=0)] | None = None
    inventory: Annotated[list[ItemStack], Field(max_length=100)] = Field(default_factory=list)
    bank: Bank | None = None
    captured_at: datetime | None = None

    @field_validator("account_type", mode="before")
    @classmethod
    def _tolerate_unknown_account_types(cls, value: Any) -> Any:
        # A future account type must not make the whole request fail.
        if isinstance(value, str) and value.strip().lower() not in {a.value for a in AccountType}:
            return AccountType.UNKNOWN
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("skills", mode="before")
    @classmethod
    def _known_skills_only(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        cleaned: dict[str, Any] = {}
        for name, level in value.items():
            key = str(name).strip().lower()
            if key in SKILL_NAMES:  # "overall" and anything new are not skills we reason about
                cleaned[key] = level
        return cleaned

    @field_validator("quests", mode="before")
    @classmethod
    def _quest_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            str(name).strip()[:120]: status for name, status in value.items() if str(name).strip()
        }

    # ---- derived facts (pure) --------------------------------------------------------------
    @property
    def is_ironman(self) -> bool:
        return self.account_type.is_ironman

    def level(self, skill: str) -> int | None:
        return self.skills.get(skill.strip().lower())

    @property
    def total_level(self) -> int | None:
        return sum(self.skills.values()) if len(self.skills) == len(SKILL_NAMES) else None

    @property
    def combat_level(self) -> int | None:
        """OSRS combat level (wiki formula), ``None`` when a combat skill is missing."""
        needed = ("attack", "strength", "defence", "hitpoints", "prayer", "ranged", "magic")
        if any(skill not in self.skills for skill in needed):
            return None
        s = self.skills
        base = Fraction(s["defence"] + s["hitpoints"] + s["prayer"] // 2, 4)
        melee = Fraction(13, 40) * (s["attack"] + s["strength"])
        ranged = Fraction(13, 40) * (s["ranged"] * 3 // 2)
        magic = Fraction(13, 40) * (s["magic"] * 3 // 2)
        return int(base + max(melee, ranged, magic))

    def quest_status(self, name: str) -> QuestStatus | None:
        """Status by quest name (case-insensitive); ``None`` when the plugin did not report it."""
        wanted = name.strip().casefold()
        for known, status in self.quests.items():
            if known.casefold() == wanted:
                return status
        return None


class ChatRequest(_Model):
    thread_id: str
    message: Annotated[str, Field(min_length=1)]
    player_state: PlayerState | None = None

    @field_validator("thread_id")
    @classmethod
    def _thread_id_shape(cls, value: str) -> str:
        if not THREAD_ID.match(value):
            raise ValueError("thread_id must be 8-64 characters: letters, digits, '-' or '_'")
        return value


class Source(_Model):
    title: str
    url: str


class ChatDebug(_Model):
    route_probs: dict[str, float] = Field(default_factory=dict)
    retrieval_iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)


class ChatResponse(_Model):
    answer: str
    intent: Literal[
        "quest",
        "moneymaking",
        "flipping",
        "price_lookup",
        "wiki_general",
        "out_of_domain",
        "smalltalk",
    ]
    sources: list[Source] = Field(default_factory=list)
    debug: ChatDebug | None = None  # only with DEBUG_RESPONSES=true (SPEC §4)


class ErrorBody(_Model):
    """Typed error (SPEC §4): 401 invalid_key, 402 insufficient_credits, 429 rate_limited, ..."""

    error: str
    message: str
    retry_after_s: float | None = None
