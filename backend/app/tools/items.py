"""Item name -> id resolution over the prices API ``/mapping`` (SPEC §6).

Resolution order (first hit wins):

1. exact, case-insensitive name match;
2. well-known community nickname ("tbow", "bgs", ...) -> its full item name;
3. a dose family named without a dose ("saradomin brew") -> the full-dose variant ``(4)``;
4. fuzzy match with rapidfuzz, accepted only when clearly better than the runner-up;
5. otherwise ``ambiguous`` (candidates returned for the caller to choose from) or ``not_found``
   (closest names returned as suggestions).

Notes:
* ``/mapping`` does not contain Coins or Platinum token; they are added as special items with
  their face value so inventories/guides that mention them can still be valued.
* Thresholds were tuned by hand against the live ``/mapping`` (see tests/test_items.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from app.clients.prices import ItemMeta

# Items missing from /mapping that guides and inventories still mention. Values are face values.
SPECIAL_ITEMS: tuple[ItemMeta, ...] = (
    ItemMeta(id=995, name="Coins", members=False, value=1, examine="Lovely money!"),
    ItemMeta(
        id=13204,
        name="Platinum token",
        members=False,
        value=1000,
        examine="A token worth 1,000 coins.",
    ),
)
# Not tradeable on the GE, but worth exactly this many coins each.
FACE_VALUE: dict[int, int] = {item.id: item.value or 0 for item in SPECIAL_ITEMS}

# Nicknames players actually type (all targets verified to exist in /mapping on 2026-10-14;
# unknown targets are skipped at index build time, so a renamed item never breaks resolution).
ITEM_ALIASES: dict[str, str] = {
    "tbow": "Twisted bow",
    "bgs": "Bandos godsword",
    "ags": "Armadyl godsword",
    "sgs": "Saradomin godsword",
    "zgs": "Zamorak godsword",
    "dds": "Dragon dagger",
    "dwh": "Dragon warhammer",
    "dclaws": "Dragon claws",
    "bcp": "Bandos chestplate",
    "tassets": "Bandos tassets",
    "tassy": "Bandos tassets",
    "acb": "Armadyl crossbow",
    "dcb": "Dragon crossbow",
    "rcb": "Rune crossbow",
    "dhcb": "Dragon hunter crossbow",
    "dhl": "Dragon hunter lance",
    "lance": "Dragon hunter lance",
    "fury": "Amulet of fury",
    "torture": "Amulet of torture",
    "anguish": "Necklace of anguish",
    "shadow": "Tumeken's shadow (uncharged)",
    "sang": "Sanguinesti staff (uncharged)",
    "ely": "Elysian spirit shield",
    "arcane": "Arcane spirit shield",
    "dfs": "Dragonfire shield",
    "rapier": "Ghrazi rapier",
    "bond": "Old school bond",
    "bonds": "Old school bond",
    "brews": "Saradomin brew(4)",
    "restores": "Super restore(4)",
    "ppots": "Prayer potion(4)",
    "dbones": "Dragon bones",
    "nats": "Nature rune",
    "sharks": "Shark",
    "karambwans": "Cooked karambwan",
}

CANDIDATE_MIN_SCORE = 60.0  # below this a match is not even suggested
SUGGEST_MIN_SCORE = 80.0  # best match below this -> not_found (suggestions only)
CONFIDENT_SCORE = 92.0  # accept the top fuzzy match on its own above this (with a margin)
CONFIDENT_MARGIN = 4.0
CLEAR_WINNER_SCORE = 85.0  # ...or above this when the runner-up is far behind
CLEAR_WINNER_MARGIN = 8.0
PREFIX_BONUS = 4.0  # "super combat" prefers "Super combat potion(4)" over "Divine super combat..."
FAMILY_BAND = 3.0  # dose variants within this many points of the best score form one family
AMBIGUOUS_BAND = 15.0  # alternatives shown only if within this many points of the best match
MAX_CANDIDATES = 5

_DOSE = re.compile(r"^(?P<base>.+?)\s*\((?P<dose>\d+)\)$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").casefold()).strip()


def _fuzzy_key(text: str) -> str:
    """Lowercase, punctuation -> spaces, but keep ``+`` so ``p++`` and ``p+`` stay different."""
    return re.sub(r"[^\w+]+", " ", text.replace("’", "'").casefold()).strip()


class Candidate(BaseModel):
    id: int
    name: str
    members: bool
    score: float


class Resolution(BaseModel):
    status: Literal["exact", "fuzzy", "ambiguous", "not_found"]
    query: str
    item: ItemMeta | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    note: str | None = None

    @property
    def resolved(self) -> bool:
        return self.item is not None


class ItemIndex:
    def __init__(self, items: Iterable[ItemMeta]) -> None:
        by_id = {item.id: item for item in items}
        for special in SPECIAL_ITEMS:
            by_id.setdefault(special.id, special)
        self._by_id = by_id
        self._items = list(by_id.values())
        self._keys = [_fuzzy_key(item.name) for item in self._items]

        self._exact: dict[str, list[int]] = {}
        self._families: dict[str, list[tuple[int, ItemMeta]]] = {}
        for item in self._items:
            self._exact.setdefault(_norm(item.name), []).append(item.id)
            dose = _DOSE.match(item.name)
            if dose:
                family = self._families.setdefault(_norm(dose.group("base")), [])
                family.append((int(dose.group("dose")), item))

        self._aliases: dict[str, ItemMeta] = {}
        for nickname, target in ITEM_ALIASES.items():
            ids = self._exact.get(_norm(target))
            if ids and len(ids) == 1:
                self._aliases[_norm(nickname)] = by_id[ids[0]]

    def __len__(self) -> int:
        return len(self._by_id)

    def by_id(self, item_id: int) -> ItemMeta | None:
        return self._by_id.get(item_id)

    def name_of(self, item_id: int) -> str:
        item = self._by_id.get(item_id)
        return item.name if item else f"item {item_id}"

    def _candidate(self, index: int, score: float) -> Candidate:
        item = self._items[index]
        return Candidate(id=item.id, name=item.name, members=item.members, score=round(score, 1))

    def resolve(self, query: str) -> Resolution:
        text = query.strip()
        if not text:
            return Resolution(status="not_found", query=query, note="empty item name")
        norm = _norm(text)

        exact_ids = self._exact.get(norm)
        if exact_ids:
            if len(exact_ids) == 1:
                return Resolution(status="exact", query=query, item=self._by_id[exact_ids[0]])
            return Resolution(
                status="ambiguous",
                query=query,
                candidates=[
                    Candidate(
                        id=i, name=self._by_id[i].name, members=self._by_id[i].members, score=100
                    )
                    for i in exact_ids[:MAX_CANDIDATES]
                ],
                note="several items share this exact name",
            )

        nickname = self._aliases.get(norm)
        if nickname is not None:
            return Resolution(
                status="exact",
                query=query,
                item=nickname,
                note=f"'{text}' is a common nickname for '{nickname.name}'",
            )

        family = self._families.get(norm)
        if family:
            _, item = max(family, key=lambda dose_item: dose_item[0])
            return Resolution(
                status="fuzzy",
                query=query,
                item=item,
                note=f"no dose given: defaulted to the full-dose variant '{item.name}'",
            )

        return self._resolve_fuzzy(query, text)

    def _resolve_fuzzy(self, query: str, text: str) -> Resolution:
        key = _fuzzy_key(text)
        matches = process.extract(
            key,
            self._keys,
            scorer=fuzz.WRatio,
            processor=None,
            limit=12,
            score_cutoff=CANDIDATE_MIN_SCORE,
        )
        candidates: list[Candidate] = []
        for _, score, index in matches:
            candidate_key = self._keys[index]
            prefixed = candidate_key == key or candidate_key.startswith(key + " ")
            candidates.append(
                self._candidate(index, min(100.0, score + (PREFIX_BONUS if prefixed else 0.0)))
            )
        candidates.sort(key=lambda c: -c.score)  # stable: ties keep rapidfuzz's order

        if not candidates or candidates[0].score < SUGGEST_MIN_SCORE:
            return Resolution(
                status="not_found",
                query=query,
                candidates=candidates[:3],
                note="no close match; the closest names are listed" if candidates else None,
            )

        best = candidates[0]
        collapsed = self._full_dose_variant(candidates)
        if collapsed is not None:
            item = self._by_id[collapsed.id]
            return Resolution(
                status="fuzzy",
                query=query,
                item=item,
                candidates=[c for c in candidates if c.id != item.id][:MAX_CANDIDATES],
                note=f"no dose given: defaulted to the full-dose variant '{item.name}'",
            )

        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - runner_up
        accept = (
            (best.score >= 100.0 and margin > 0)  # identical once punctuation is stripped
            or (best.score >= CONFIDENT_SCORE and margin >= CONFIDENT_MARGIN)
            or (best.score >= CLEAR_WINNER_SCORE and margin >= CLEAR_WINNER_MARGIN)
        )
        # Only plausible alternatives: a weak 77% hit next to a 90% one is noise for the caller.
        close = [
            c for c in candidates if c.score >= max(SUGGEST_MIN_SCORE, best.score - AMBIGUOUS_BAND)
        ]
        if accept:
            return Resolution(
                status="fuzzy",
                query=query,
                item=self._by_id[best.id],
                candidates=close[1:MAX_CANDIDATES],
                note=f"matched '{best.name}' (score {best.score})",
            )
        return Resolution(
            status="ambiguous",
            query=query,
            candidates=close[:MAX_CANDIDATES],
            note="several items match; use one of the exact names",
        )

    @staticmethod
    def _full_dose_variant(candidates: list[Candidate]) -> Candidate | None:
        """If the best matches are all doses of one item, return its highest-dose variant."""
        top = candidates[0].score
        band = [c for c in candidates if top - c.score <= FAMILY_BAND]
        parsed = [(c, _DOSE.match(c.name)) for c in band]
        if len(band) < 2 or any(match is None for _, match in parsed):
            return None
        if len({_norm(match.group("base")) for _, match in parsed if match}) != 1:
            return None
        return max(parsed, key=lambda pair: int(pair[1].group("dose")) if pair[1] else -1)[0]


def resolution_error(resolution: Resolution) -> dict[str, Any]:
    """Tool-result payload for a name that could not be resolved to a single item."""
    candidates = [{"name": c.name, "id": c.id, "members": c.members} for c in resolution.candidates]
    return {
        "error": "item_ambiguous" if resolution.status == "ambiguous" else "item_not_found",
        "query": resolution.query,
        "message": resolution.note,
        "candidates": candidates,
        "hint": (
            "Call the tool again with one of the exact candidate names."
            if candidates
            else "Check the spelling or use the official in-game item name."
        ),
    }
