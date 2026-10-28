"""Quest requirements against the player's snapshot (SPEC §7 ``quest_workflow``): no LLM.

The wiki's ``{{Quest details}}`` block is parsed at ingestion into skills (with whether they can be
boosted), quest points, combat level, required quests (the bullet tree gives a free order: a deeper
bullet is a prerequisite of a shallower one) and items to bring. ``check_quest`` compares all of it
with the snapshot and returns plain data for the model to explain: what is met, what is missing,
what the snapshot cannot tell, and a suggested order.

What the snapshot does not report is never assumed: an unreported quest or skill is ``unknown``, a
bank that was never opened means items may be there, and "partial completion of X" (which a quest
state cannot express) is flagged for the player to verify.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any

import psycopg

from app.api.schemas import ItemStack, PlayerState, QuestStatus
from app.ingest.quests import QuestItem, QuestRecord, QuestRequirement
from app.rag.titles import TitleIndex
from app.tools.items import ItemIndex

ConnectionFactory = Callable[[], AbstractAsyncContextManager[psycopg.AsyncConnection]]
COINS_ID = 995


class QuestStore:
    """The structured quest records, loaded once from Postgres."""

    def __init__(self, records: Iterable[QuestRecord]) -> None:
        self.records = {r.title: r for r in records}
        self._titles = TitleIndex(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    async def load(cls, connect: ConnectionFactory) -> QuestStore:
        async with connect() as conn:
            cursor = await conn.execute("select data from quests order by page_title")
            rows = await cursor.fetchall()
        return cls(QuestRecord.model_validate(row[0]) for row in rows)

    def resolve(self, *texts: str) -> list[QuestRecord]:
        """Quests named in the texts, longest names first (``[]`` when none is named)."""
        return [self.records[t] for t in self._titles.match(*texts) if t in self.records]


# --------------------------------------------------------------------------- requirements
def _skill_gaps(record: QuestRecord, player: PlayerState) -> dict[str, list[dict[str, Any]]]:
    met: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    unknown: list[str] = []
    for skill, need in sorted(record.skill_requirements.items()):
        have = player.level(skill)
        if have is None:
            unknown.append(skill)
        elif have >= need:
            met.append({"skill": skill, "have": have, "need": need})
        else:
            missing.append(
                {
                    "skill": skill,
                    "have": have,
                    "need": need,
                    "levels_short": need - have,
                    "boostable": record.boostable.get(skill),
                }
            )
    return {"met": met, "missing": missing, "unknown": [{"skill": s} for s in unknown]}


def _quest_state(req: QuestRequirement, status: QuestStatus | None) -> str:
    """``ok`` | ``missing`` | ``unknown`` | ``verify`` for one required quest."""
    if status is None:
        return "unknown"
    if status is QuestStatus.FINISHED:
        return "ok"
    if req.qualifier == "started":
        return "ok" if status is QuestStatus.IN_PROGRESS else "missing"
    if req.qualifier == "partial":
        return "verify"  # a quest state cannot say whether a given part is done
    return "missing"


def _quest_gaps(record: QuestRecord, player: PlayerState) -> dict[str, Any]:
    done: list[str] = []
    missing: list[dict[str, Any]] = []
    unknown: list[str] = []
    verify: list[str] = []
    for req in record.quest_requirements:  # already ordered: deepest prerequisites first
        status = player.quest_status(req.quest)
        state = _quest_state(req, status)
        entry = {
            "quest": req.quest,
            "needed": req.qualifier,
            "depth": req.depth,
            "status": status.value if status else None,
        }
        if state == "ok":
            done.append(req.quest)
        elif state == "missing":
            missing.append(entry)
        elif state == "verify":
            verify.append(req.quest)
        else:
            unknown.append(req.quest)
    return {"done": done, "missing": missing, "unknown": unknown, "verify_in_game": verify}


def _count(stacks: Iterable[ItemStack], item_id: int) -> int:
    return sum(s.qty for s in stacks if s.id == item_id)


def _item_status(item: QuestItem, player: PlayerState, index: ItemIndex | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"text": item.text, "qty": item.qty or 1}
    resolution = index.resolve(item.item) if index is not None and item.item else None
    if resolution is None or resolution.status != "exact" or resolution.item is None:
        entry["status"] = "unverified"  # free text such as "a light source"
        return entry
    item_id = resolution.item.id
    entry["item"] = resolution.item.name
    need = entry["qty"]
    in_inventory = _count(player.inventory, item_id)
    if item_id == COINS_ID and player.gp is not None:
        in_inventory = max(in_inventory, player.gp)
    bank_items = player.bank.items if player.bank else []
    bank_known = bool(player.bank and player.bank.captured_at)
    in_bank = _count(bank_items, item_id)
    if in_inventory >= need:
        entry["status"] = "in_inventory"
    elif bank_known and in_inventory + in_bank >= need:
        entry["status"] = "in_bank"
    elif bank_known:
        entry["status"] = "not_found"
    else:
        entry["status"] = "not_in_inventory_bank_unknown"
    return entry


def check_quest(
    record: QuestRecord, player: PlayerState, index: ItemIndex | None = None
) -> dict[str, Any]:
    """Everything the player still needs for ``record``, as plain data."""
    skills = _skill_gaps(record, player)
    quests = _quest_gaps(record, player)
    own_status = player.quest_status(record.title)

    numeric: dict[str, Any] = {}
    if record.quest_points is not None:
        have = player.quest_points
        numeric["quest_points"] = {"need": record.quest_points, "have": have}
    if record.combat_level is not None:
        numeric["combat_level"] = {"need": record.combat_level, "have": player.combat_level}
    numeric_missing = [
        name for name, v in numeric.items() if v["have"] is not None and v["have"] < v["need"]
    ]
    numeric_unknown = [name for name, v in numeric.items() if v["have"] is None]

    bank_known = bool(player.bank and player.bank.captured_at)
    items = [_item_status(i, player, index) for i in record.items_required if not i.optional]
    members_missing = bool(record.members and player.members is False)

    blockers = bool(skills["missing"] or quests["missing"] or numeric_missing or members_missing)
    unknowns = bool(
        skills["unknown"] or quests["unknown"] or numeric_unknown or quests["verify_in_game"]
    )
    order = [m["quest"] for m in quests["missing"]]  # deepest prerequisites first
    return {
        "quest": record.title,
        "own_status": own_status.value if own_status else None,
        "members_only": record.members,
        "members_requirement_unmet": members_missing,
        "start": record.start,
        "difficulty": record.difficulty,
        "length": record.length,
        "ready_to_start": False if blockers else (None if unknowns else True),
        "skills": skills,
        "quests": quests,
        "numeric_requirements": numeric,
        "other_requirements": record.other_requirements,
        "suggested_order": [*order, record.title] if order else [record.title],
        "items": {
            "required": items,
            "bank_known": bank_known,
            "bank_captured_at": (
                player.bank.captured_at.isoformat()
                if player.bank and player.bank.captured_at
                else None
            ),
            "note": None
            if bank_known
            else "The bank has not been opened this session: items may be there.",
        },
        "recommended": record.recommended,
    }
