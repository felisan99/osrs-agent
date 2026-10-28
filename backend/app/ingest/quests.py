"""Structured quest requirements from ``{{Quest details}}`` (SPEC §7 ``quest_workflow``).

The quest workflow compares requirements against the player's state deterministically, so the
requirements are extracted into data instead of being left as prose for the LLM to read:

* skill levels (``{{SCP|Magic|75}}``) with whether the level can be boosted (``{{Boostable}}``);
* quest points / combat level;
* required quests, as the wiki's nested bullet tree: a deeper bullet is a prerequisite of a
  shallower one, so ``depth`` gives a free suggested order (deepest first);
* items to bring, with quantities where the line states them.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Literal

import mwparserfromhell as mw
from mwparserfromhell.nodes import Template
from mwparserfromhell.wikicode import Wikicode
from pydantic import BaseModel, Field

from app.ingest.mmg import NOT_QUESTS, SKILLS
from app.ingest.render import Renderer, flatten_parser_functions, normalize_name

QuestQualifier = Literal["completed", "started", "partial"]


class QuestRequirement(BaseModel):
    quest: str
    qualifier: QuestQualifier = "completed"
    depth: int = 1  # 1 = listed directly; deeper = a prerequisite of a prerequisite


class QuestItem(BaseModel):
    text: str
    item: str | None = None  # first linked item, singular as linked
    qty: int | None = None
    optional: bool = False


class QuestRecord(BaseModel):
    title: str
    members: bool | None = None
    difficulty: str | None = None
    length: str | None = None
    start: str | None = None
    skill_requirements: dict[str, int] = Field(default_factory=dict)
    boostable: dict[str, bool] = Field(default_factory=dict)
    quest_points: int | None = None
    combat_level: int | None = None
    quest_requirements: list[QuestRequirement] = Field(default_factory=list)
    other_requirements: list[str] = Field(default_factory=list)
    items_required: list[QuestItem] = Field(default_factory=list)
    recommended: list[str] = Field(default_factory=list)
    kills: list[str] = Field(default_factory=list)


_BULLET = re.compile(r"^(\*+)\s*(.*)$")
_QUALIFIER = re.compile(r"\b(started|partial(?:ly)?|portion|completion of|complete)\b", re.I)
_OPTIONAL = re.compile(
    r"optional|can be obtained|\bif (?:you|obtaining)|\(recommended\)|alternatively", re.I
)
_QTY = re.compile(r"(?:at least\s+|about\s+|\b)(\d+)\s*(?:x\b|\[\[|[A-Za-z])", re.I)
_YES = {"yes", "y", "true", "1"}


def _link_title(link) -> str:
    return str(link.title).split("#", 1)[0].replace("_", " ").strip()


def _qualifier(text: str) -> QuestQualifier:
    if re.search(r"\bportion\b|\bpartial", text, re.I):  # "Started the pyre ship portion of ..."
        return "partial"
    if re.search(r"\bstarted\b", text, re.I):
        return "started"
    return "completed"


def _params(template: Template) -> dict[str, Wikicode]:
    return {str(p.name).strip().lower(): p.value for p in template.params}


def _find(code: Wikicode, *names: str) -> Template | None:
    for template in code.filter_templates(recursive=True):
        if normalize_name(str(template.name)) in names:
            return template
    return None


def _bullets(code: Wikicode | None) -> list[tuple[int, str]]:
    """``(depth, wikitext)`` for each bullet line; non-bullet lines count as depth 0."""
    if code is None:
        return []
    out: list[tuple[int, str]] = []
    for raw in str(code).split("\n"):
        line = raw.strip()
        if not line:
            continue
        match = _BULLET.match(line)
        out.append((len(match.group(1)), match.group(2)) if match else (0, line))
    return out


def _lines(renderer: Renderer, code: Wikicode | None) -> list[str]:
    text = renderer.render(code) if code is not None else ""
    return [re.sub(r"^\s*-\s*", "", line).strip() for line in text.split("\n") if line.strip()]


def _parse_requirements(
    record: QuestRecord, code: Wikicode | None, renderer: Renderer, quest_titles: Collection[str]
) -> None:
    known = set(quest_titles)
    by_name: dict[str, QuestRequirement] = {}
    for depth, wikitext in _bullets(code):
        parsed = mw.parse(wikitext)
        text = renderer.inline(parsed)
        if not text or text.lower().strip(": ") in {"none", "n/a"}:
            continue
        lowered = text.lower()
        scp = [
            t
            for t in parsed.filter_templates(recursive=True)
            if normalize_name(str(t.name)) == "scp"
        ]
        booster = next(
            (
                t
                for t in parsed.filter_templates(recursive=True)
                if normalize_name(str(t.name)) == "boostable"
            ),
            None,
        )
        handled = False
        for template in scp:
            args = [renderer.inline(p.value) for p in template.params if not p.showkey]
            level_match = re.fullmatch(r"(\d{1,4})\+?", args[1].strip()) if len(args) > 1 else None
            if level_match is None:
                continue
            skill, level = args[0].strip().lower(), int(level_match.group(1))
            if skill in SKILLS:
                name = skill.capitalize()
                record.skill_requirements[name] = max(record.skill_requirements.get(name, 0), level)
                if booster is not None:
                    flag = (
                        renderer.inline(booster.params[0].value).lower() if booster.params else ""
                    )
                    record.boostable[name] = flag in _YES
                handled = True
            elif skill == "quest":
                record.quest_points = max(record.quest_points or 0, level)
                handled = True
            elif skill == "combat":
                record.combat_level = max(record.combat_level or 0, level)
                handled = True
        if handled:
            continue

        links = [_link_title(link) for link in parsed.filter_wikilinks()]
        quest_links = [
            title
            for title in links
            if title
            and title.lower() not in NOT_QUESTS
            and (title in known or (not known and depth >= 2))
        ]
        is_header = lowered.rstrip(":").startswith(
            ("completion of the following", "completion of all")
        )
        if is_header:
            continue
        pure_link = (
            len(quest_links) == 1
            and len(re.sub(r"\([^)]*\)", "", text).strip()) <= len(quest_links[0]) + 12
        )
        mentions_quest = bool(quest_links) and (
            pure_link or _QUALIFIER.search(text) is not None or depth >= 2
        )
        if mentions_quest and len(quest_links) <= 2:
            for title in quest_links:
                qualifier = _qualifier(text)
                existing = by_name.get(title)
                if existing is None:
                    by_name[title] = QuestRequirement(
                        quest=title, qualifier=qualifier, depth=max(depth, 1)
                    )
                else:
                    existing.depth = max(existing.depth, depth)
                    if qualifier == "completed":
                        existing.qualifier = "completed"
            continue
        record.other_requirements.append(text)
    record.quest_requirements = sorted(by_name.values(), key=lambda r: (-r.depth, r.quest))


def _parse_items(code: Wikicode | None, renderer: Renderer) -> list[QuestItem]:
    items: list[QuestItem] = []
    for depth, wikitext in _bullets(code):
        parsed = mw.parse(wikitext)
        text = renderer.inline(parsed)
        if not text or text.lower() in {"none", "n/a"}:
            continue
        links = [_link_title(link) for link in parsed.filter_wikilinks()]
        qty_match = _QTY.search(re.sub(r"'''?", "", wikitext))
        items.append(
            QuestItem(
                text=text,
                item=links[0] if links else None,
                qty=int(qty_match.group(1)) if qty_match else None,
                optional=bool(_OPTIONAL.search(text)) or depth >= 2,
            )
        )
    return items


def parse_quest(
    title: str, wikitext: str, *, quest_titles: Collection[str] = ()
) -> QuestRecord | None:
    """Parse one quest page; ``None`` if it has no ``{{Quest details}}`` block."""
    code = mw.parse(flatten_parser_functions(wikitext))
    details = _find(code, "quest details")
    if details is None:
        return None
    renderer = Renderer()
    params = _params(details)

    def text(key: str) -> str | None:
        value = params.get(key)
        rendered = renderer.inline(value) if value is not None else ""
        return rendered or None

    record = QuestRecord(
        title=title,
        difficulty=text("difficulty"),
        length=text("length"),
        start=text("start"),
    )
    infobox = _find(code, "infobox quest")
    if infobox is not None:
        members = _params(infobox).get("members")
        flag = renderer.inline(members).lower() if members is not None else ""
        record.members = (
            True if flag in _YES else False if flag in {"no", "n", "false", "0"} else None
        )

    _parse_requirements(record, params.get("requirements"), renderer, quest_titles)
    record.items_required = _parse_items(params.get("items"), renderer)
    record.recommended = _lines(renderer, params.get("recommended"))
    record.kills = _lines(renderer, params.get("kills"))
    return record
