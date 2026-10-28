"""Where does an item drop? Derived from the monsters' drop tables, never written by a model.

The wiki's item pages end with a "Drop sources" table, but it is generated at view time from a
database query, so it is not in the page source this project ingests: "Which monsters drop the
dragon pickaxe?" has no answer in the item page, and the monster pages mention the item in one row
of a table of forty. This module inverts the drop tables instead: every ``{{DropsLine}}`` of every
page becomes a row (monster, item, quantity, rarity, which of the monster's tables), and an item
page gets a "Drop sources" section listing the monsters that drop it, most likely first.

All figures are copied from the wiki's own drop tables; ranking is by the parsed rarity.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import mwparserfromhell as mw

from app.ingest.dump import CachedPage
from app.ingest.render import Renderer, flatten_parser_functions, normalize_name

DROP_TEMPLATES = frozenset({"dropsline", "dropslinereward", "dropslineskill", "dropslineclue"})
MAX_SOURCES = 25  # rows listed per item; the rest is summarised
_FRACTION = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*/\s*(\d[\d,]*(?:\.\d+)?)")
_WORD_RARITY = {  # the wiki's colour words, as rough probabilities
    "always": 1.0,
    "common": 0.1,
    "uncommon": 0.02,
    "rare": 0.005,
    "very rare": 0.001,
}


@dataclass(frozen=True)
class DropRow:
    monster: str
    item: str
    quantity: str
    rarity: str
    version: str  # which of the monster's tables ("Wilderness Slayer Cave"), "" when it has one
    flags: str  # "(members only)" / "(free-to-play only)"

    @property
    def probability(self) -> float:
        """Rough chance per kill, only to order rows (``Always`` first, unknown last)."""
        text = self.rarity.replace("~", "").strip()
        match = _FRACTION.search(text)
        if match:
            numerator = float(match.group(1).replace(",", ""))
            denominator = float(match.group(2).replace(",", ""))
            return numerator / denominator if denominator else 0.0
        return _WORD_RARITY.get(text.lower(), 0.0)


def table_version(path: tuple[str, ...]) -> str:
    """Which of a monster's drop tables a heading path belongs to.

    ``Drops > Wilderness Slayer Cave > Weapons`` -> "Wilderness Slayer Cave"; a single table
    (``Drops > Tertiary``) has no version. The last heading is the kind of drop (``100%``,
    ``Runes``), never the version.
    """
    below = list(path[1:] if path and path[0].lower() == "drops" else path)
    return below[0] if len(below) >= 2 else ""


def extract_rows(page: CachedPage, renderer: Renderer) -> list[DropRow]:
    """Every drop line of a page, with the table (version) it sits in."""
    if "dropsline" not in page.wikitext.lower():
        return []
    code = mw.parse(flatten_parser_functions(page.wikitext))
    rows: list[DropRow] = []
    seen: set[DropRow] = set()
    stack: list[tuple[int, str]] = []
    for part in code.get_sections(levels=[2, 3, 4, 5, 6], include_lead=True, flat=True):
        nodes = part.nodes
        heading = nodes[0] if nodes and isinstance(nodes[0], mw.nodes.Heading) else None
        if heading is not None:
            level = int(heading.level)
            stack = [(lvl, t) for lvl, t in stack if lvl < level] + [
                (level, renderer.inline(heading.title))
            ]
        version = table_version(tuple(t for _, t in stack))
        for template in part.filter_templates(recursive=False):
            if normalize_name(str(template.name)) not in DROP_TEMPLATES:
                continue
            fields = renderer.named(template)
            item = fields.get("name") or ""
            clue = fields.get("tier") or fields.get("type")
            if not item and clue:
                item = f"Clue scroll ({clue})"
            if not item or not fields.get("rarity"):
                continue
            row = DropRow(
                monster=page.title,
                item=item,
                quantity=fields.get("quantity", ""),
                rarity=fields["rarity"],
                version=version,
                flags=fields.get("namenotes", ""),
            )
            if row not in seen:
                seen.add(row)
                rows.append(row)
    return rows


def build_index(
    pages: Iterable[CachedPage], renderer: Renderer | None = None
) -> dict[str, list[DropRow]]:
    """``{item name (lower case): rows}`` for every drop line of every page."""
    renderer = renderer or Renderer()
    index: dict[str, list[DropRow]] = defaultdict(list)
    for page in pages:
        for row in extract_rows(page, renderer):
            index[row.item.lower()].append(row)
    return dict(index)


def sources_text(item: str, rows: list[DropRow]) -> str:
    """The "Drop sources" section of an item page: most likely droppers first, capped."""
    ordered = sorted(rows, key=lambda r: (-r.probability, r.monster, r.version))
    lines = [f"Monsters that drop {item}, from their drop tables on the wiki:"]
    for row in ordered[:MAX_SOURCES]:
        name = f"{row.monster} ({row.version})" if row.version else row.monster
        if row.flags:
            name = f"{name} {row.flags}"
        quantity = f"{row.quantity}, " if row.quantity else ""
        lines.append(f"- {name}: {quantity}rarity {row.rarity}")
    if len(ordered) > MAX_SOURCES:
        lines.append(f"(and {len(ordered) - MAX_SOURCES} more drop tables)")
    return "\n".join(lines)
