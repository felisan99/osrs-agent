"""Structural chunking (SPEC §9): sections first, token budget second.

* A chunk never mixes pages, and never mixes top-level sections: "Moss giant > Drops" and
  "Moss giant > Locations" are different chunks even when both are tiny, because what a line
  *means* depends on the section it sits in (an item in a drop table is not an item mentioned in
  the lead, e.g. as a weakness).
* Within one top-level section, sub-sections are packed greedily up to ``max_tokens``; a
  sub-section boundary ends a chunk once ``min_tokens`` is reached.
* Every chunk starts with ``Title > Section path``, where the path is the one the *whole* chunk
  shares (the parent section when several sub-sections were packed together). Each sub-section
  inside carries a ``[Relative > Path]`` label. Title and heading are embedded and indexed with
  the body, and shown to the model with it.
* Oversized blocks are split without cutting lines: lists and tables by row, paragraphs by
  sentence, and only as a last resort by word. Split paragraphs overlap by one sentence; every
  piece repeats the header.
* ``max_tokens`` defaults to 480, below the embedding model's input limit (SPEC §9 says 300-600;
  see ``Settings.chunk_target_max_tokens``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.ingest.mmg import SKILLS
from app.ingest.parse import ParsedPage

TokenCounter = Callable[[str], int]
QuestMatcher = Callable[[str], list[str]]
TAIL_SLACK = 1.15  # a tiny last chunk is merged back if the result stays under max * this


def estimate_tokens(text: str) -> int:
    """Dependency-free, deliberately conservative estimate (~3.6 chars/token for wiki prose).

    ``ceil`` per piece guarantees a sum of pieces never underestimates the joined text, which is
    what lets chunks be packed from separately counted blocks without exceeding the budget.
    """
    return math.ceil(len(text) / 3.6) if text else 0


@dataclass(frozen=True)
class ChunkConfig:
    min_tokens: int = 300
    max_tokens: int = 480
    overlap_tokens: int = 40

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkConfig:
        return cls(
            min_tokens=settings.chunk_target_min_tokens,
            max_tokens=settings.chunk_target_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )


@dataclass(frozen=True)
class Chunk:
    page_title: str
    section: str  # "A > B" or "" for the lead
    page_type: str
    url: str
    content: str  # header + body: exactly what is embedded and full-text indexed
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- metadata
_SKILL_NAMES = "|".join(sorted((s.capitalize() for s in SKILLS), key=len, reverse=True))
_SKILL_RE = re.compile(
    rf"\b(?:\d{{1,3}}\+?\s+(?:in\s+)?({_SKILL_NAMES})\b"
    rf"|({_SKILL_NAMES})\s+level\b|level\s+\d{{1,3}}\s+({_SKILL_NAMES})\b)"
)


def mentioned_skills(text: str) -> list[str]:
    """Skills named with a level ("70 Attack", "Prayer level"); wiki-style capitals only."""
    found = {next(g for g in m.groups() if g) for m in _SKILL_RE.finditer(text)}
    return sorted(found)


def build_quest_matcher(titles: Collection[str]) -> QuestMatcher:
    """A function returning the known quest titles that appear (whole-word) in a text."""
    ordered = sorted({t for t in titles if len(t) >= 4}, key=len, reverse=True)
    if not ordered:
        return lambda text: []
    pattern = re.compile(r"(?<![\w'])(?:" + "|".join(re.escape(t) for t in ordered) + r")(?![\w'])")
    return lambda text: sorted({m.group(0) for m in pattern.finditer(text)})


# --------------------------------------------------------------------------- splitting
@dataclass
class _Unit:
    path: tuple[str, ...]
    text: str
    tokens: int


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _split_words(text: str, limit: int, count: TokenCounter) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if count(" ".join(current)) >= limit:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _atoms(block: str, limit: int, count: TokenCounter) -> list[str]:
    """Smallest pieces to pack: rows/lines, then sentences, then words for runaway lines."""
    atoms: list[str] = []
    for line in block.split("\n"):
        if not line.strip():
            continue
        if count(line) <= limit:
            atoms.append(line)
            continue
        for sentence in _SENTENCE.split(line):
            if count(sentence) <= limit:
                atoms.append(sentence)
            else:
                atoms.extend(_split_words(sentence, limit, count))
    return atoms


_LIST_LABEL = re.compile(r"^[^-\s].*:$")  # "Items required:", "Recommended equipment (...):"


def _is_bullet(line: str) -> bool:
    return line.lstrip().startswith("- ")


def _is_table_header(line: str) -> bool:
    """A row of column names: several cells, fewer than half of them numbers."""
    cells = [cell.strip() for cell in line.split(" | ")]
    numeric = sum(
        cell.replace(",", "").replace(".", "").replace("%", "").isdigit() for cell in cells
    )
    return len(cells) >= 2 and numeric * 2 < len(cells)


def _continuation_context(atom: str, label: str | None, header: str | None) -> str | None:
    """What a piece that starts mid-list or mid-table needs to be understood on its own: the line
    that names the list ("Items required:") or the table's column names."""
    if _is_bullet(atom) and label:
        return label
    if " | " in atom and header and atom != header:
        return header
    return None


def _split_block(block: str, cfg: ChunkConfig, count: TokenCounter) -> list[str]:
    """Split one oversized block into pieces <= max_tokens; paragraphs overlap by a sentence.

    A list or table that continues in the next piece repeats its label line / header row there,
    so no piece is a bare run of rows.
    """
    atoms = _atoms(block, cfg.max_tokens, count)
    is_prose = "\n" not in block.strip()
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    label: str | None = None
    header: str | None = None
    for atom in atoms:
        size = count(atom)
        if current and used + size > cfg.max_tokens:
            pieces.append("\n".join(current) if not is_prose else " ".join(current))
            tail = current[-1]
            current = [tail] if is_prose and count(tail) <= cfg.overlap_tokens else []
            used = count(tail) if current else 0
            context = None if is_prose else _continuation_context(atom, label, header)
            # only when the row still fits behind it: the budget is a hard limit (embedding input)
            if context is not None and count(context) + size <= cfg.max_tokens:
                current, used = [context], count(context)
        current.append(atom)
        used += size
        if not _is_bullet(atom) and _LIST_LABEL.match(atom):
            label = atom
        elif header is None and " | " in atom and _is_table_header(atom):
            header = atom
    if current:
        pieces.append("\n".join(current) if not is_prose else " ".join(current))
    return pieces


def _blocks(text: str) -> list[str]:
    return [b.strip("\n") for b in re.split(r"\n\s*\n", text) if b.strip()]


# --------------------------------------------------------------------------- packing
@dataclass
class _Draft:
    segments: list[tuple[tuple[str, ...], str]] = field(default_factory=list)
    tokens: int = 0

    @property
    def last_path(self) -> tuple[str, ...] | None:
        return self.segments[-1][0] if self.segments else None

    def add(self, unit: _Unit, label_cost: int) -> None:
        extra = 1  # the blank line that separates blocks
        if unit.path and unit.path != self.last_path:
            extra += label_cost  # worst case: the label shows the whole path, not the relative one
        self.segments.append((unit.path, unit.text))
        self.tokens += unit.tokens + extra

    def header_path(self) -> tuple[str, ...]:
        """The section path every segment of the chunk shares (``()``: only the page title)."""
        prefix = self.segments[0][0]
        for path, _ in self.segments[1:]:
            common = 0
            while common < min(len(prefix), len(path)) and prefix[common] == path[common]:
                common += 1
            prefix = prefix[:common]
        return prefix

    def body(self, prefix: tuple[str, ...]) -> str:
        """Segments in order; each sub-section below the header path gets a relative label."""
        out: list[str] = []
        previous: tuple[str, ...] | None = None
        for path, text in self.segments:
            relative = path[len(prefix) :]
            out.append(
                f"[{' > '.join(relative)}]\n{text}" if relative and path != previous else text
            )
            previous = path
        return "\n\n".join(out)


def _pack(units: list[_Unit], cfg: ChunkConfig, count: TokenCounter) -> list[_Draft]:
    labels: dict[tuple[str, ...], int] = {}

    def label_cost(path: tuple[str, ...]) -> int:
        if path not in labels:
            labels[path] = count(f"[{' > '.join(path)}]") + 1
        return labels[path]

    drafts: list[_Draft] = []
    current = _Draft()
    for unit in units:
        last = current.last_path
        crossing = last is not None and unit.path != last
        other_section = last is not None and unit.path[:1] != last[:1]
        cost = unit.tokens + 1 + (label_cost(unit.path) if crossing and unit.path else 0)
        overflow = current.tokens + cost > cfg.max_tokens
        full = crossing and current.tokens >= cfg.min_tokens
        if current.segments and (other_section or overflow or full):
            drafts.append(current)
            current = _Draft()
        current.add(unit, label_cost(unit.path) if unit.path else 0)
    if current.segments:
        drafts.append(current)
    # a runt at the end of a section is folded into the chunk before it (same top-level section)
    if len(drafts) > 1 and drafts[-1].tokens < cfg.min_tokens // 2:
        before, tail = drafts[-2], drafts[-1]
        same_section = before.segments[-1][0][:1] == tail.segments[0][0][:1]
        if same_section and before.tokens + tail.tokens <= cfg.max_tokens * TAIL_SLACK:
            drafts.pop()
            before.segments.extend(tail.segments)
            before.tokens += tail.tokens
    return drafts


def chunk_page(
    parsed: ParsedPage,
    cfg: ChunkConfig,
    *,
    count: TokenCounter = estimate_tokens,
    quest_matcher: QuestMatcher | None = None,
) -> list[Chunk]:
    # Every chunk starts with "Title > Section path": keep room for the longest such header.
    reserve = 4 + max(
        (count(" > ".join((parsed.title, *s.path))) for s in parsed.sections), default=0
    )
    budget = max(int((cfg.max_tokens - reserve) * 0.95), 50)  # 5%: real tokenizers aren't additive
    inner = ChunkConfig(
        min_tokens=min(cfg.min_tokens, budget),
        max_tokens=budget,
        overlap_tokens=cfg.overlap_tokens,
    )
    units: list[_Unit] = []
    for section in parsed.sections:
        for block in _blocks(section.text):
            size = count(block)
            pieces = _split_block(block, inner, count) if size > inner.max_tokens else [block]
            units += [_Unit(section.path, piece, count(piece)) for piece in pieces]

    chunks: list[Chunk] = []
    for index, draft in enumerate(_pack(units, inner, count)):
        path = draft.header_path()
        header = " > ".join((parsed.title, *path))
        body = draft.body(path)
        content = f"{header}\n{body}"
        chunks.append(
            Chunk(
                page_title=parsed.title,
                section=" > ".join(path),
                page_type=parsed.page_type,
                url=parsed.url,
                content=content,
                tokens=count(content),
                metadata={
                    "page_title": parsed.title,
                    "section": " > ".join(path),
                    "page_type": parsed.page_type,
                    "url": parsed.url,
                    "chunk_index": index,
                    "revision_id": parsed.revision_id,
                    "skills": mentioned_skills(body),
                    "quests": quest_matcher(body) if quest_matcher else [],
                },
            )
        )
    return chunks
