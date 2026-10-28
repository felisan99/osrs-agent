"""Tidying the model's answer for a plain-text side panel.

The prompt asks for short paragraphs and flat lists, but models still reach for markdown tables,
headings and horizontal rules, which a Swing text pane would show as raw pipes and dashes, and they
sometimes cite a passage by its number ("[3]") although the prompt asks for page names. These are
fixed here, in code, so the plugin never has to look at them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.rag.retrieve import Passage

_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_RULE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_CITATION_RUN = re.compile(r"(?:\[\d{1,2}\]\s*)+")
_NUMBER = re.compile(r"\d+")


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _table_to_bullets(rows: list[str]) -> list[str]:
    """``| a | b |`` tables become one bullet per row: ``- first: second`` (or with headers)."""
    header = _cells(rows[0])
    body = [r for r in rows[1:] if not _SEPARATOR_ROW.match(r)]
    bullets: list[str] = []
    for row in body:
        cells = _cells(row)
        if len(cells) == 2:
            bullets.append(f"- {cells[0]}: {cells[1]}")
            continue
        rest = "; ".join(
            f"{h}: {c}" for h, c in zip(header[1:], cells[1:], strict=False) if c and h
        )
        bullets.append(f"- {cells[0]}" + (f" ({rest})" if rest else ""))
    return bullets


def _resolve_citations(text: str, passages: Sequence[Passage]) -> str:
    def replace(match: re.Match[str]) -> str:
        titles: list[str] = []
        for number in _NUMBER.findall(match.group(0)):
            index = int(number)
            if not 1 <= index <= len(passages):
                return match.group(0)  # not a passage reference: leave it alone
            title = passages[index - 1].page_title
            if title not in titles:
                titles.append(title)
        trailing = " " if match.group(0).endswith((" ", "\n")) else ""
        return "".join(f"[{t}]" for t in titles) + trailing

    return _CITATION_RUN.sub(replace, text)


def clean_answer(text: str, passages: Sequence[Passage] = ()) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|"):
            table = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table.append(lines[i])
                i += 1
            out.extend(_table_to_bullets(table) if len(table) > 1 else [table[0].strip(" |")])
            continue
        i += 1
        if _RULE.match(line):
            continue
        heading = _HEADING.match(line)
        out.append(f"**{heading.group(1)}**" if heading and heading.group(1) else line)
    cleaned = "\n".join(out)
    if passages:
        cleaned = _resolve_citations(cleaned, passages)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
