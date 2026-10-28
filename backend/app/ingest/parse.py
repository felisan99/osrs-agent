"""Page -> clean sections (SPEC §9): headings split the page, templates become readable text."""

from __future__ import annotations

from dataclasses import dataclass

import mwparserfromhell as mw
from mwparserfromhell.nodes import Heading

from app.clients.wiki import WikiPage, page_url
from app.ingest.mmg import MoneyMakingGuide, find_table, parse_money_making_guide, table_params
from app.ingest.render import Renderer, flatten_parser_functions

# Sections that add noise, not knowledge, to a retrieval corpus.
SKIP_SECTIONS: frozenset[str] = frozenset(
    {
        "references",
        "reference",
        "gallery",
        "sounds",
        "sound effects",
        "music",
        "trivia",
        "transcript",
        "transcripts",
        "dialogue",
        "see also",
        "external links",
        "navigation",
        "update history",
        "changes",
        "videos",
        "cheers",
    }
)


@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]  # headings from the top; () is the lead
    text: str


@dataclass(frozen=True)
class ParsedPage:
    title: str
    url: str
    page_type: str
    revision_id: int
    sections: list[Section]
    guide: MoneyMakingGuide | None = None


def _number(value: float) -> str:
    text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:.4g}"
    return text


def guide_summary(guide: MoneyMakingGuide, details: str = "") -> str:
    """Prose for a money making guide: requirements, rates and items, but NEVER prices.

    Prices on the wiki page are live values that go stale in a dump; profit is computed with
    ``rank_moneymakers`` at query time instead.
    """
    lines = [f"Money making guide: {guide.activity}"]
    facts = [
        ("Category", guide.category),
        ("Intensity", guide.intensity),
        ("Location", guide.location),
        ("Members only", {True: "yes", False: "no", None: None}[guide.members_only]),
    ]
    lines += [f"{label}: {value}" for label, value in facts if value]
    if guide.requirements_text:
        lines.append("Requirements:")
        lines += [f"- {text}" for text in guide.requirements_text]
    if guide.kind == "recurring" and guide.activity_minutes:
        again = f"; can be repeated every {guide.recurrence}" if guide.recurrence else ""
        lines.append(f"Takes {_number(guide.activity_minutes)} minutes per instance{again}")
    elif guide.is_per_kill and guide.kph:
        unit = guide.kph_label or "actions per hour"
        lines.append(f"Rate: {_number(guide.kph)} ({unit})")
    per = "per instance" if guide.kind == "recurring" else "per hour"
    for title, items in (("Inputs", guide.items_in), ("Outputs", guide.items_out)):
        if not items:
            continue
        lines.append(f"{title} ({per}):")
        for item in items:
            amount = item.qty if guide.kind == "recurring" else (item.qty_per_hour or item.qty)
            names = (
                f"{item.name} (cheapest of {', '.join(item.cheapest_of)})"
                if item.cheapest_of
                else item.name
            )
            lines.append(f"- {_number(amount)} {names}")
    if guide.experience:
        lines.append(
            "Experience: "
            + ", ".join(f"{_number(x.qty_per_hour or x.qty)} {x.skill}" for x in guide.experience)
        )
    if details:
        lines += ["", details]
    return "\n".join(lines)


def split_sections(wikitext: str, renderer: Renderer) -> list[Section]:
    """Split on ``==`` headings; render each body; drop empty and noise sections."""
    code = mw.parse(flatten_parser_functions(wikitext))
    renderer.collect_notes(code)  # so a bare reuse of a named note shows the note's text
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    skipping_below: int | None = None  # skip a noisy section together with its subsections
    for part in code.get_sections(levels=[2, 3, 4, 5, 6], include_lead=True, flat=True):
        nodes = part.nodes
        heading = nodes[0] if nodes and isinstance(nodes[0], Heading) else None
        if heading is None:
            path: tuple[str, ...] = ()
            body = nodes
        else:
            level = int(heading.level)
            title = renderer.inline(heading.title)
            if skipping_below is not None and level > skipping_below:
                continue
            skipping_below = None
            if title.lower() in SKIP_SECTIONS:
                skipping_below = level
                continue
            stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
            path = tuple(t for _, t in stack)
            body = nodes[1:]
        text = renderer.render_nodes(body)
        if text:
            sections.append(Section(path, text))
    return sections


def parse_page(
    page: WikiPage,
    page_type: str,
    *,
    renderer: Renderer | None = None,
    base_url: str = "https://oldschool.runescape.wiki/w/",
) -> ParsedPage | None:
    """Parse one wiki page into sections (``None`` for redirects and pages with no text)."""
    if page.is_redirect:
        return None
    renderer = renderer or Renderer()
    sections = split_sections(page.wikitext, renderer)
    guide: MoneyMakingGuide | None = None
    if page_type == "mmg":
        guide = parse_money_making_guide(page.title, page.wikitext)
        if guide is not None:
            table = find_table(mw.parse(flatten_parser_functions(page.wikitext)))
            details_code = table_params(table).get("details") if table is not None else None
            details = renderer.render(details_code) if details_code is not None else ""
            sections.insert(0, Section((), guide_summary(guide, details)))
    if not sections:
        return None
    return ParsedPage(
        title=page.title,
        url=page_url(page.title, base_url),
        page_type=page_type,
        revision_id=page.revision_id,
        sections=sections,
        guide=guide,
    )
