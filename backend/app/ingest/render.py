"""Wikitext -> readable plain text for the OSRS Wiki (SPEC §9).

Design rules:

* Text only: no markup, no images, no navigation boxes, no citation noise.
* Dynamic prices are DROPPED (``{{GEP}}``, ``{{HA}}``, ``{{Average drop value}}``): the wiki renders
  live numbers there, our dump would freeze stale ones, and prices must come from the tools.
* Templates that carry facts are rendered explicitly (skill requirements, drop lines, infoboxes,
  quest details, recipes). Unknown templates fall back to a conservative policy: keep short
  positional text, drop everything else (navboxes and data blobs).
* Tables become one ``cell | cell | cell`` line per row.
* Explanatory footnotes stay as ``(note: ...)`` right where they apply (``{{efn}}``, ``{{Refn}}``
  and ``<ref group=...>``: "while on a Slayer task the drop rate is 1/75" is part of the drop
  rate); plain ``<ref>`` citations are dropped.
* Tab groups (``<tabber>``, ``{{Tabber}}``, ``{{Switch infobox}}``) keep every tab under its label.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

import mwparserfromhell as mw
from mwparserfromhell.nodes import (
    Argument,
    Comment,
    ExternalLink,
    Heading,
    HTMLEntity,
    Tag,
    Template,
    Text,
    Wikilink,
)
from mwparserfromhell.wikicode import Wikicode

from app.expr import ExprError, evaluate

MAX_DEPTH = 40
MAX_VALUE_CHARS = 300
MAX_NOTE_CHARS = 450

# Infobox keys that only add noise to a text chunk (stat blocks nobody asks about in prose).
INFOBOX_SKIP_PREFIXES: dict[str, tuple[str, ...]] = {
    "monster": (
        "astab",
        "aslash",
        "acrush",
        "amagic",
        "arange",
        "attbns",
        "strbns",
        "mbns",
        "rngbns",
        "xpbonus",
        "id",
        "poisonres",
        "venomres",
        "immunecannon",
        "immunethrall",
        "size",
    ),
    "item": ("exchange", "options"),
}

DROP_TEMPLATES: frozenset[str] = frozenset(
    {
        "subject changes",
        "subject changes header",
        "subject changes footer",
        "clear",
        "external",
        "reflist",
        "notelist",
        "hastranscript",
        "has quick guide",
        "hasstrategy",
        "hastask",
        "has task",
        "rsl",
        "listen",
        "map",
        "maplink",
        "wp",
        "otheruses",
        "pagename",
        "sync",
        "synced switch",
        "leagueregion",
        "leagueregions",
        "droplogproject",
        "dropstablehead",
        "dropstablebottom",
        "loctablehead",
        "loctablebottom",
        "costtablehead",
        "costtablebottom",
        "average drop value",
        "herbdroptableinfo",
        "stub",
        "expand",
        "tocright",
        "toc",
        "bosses",
        "monsters",
        "main page",
        "questrelease",
        "update",
        "historical",
        "sic",
        "anchor",
        "mes",
        "infotable bonuses row",
        "graphicalupdatetable",
        "uses material list",
        "drop sources",
        "store locations list",
        "used in recommended equipment",
        "costume storage",
        "transcript",
        "emoteclue",
        "combatstyles",
        "hasquestguide",
        "confuse",  # hatnotes: "Not to be confused with ..." (a stray page name in the lead)
        "redirect",
        "namedref",  # re-cites an earlier reference: a citation, not a note
    }
)
DROP_PREFIXES: tuple[str, ...] = ("cite", "navbox", "#var", "#vardefine", "#time", "#if", "#switch")
DROP_SUFFIXES: tuple[str, ...] = ("droptableinfo",)  # "There is a 5/128 chance of rolling the ..."
SKIP_NAMESPACES = {"file", "image", "category", "media"}
DROP_TAGS = {
    "ref",
    "references",
    "gallery",
    "imagemap",
    "timeline",
    "mapframe",
    "maplink",
    "includeonly",
}

_SKILL_WORDS = {
    "quest": "{n} quest points",
    "combat": "combat level {n}",
    "total": "total level {n}",
}
_BLOCK_START = ("- ", "[", "Recipe", "Main article", "See also")
_NAME_WS = re.compile(r"[\s_]+")


def normalize_name(raw: str) -> str:
    return _NAME_WS.sub(" ", raw.strip()).lower()


# --------------------------------------------------------------------------- post-processing
_ATTR_CELL = re.compile(
    r"""^\s*(?:[\w-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s|]+)\s*)+\|(?!\|)(?P<content>.*)$""", re.S
)
_MAGIC_WORDS = re.compile(r"__(?:TOC|NOTOC|FORCETOC|NOEDITSECTION|NOINDEX|INDEX|NOGALLERY)__")


_PRICE_ARITHMETIC = re.compile(r"[*/+]|\s-\s")


def tidy_cell(text: str) -> str:
    """A table cell that is a formula over live prices carries no fact, only noise.

    The wiki shows the computed profit there; the dump only has the formula ("(0.75x GE price of
    Combat potion(4)*61/60 - GE price of Harralander potion (unf)) / 84 coins"), which would bloat
    every row of a training table. A plain "GE price of X coins" stays: it says what the figure
    depends on.
    """
    if "GE price of" in text and (
        text.count("GE price of") > 1 or _PRICE_ARITHMETIC.search(text.replace("GE price of", ""))
    ):
        return "(depends on GE prices)"
    return text


def _strip_cell(cell: str) -> str:
    match = _ATTR_CELL.match(cell)
    return tidy_cell((match.group("content") if match else cell).strip())


def convert_tables(text: str) -> str:
    """Turn ``{| ... |}`` table markup into one ``cell | cell`` line per row."""
    out: list[str] = []
    depth = 0
    rows: list[list[str]] = []
    row: list[str] = []

    def flush_row() -> None:
        nonlocal row
        if any(cell for cell in row):
            rows.append(row)
        row = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("{|"):
            if depth == 0:
                rows, row = [], []
            depth += 1
            continue
        if depth == 0:
            out.append(line)
            continue
        if stripped.startswith("|}"):
            depth -= 1
            if depth == 0:
                flush_row()
                out.extend(" | ".join(r) for r in rows)
                out.append("")
            continue
        if stripped.startswith("|+"):
            out.append(_strip_cell(stripped[2:]))
        elif stripped.startswith("|-"):
            flush_row()
        elif stripped.startswith("!"):
            row.extend(_strip_cell(c) for c in re.split(r"!!|\|\|", stripped[1:]))
        elif stripped.startswith("|"):
            row.extend(_strip_cell(c) for c in stripped[1:].split("||"))
        elif stripped and row:
            row[-1] = f"{row[-1]} {stripped}".strip()  # multi-line cell
    if depth > 0:  # unterminated table: keep what we have
        flush_row()
        out.extend(" | ".join(r) for r in rows)
    return "\n".join(out)


def flatten_parser_functions(text: str) -> str:
    """Collapse newlines inside ``{{#...}}`` parser functions (brace-balanced).

    A multi-line ``{{#expr: ... }}`` is valid on the wiki, but mwparserfromhell refuses a template
    name with a newline in the middle and then mis-pairs the closing braces, which can terminate
    an enclosing template early and silently drop the rest of the page.
    """
    out: list[str] = []
    n = len(text)
    i = 0
    while True:
        start = text.find("{{#", i)
        if start < 0:
            out.append(text[i:])
            break
        out.append(text[i:start])
        depth, k = 0, start
        while k < n:
            if text.startswith("{{", k):
                depth += 1
                k += 2
            elif text.startswith("}}", k):
                depth -= 1
                k += 2
                if depth <= 0:
                    break
            else:
                k += 1
        out.append(re.sub(r"\s*\n\s*", " ", text[start:k]))
        i = k
    return "".join(out)


_LIST_GAP = re.compile(r"(\n *- [^\n]*)\n\n(?= *- )")


def tidy(text: str) -> str:
    """Normalise whitespace, lists and leftover markup of already-rendered text."""
    text = text.replace("\xa0", " ").replace("\r", "")
    text = _MAGIC_WORDS.sub("", text)
    text = re.sub(r"'{2,5}", "", text)
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        match = re.match(r"^([*#:;]+)\s*(.*)$", line.lstrip())
        if match:
            marks, rest = match.groups()
            # '*' and '#' are list items; ':' (indent) and ';' (term) just carry text.
            line = "  " * (len(marks) - 1) + "- " + rest if marks[0] in "*#" else rest
        elif re.fullmatch(r"-{4,}", line.strip()):
            line = ""
        is_item = line.lstrip(" ").startswith("- ")
        lead = len(line) - len(line.lstrip(" ")) if is_item else 0
        line = " " * lead + re.sub(r"[ \t]{2,}", " ", line.strip())
        lines.append("" if line.strip() in {"-", ""} else line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Items rendered as blocks (drop lines, recipes) arrive separated by blank lines: compact them.
    while (compact := _LIST_GAP.sub(r"\1\n", text)) != text:
        text = compact
    return text.strip()


# --------------------------------------------------------------------------- renderer
Handler = Callable[["Renderer", Template], str]


def _block(text: str) -> str:
    return f"\n{text}\n" if text else ""


class Renderer:
    """Stateless apart from a recursion guard; safe to reuse across pages."""

    def __init__(self) -> None:
        self._depth = 0
        self._handlers: dict[str, Handler] = _build_handlers()
        self._notes: dict[str, str] = {}  # named explanatory notes of the page being rendered
        self._expanded: set[str] = set()  # named notes already shown in the current section

    # ---- public API
    def render(self, source: str | Wikicode) -> str:
        code = mw.parse(flatten_parser_functions(source)) if isinstance(source, str) else source
        return tidy(convert_tables(self._top(code.nodes)))

    def render_nodes(self, nodes: Any) -> str:
        """Render already-parsed nodes (e.g. the body of one section)."""
        return tidy(convert_tables(self._top(nodes)))

    def _top(self, nodes: Any) -> str:
        if self._depth == 0:
            self._expanded.clear()  # a named note is spelled out once per section
        return self._nodes(nodes)

    def collect_notes(self, code: Wikicode) -> None:
        """Remember the text of the page's named notes, so that a bare reuse of one
        (``<ref group=d name=x/>`` or ``{{efn|name=x}}``) can show it. Call once per page,
        before rendering its sections."""
        self._notes = {}
        for tag in code.filter_tags(recursive=True):
            if str(tag.tag).strip().lower() != "ref" or not tag.has("group"):
                continue
            name = _attribute(tag, "name")
            if name and tag.contents is not None:
                text = self.inline(tag.contents)
                if text:
                    self._notes.setdefault(name, text)
        for template in code.filter_templates(recursive=True):
            if normalize_name(str(template.name)) in {"efn", "refn"}:
                name, text = self._footnote_parts(template)
                if name and text:
                    self._notes.setdefault(name, text)

    def _footnote_parts(self, node: Template) -> tuple[str, str]:
        name, text = "", ""
        for param in node.params:
            key = normalize_name(str(param.name)) if param.showkey else "1"
            if key == "name":
                name = self.inline(param.value).strip("\"' ")
            elif key == "1" and not text:
                text = self.inline(param.value)
        return name, text

    def note(self, text: str, name: str = "") -> str:
        """`` (note: ...)`` for an explanatory footnote; empty for a note already shown."""
        text = text.strip() or self._notes.get(name, "")
        if not text:
            return ""
        if name:
            if name in self._expanded:
                return ""
            self._expanded.add(name)
        if len(text) > MAX_NOTE_CHARS:
            text = text[:MAX_NOTE_CHARS].rsplit(" ", 1)[0] + "..."
        return f" (note: {text})"

    def register(self, name: str, handler: Handler) -> None:
        """Add or replace the handler for a template (names are normalised)."""
        self._handlers[normalize_name(name)] = handler

    def inline(self, source: str | Wikicode) -> str:
        """Render and collapse to a single line (for template parameter values)."""
        return re.sub(r"\s+", " ", self.render(source).replace("\n", " ")).strip()

    # ---- nodes
    def _nodes(self, nodes: Any) -> str:
        self._depth += 1
        try:
            if self._depth > MAX_DEPTH:
                return ""
            return "".join(self._node(node) for node in nodes)
        finally:
            self._depth -= 1

    def _node(self, node: Any) -> str:
        if isinstance(node, Text):
            return str(node.value)
        if isinstance(node, Wikilink):
            return self._wikilink(node)
        if isinstance(node, Template):
            return self._template(node)
        if isinstance(node, Tag):
            return self._tag(node)
        if isinstance(node, Comment | Argument):
            return ""
        if isinstance(node, HTMLEntity):
            return html.unescape(node.normalize()) if isinstance(node.normalize(), str) else ""
        if isinstance(node, ExternalLink):
            return self._nodes(node.title.nodes) if node.title is not None else ""
        if isinstance(node, Heading):
            return self._nodes(node.title.nodes)
        return ""

    def _wikilink(self, node: Wikilink) -> str:
        title = str(node.title).strip()
        namespace = title.split(":", 1)[0].strip().lower() if ":" in title else ""
        if namespace in SKIP_NAMESPACES and not title.startswith(":"):
            return ""
        if node.text is not None:
            return self._nodes(node.text.nodes)
        return title.lstrip(":").split("#", 1)[0] or title

    def _tag(self, node: Tag) -> str:
        name = str(node.tag).strip().lower()
        if name == "ref":
            # <ref group=x> is an explanatory note; a plain <ref> is only a citation.
            if not node.has("group"):
                return ""
            text = self.inline(node.contents) if node.contents is not None else ""
            return self.note(text, _attribute(node, "name"))
        if name in DROP_TAGS:
            return ""
        if name == "tabber" and node.contents is not None:
            return _tabber_tag(self._nodes(node.contents.nodes))
        if name in {"li", "dd", "dt"} and node.wiki_markup:
            return node.wiki_markup  # "*", "#", ":" or ";": the item text follows as siblings
        if name in {"br", "hr"}:
            return "\n"
        if name == "table":
            return self._table(node)
        if node.contents is None:
            return ""
        inner = self._nodes(node.contents.nodes)
        if name == "li":
            return f"\n* {inner.strip()}"
        if name in {"p", "div", "ul", "ol", "blockquote", "center"}:
            return f"\n{inner}\n"
        return inner

    def _cell(self, cell: Tag) -> str:
        return tidy_cell(self.inline(cell.contents)) if cell.contents is not None else ""

    def _table(self, node: Tag) -> str:
        """One line per row. mwparserfromhell puts the first row's cells directly in the table
        (and a ``|+`` caption as a ``td`` starting with ``+``); later rows are ``tr`` tags."""
        rows: list[list[str]] = []
        row: list[str] = []
        caption = ""

        def flush() -> None:
            nonlocal row
            if any(row):
                rows.append(row)
            row = []

        children = node.contents.nodes if node.contents is not None else []
        for child in children:
            if not isinstance(child, Tag):
                continue
            kind = str(child.tag).strip().lower()
            if kind in {"td", "th"}:
                text = self._cell(child)
                if (
                    kind == "td"
                    and child.wiki_markup == "|"
                    and text.startswith("+")
                    and not rows
                    and not row
                ):
                    caption = text[1:].strip()
                    continue
                row.append(text)
            elif kind == "tr":
                flush()
                for cell in child.contents.nodes if child.contents is not None else []:
                    if isinstance(cell, Tag) and str(cell.tag).strip().lower() in {"td", "th"}:
                        row.append(self._cell(cell))
                flush()
        flush()
        lines = ([caption] if caption else []) + [" | ".join(r) for r in rows]
        return _block("\n".join(lines))

    # ---- templates
    def _template(self, node: Template) -> str:
        name = normalize_name(str(node.name))
        if name.startswith("#expr:"):
            return self._expr(node)
        if (
            not name
            or name in DROP_TEMPLATES
            or name.startswith(DROP_PREFIXES)
            or name.endswith(DROP_SUFFIXES)
        ):
            return ""
        handler = self._handlers.get(name)
        if handler is not None:
            return handler(self, node)
        if name.startswith("infobox "):
            return self.infobox(node, name.removeprefix("infobox ").strip())
        return self._unknown(node)

    def _expr(self, node: Template) -> str:
        """``{{#expr: ...}}`` with only numbers in it (a drop rate such as 1/{{#expr:1000/0.999}});
        anything that still holds an unresolved template or variable stays out."""
        expression = self.inline(node.name).split(":", 1)[1].strip()
        try:
            value = evaluate(expression)
        except ExprError:
            return ""
        return f"{value:.10g}"

    def positional(self, node: Template) -> list[str]:
        values: list[str] = []
        for param in node.params:
            if not param.showkey:
                values.append(self.inline(param.value))
        return values

    def params_code(self, node: Template) -> dict[str, Wikicode]:
        """Parameter values as parsed wikitext (``positional`` and ``named`` collapse newlines)."""
        return {normalize_name(str(param.name)): param.value for param in node.params}

    def named(self, node: Template) -> dict[str, str]:
        out: dict[str, str] = {}
        for param in node.params:
            if param.showkey:
                out[normalize_name(str(param.name))] = self.inline(param.value)
        return out

    def _unknown(self, node: Template) -> str:
        parts = [p for p in self.positional(node) if p]
        if not parts:
            return ""
        text = " ".join(parts)
        return text if len(text) <= 80 else ""

    def infobox(self, node: Template, kind: str) -> str:
        """``[Kind] Name`` plus one ``Label: value; ...`` line per version.

        ``version1``/``version2``... name the versions; keys ending in the same number
        (``combat1``, ``hitpoints1``) belong to that version, the rest apply to all of them.
        """
        kind = kind.lower()
        params: list[tuple[str, str]] = []
        for param in node.params:
            value = self.inline(param.value)
            if value:
                params.append((normalize_name(str(param.name)), value[:MAX_VALUE_CHARS]))
        labels = {
            int(m.group(2)): value
            for key, value in params
            if (m := _VERSION_KEY.fullmatch(key)) is not None
        }
        name = next((value for key, value in params if key == "name"), "")
        common: list[tuple[str, str]] = []
        versions: dict[int, list[tuple[str, str]]] = {}
        for key, value in params:
            digits = _TRAILING_DIGITS.match(key)
            if digits and labels and int(digits.group(2)) in labels and digits.group(1):
                base, number = digits.group(1), int(digits.group(2))
            else:
                base, number = key, None
            if _infobox_noise(base, kind):
                continue
            (common if number is None else versions.setdefault(number, [])).append((base, value))
        lines = []
        if common:
            lines.append(_infobox_line(kind, common))
        for number in sorted(versions):
            line = _infobox_line(kind, versions[number])
            if line:
                lines.append(f'Version "{labels[number]}": {line}')
        lines = [line for line in lines if line]
        if not lines:
            return ""
        header = f"[{kind.title()}] {name}".strip()
        return _block(header + "\n" + "\n".join(lines))

    def labelled(self, node: Template, title: str, fields: list[tuple[str, str]]) -> str:
        """Block with ``Label: value`` for each present field; list values go on their own lines."""
        params = {normalize_name(str(p.name)): p.value for p in node.params}
        lines = [f"[{title}]"]
        for key, label in fields:
            code = params.get(key)
            if code is None:
                continue
            text = self.render(code)
            if not text:
                continue
            if "\n" in text:
                lines.append(f"{label}:\n{text}")
            else:
                lines.append(f"{label}: {text}")
        return _block("\n".join(lines)) if len(lines) > 1 else ""


# --------------------------------------------------------------------------- infobox helpers
_VERSION_KEY = re.compile(r"(version)(\d{1,2})")
_TRAILING_DIGITS = re.compile(r"^(.*?)(\d{1,2})$")
_INFOBOX_NOISE = frozenset(
    {
        *("name", "version", "update", "id", "cat", "dropversion", "bucket", "bucketname"),
        *("options", "exchange", "removalupdate", "gemw", "defver", "aka"),
    }
)
_INFOBOX_LABELS: dict[str, str] = {
    "combat": "Combat level",
    "hitpoints": "Hitpoints",
    "max hit": "Max hit",
    "attack style": "Attack style",
    "attack speed": "Attack speed (ticks)",
    "slayxp": "Slayer XP",
    "slaylvl": "Slayer level required",
    "assignedby": "Assigned by",
    "att": "Attack level",
    "str": "Strength level",
    "def": "Defence level",
    "mage": "Magic level",
    "range": "Ranged level",
    "examine": "Examine",
    "respawn": "Respawn",
    "members": "Members",
    "release": "Release",
    "tradeable": "Tradeable",
    "equipable": "Equipable",
    "stackable": "Stackable",
    "noteable": "Noteable",
    "bankable": "Bankable",
    "value": "Store value (coins)",
    "weight": "Weight (kg)",
    "wornoptions": "Worn options",
    "quest": "Quest item",
    "elementalweaknesstype": "Elemental weakness",
    "poisonous": "Poisonous",
    "aggressive": "Aggressive",
}
_DEFENCE_KEYS = (
    ("dstab", "stab"),
    ("dslash", "slash"),
    ("dcrush", "crush"),
    ("dmagic", "magic"),
    ("drange", "ranged"),
    ("dlight", "light ranged"),
    ("dstandard", "standard ranged"),
    ("dheavy", "heavy ranged"),
)


def _attribute(tag: Tag, name: str) -> str:
    return str(tag.get(name).value).strip().strip("\"'") if tag.has(name) else ""


def _infobox_noise(base: str, kind: str) -> bool:
    always = ("image", "altimage", "icon", "smwname", "league", "hide", "pic")
    skipped = always + INFOBOX_SKIP_PREFIXES.get(kind, ())
    return base in _INFOBOX_NOISE or base.startswith(skipped)


def _infobox_line(kind: str, pairs: list[tuple[str, str]]) -> str:
    """``Label: value; Label: value``: the weakness and the defence bonuses are merged."""
    values = dict(pairs)
    parts: list[str] = []
    done: set[str] = set()
    for key, value in pairs:
        if key in done:
            continue
        done.add(key)
        if key == "elementalweaknesspercent" and "elementalweaknesstype" in values:
            continue  # shown with its type
        if key == "elementalweaknesstype":
            percent = values.get("elementalweaknesspercent")
            value = f"{value} ({percent}%)" if percent else value
        elif key in dict(_DEFENCE_KEYS):
            bonuses = [f"{label} {values[k]}" for k, label in _DEFENCE_KEYS if k in values]
            done.update(k for k, _ in _DEFENCE_KEYS)
            parts.append("Defence bonuses: " + ", ".join(bonuses))
            continue
        parts.append(f"{_INFOBOX_LABELS.get(key, key)}: {value}")
    return "; ".join(parts)


def _tabber_tag(text: str) -> str:
    """``<tabber>`` markup: ``Label=`` opens a tab and ``|-|`` separates tabs."""
    tabs = re.split(r"\n\s*\|-\|\s*\n?", "\n" + text.strip("\n") + "\n")
    out: list[str] = []
    for tab in tabs:
        match = re.match(r"\s*([^\n=]{1,80})=[ \t]*\n(.*)$", tab, re.S)
        if match:
            label, body = match.group(1).strip(), match.group(2).strip()
        else:
            label, body = "", tab.strip()
        if body:
            out.append(f"{label}:\n{body}" if label else body)
    return _block("\n\n".join(out))


# --------------------------------------------------------------------------- template handlers
def _first(r: Renderer, node: Template) -> str:
    parts = r.positional(node)
    return parts[0] if parts else ""


def _scp(r: Renderer, node: Template) -> str:
    parts = r.positional(node)
    if not parts:
        return ""
    skill = parts[0].strip()
    level = parts[1].strip() if len(parts) > 1 else ""
    template = _SKILL_WORDS.get(skill.lower())
    if template and level:
        return template.format(n=level)
    skill = skill[:1].upper() + skill[1:]
    return f"{level} {skill}".strip()


def _gep(r: Renderer, node: Template) -> str:
    """``{{GEP|Item|N}}`` is a live price on the wiki; keep the *dependency*, not a stale number."""
    parts = [p for p in r.positional(node) if p]
    if not parts:
        return ""
    qty = parts[1] if len(parts) > 1 and re.fullmatch(r"[\d.,]+", parts[1]) else ""
    return f"{qty}x GE price of {parts[0]}" if qty else f"GE price of {parts[0]}"


def _high_alch(r: Renderer, node: Template) -> str:
    item = _first(r, node)
    return f"high alchemy value of {item}" if item else ""


def _coins(r: Renderer, node: Template) -> str:
    amount = _first(r, node)
    return f"{amount} coins" if amount else "coins"


def _yes_no(yes: str, no: str) -> Handler:
    def handler(r: Renderer, node: Template) -> str:
        return yes if _first(r, node).lower() in {"yes", "y", "true", "1"} else no

    return handler


def _const(text: str) -> Handler:
    return lambda r, node: text


def _colour(r: Renderer, node: Template) -> str:
    parts = r.positional(node)
    return parts[-1] if len(parts) > 1 else ""


def _floor(r: Renderer, node: Template) -> str:
    uk = r.named(node).get("uk", "")
    if uk == "0":
        return "ground floor"
    return f"floor {uk}" if uk else "floor"


def _fairy(r: Renderer, node: Template) -> str:
    return _first(r, node)  # the prose around it already says "fairy ring code"


def _key_press(r: Renderer, node: Template) -> str:
    return "+".join(p for p in r.positional(node) if p)


def _formatexpr(r: Renderer, node: Template) -> str:
    expression = _first(r, node)
    try:
        value = evaluate(expression)
    except ExprError:
        return expression
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def _main(r: Renderer, node: Template) -> str:
    parts = [p for p in r.positional(node) if p]
    return _block(f"Main article: {', '.join(parts)}") if parts else ""


def _see_also(r: Renderer, node: Template) -> str:
    parts = [p for p in r.positional(node) if p]
    return _block(f"See also: {', '.join(parts)}") if parts else ""


_BROKEN_FRACTION = re.compile(r"(^|[^\d.,])/|/($|[^\d])")


def _drops_line(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    name = fields.get("name") or ""
    clue = fields.get("tier") or fields.get("type")  # DropsLineClue names the clue's tier
    if not name and clue:
        name = f"Clue scroll ({clue})"
    if not name:
        return ""
    quantity = " ".join(
        part for part in (fields.get("quantity"), fields.get("quantitynotes")) if part
    )
    rarity = fields.get("rarity", "")
    if _BROKEN_FRACTION.search(
        rarity
    ):  # "1/" or "/100": a part was a variable the dump cannot resolve
        rarity = ""
    detail = ", ".join(
        part
        for part in (
            quantity,
            f"rarity {rarity}" if rarity else "",
            f"alternative rarity {fields['altrarity']}" if fields.get("altrarity") else "",
            f"{fields['rolls']} rolls" if fields.get("rolls") not in (None, "", "1") else "",
        )
        if part
    )
    # namenotes carry "(members only)" / "(free-to-play only)"; raritynotes the conditions
    # ("1/75 while on a Slayer task"): both change what the row means.
    label = f"{name} {fields['namenotes']}" if fields.get("namenotes") else name
    notes = f" {fields['raritynotes']}" if fields.get("raritynotes") else ""
    return _block(f"- {label}: {detail}{notes}" if detail else f"- {label}{notes}")


def _table_lines(label: str) -> Handler:
    """Shared drop tables the wiki expands with Lua (herbs, seeds, gems): the contents are not in
    the page source, but *that the monster rolls the table, and how often* is."""

    def handler(r: Renderer, node: Template) -> str:
        parts = [p for p in r.positional(node) if p]
        rarity = parts[0] if parts and re.search(r"\d", parts[0]) else ""
        rolls = r.named(node).get("rolls") or (parts[1] if len(parts) > 1 else "")
        text = f"- {label}: rarity {rarity}" if rarity else f"- {label}"
        if rolls.isdigit() and rolls != "1":
            text += f", {rolls} rolls"
        return _block(text)

    return handler


def _rare_drop_table(r: Renderer, node: Template) -> str:
    parts = [p for p in r.positional(node) if p]
    lines = [f"- Rare drop table: rarity {parts[0]}" if parts else "- Rare drop table"]
    if len(parts) > 1:
        lines.append(f"- Gem drop table: rarity {parts[1]}")
    return _block("\n".join(lines))


def _footnote(r: Renderer, node: Template) -> str:
    name, text = r._footnote_parts(node)
    return r.note(text, name)


_MAP_KEYS = frozenset({"x", "y", "r", "plane", "mtype", "leagueregion", "dropversion", "gemw"})


def _key_values(label: str, skip: tuple[str, ...] = ()) -> Handler:
    def handler(r: Renderer, node: Template) -> str:
        pairs = [
            f"{key}: {value}"
            for key, value in r.named(node).items()
            if value
            and key not in _MAP_KEYS
            and not key.startswith(("image", "map", "icon", *skip))
        ]
        return _block(f"- {label}: " + "; ".join(pairs)) if pairs else ""

    return handler


def _loc_line(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    where = fields.get("location")
    if not where:
        return ""
    members = fields.get("members", "").lower()
    bits = [
        fields.get("name", ""),
        f"level {fields['levels']}" if fields.get("levels") else "",
        "members" if members == "yes" else "free-to-play" if members == "no" else "",
    ]
    detail = ", ".join(bit for bit in bits if bit)
    return _block(f"- {where} ({detail})" if detail else f"- {where}")


def _recipe(r: Renderer, node: Template) -> str:
    fields = r.named(node)

    def numbered(prefix: str) -> list[str]:
        items = []
        for i in range(1, 12):
            item = fields.get(f"{prefix}{i}")
            if item:
                qty = fields.get(f"{prefix}{i}quantity") or fields.get(f"{prefix}{i}qty")
                items.append(f"{qty} {item}" if qty else item)
        return items

    skills = []
    for i in range(1, 6):
        skill = fields.get(f"skill{i}")
        if skill:
            level = fields.get(f"skill{i}lvl", "")
            xp = fields.get(f"skill{i}exp", "")
            skills.append(f"{level} {skill}".strip() + (f" ({xp} xp)" if xp else ""))
    parts = []
    if skills:
        parts.append("skills: " + ", ".join(skills))
    for label, prefix in (("materials", "mat"), ("output", "output")):
        items = numbered(prefix)
        if items:
            parts.append(f"{label}: " + ", ".join(items))
    for label in ("facility", "tools", "notes"):
        if fields.get(label):
            parts.append(f"{label}: {fields[label]}")
    return _block("Recipe: " + "; ".join(parts)) if parts else ""


def _bonuses(r: Renderer, node: Template) -> str:
    f = r.named(node)

    def group(label: str, keys: tuple[tuple[str, str], ...]) -> str:
        items = [f"{name} {f[key]}" for key, name in keys if f.get(key)]
        return f"{label}: {', '.join(items)}" if items else ""

    lines = [
        group(
            "Attack bonuses",
            (
                ("astab", "stab"),
                ("aslash", "slash"),
                ("acrush", "crush"),
                ("amagic", "magic"),
                ("arange", "ranged"),
            ),
        ),
        group(
            "Defence bonuses",
            (
                ("dstab", "stab"),
                ("dslash", "slash"),
                ("dcrush", "crush"),
                ("dmagic", "magic"),
                ("drange", "ranged"),
            ),
        ),
        group(
            "Other bonuses",
            (
                ("str", "melee strength"),
                ("rstr", "ranged strength"),
                ("mdmg", "magic damage %"),
                ("prayer", "prayer"),
            ),
        ),
        group(
            "Weapon",
            (
                ("slot", "slot"),
                ("speed", "speed (ticks)"),
                ("attackrange", "range"),
                ("combatstyle", "style"),
            ),
        ),
    ]
    body = "\n".join(line for line in lines if line)
    return _block(f"[Equipment bonuses]\n{body}") if body else ""


def _quest_details(r: Renderer, node: Template) -> str:
    return r.labelled(
        node,
        "Quest details",
        [
            ("start", "Start"),
            ("difficulty", "Difficulty"),
            ("length", "Length"),
            ("description", "Description"),
            ("requirements", "Requirements"),
            ("items", "Items required"),
            ("recommended", "Recommended"),
            ("kills", "Enemies to defeat"),
        ],
    )


def _quest_rewards(r: Renderer, node: Template) -> str:
    return r.labelled(
        node,
        "Quest rewards",
        [("qp", "Quest points"), ("rewards", "Rewards"), ("xp", "Experience"), ("other", "Other")],
    )


def _item_spawn(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    name = fields.get("name") or fields.get("item") or ""
    where = fields.get("location", "")
    return _block(f"- {name}: {where}".strip(": ")) if name or where else ""


def _wrapper(label: str, key: str = "1") -> Handler:
    """Templates that only wrap content (checklists, plain lists): keep it, with its lists."""

    def handler(r: Renderer, node: Template) -> str:
        code = r.params_code(node).get(key)
        text = r.render(code) if code is not None else ""
        return _block(f"{label}\n{text}" if label and text else text)

    return handler


def _needed(r: Renderer, node: Template) -> str:
    code = r.params_code(node)
    lines = []
    keys = (("1", "Items needed"), ("recommended", "Items recommended"), ("skills", "Skills"))
    for key, label in keys:
        text = r.inline(code[key]) if key in code else ""
        if text:
            lines.append(f"{label}: {text}")
    return _block("\n".join(lines))


def _tabs(label_key: str, body_key: str) -> Handler:
    """``{{Tabber}}`` / ``{{Switch infobox}}``: every tab, under its own label."""

    def handler(r: Renderer, node: Template) -> str:
        code = r.params_code(node)
        out = []
        for i in range(1, 41):
            body = code.get(body_key.format(i=i))
            if body is None:
                continue
            text = r.render(body)
            label = r.inline(code[label_key.format(i=i)]) if label_key.format(i=i) in code else ""
            if text:
                out.append(f"{label}:\n{text}" if label else text)
        return _block("\n\n".join(out))

    return handler


def _clue_info(r: Renderer, node: Template) -> str:
    code = r.params_code(node)
    lines = []
    for key, label in (("text", "Clue text"), ("notes", "Solution"), ("items", "Items needed")):
        text = r.inline(code[key]) if key in code else ""
        if text:
            lines.append(f"{label}: {text}")
    return _block("\n".join(lines))


def _cryptic(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    if not fields.get("clue"):
        return ""
    solution = f" Solution: {fields['note']}" if fields.get("note") else ""
    return _block(f"- Clue: {fields['clue']}{solution}")


def _coordinates(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    if not fields.get("coords"):
        return ""
    extras = [
        f"{label}: {fields[key]}"
        for key, label in (("notes", "Solution"), ("req", "Requirements"), ("fight", "Fight"))
        if fields.get(key)
    ]
    return _block(f"- Coordinates {fields['coords']}. " + " ".join(extras))


_SLOTS = (
    *(("head", "Head"), ("neck", "Neck"), ("cape", "Cape"), ("body", "Body"), ("legs", "Legs")),
    *(("weapon", "Weapon"), ("shield", "Shield"), ("ammo", "Ammo"), ("hands", "Hands")),
    *(("feet", "Feet"), ("ring", "Ring"), ("special", "Special attack weapon")),
)


def _recommended_equipment(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    style = fields.get("style", "")
    lines = [
        f"Recommended equipment ({style}), options listed best first:"
        if style
        else "Recommended equipment, options listed best first:"
    ]
    for slot, label in _SLOTS:
        keys = [k for k in fields if re.fullmatch(rf"{slot}\d*", k) and fields[k]]
        keys.sort(key=lambda k: int(k[len(slot) :] or 0))
        if keys:
            lines.append(f"- {label}: " + "; ".join(fields[k] for k in keys))
    return _block("\n".join(lines)) if len(lines) > 1 else ""


def _equipment_setup(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    slots = ("head", "cape", "neck", "ammo", "weapon", "torso", "body", "legs", "shield", "gloves")
    slots += ("hands", "boots", "feet", "ring")
    worn = [f"{k}: {fields[k]}" for k in slots if fields.get(k)]
    return _block("Equipment setup: " + "; ".join(worn)) if worn else ""


def _inventory(r: Renderer, node: Template) -> str:
    """``{{Inventory|Shark|Shark|Coins\\1400|...}}``: the slots in order; a backslash and a number
    after an item is its quantity, ``;n`` marks it as noted."""
    items: list[list[Any]] = []
    for param in node.params:
        if param.showkey and not str(param.name).strip().isdigit():
            continue  # align=right, ...
        raw = r.inline(param.value)
        if not raw:
            continue  # an empty slot
        name, _, quantity = raw.partition("\\")
        noted = name.endswith(";n")
        name = name.removesuffix(";n").strip()
        label = f"{name} (noted)" if noted else name
        count = int(quantity) if quantity.isdigit() else 1
        if items and items[-1][0] == label:
            items[-1][1] += count  # the same item again: one more slot
        else:
            items.append([label, count])
    shown = [f"{n:,} {name}" if n > 1 else name for name, n in items]
    return _block("Inventory: " + ", ".join(shown)) if shown else ""


_RUNES = frozenset(
    {*("air", "water", "earth", "fire", "mind", "body", "cosmic", "chaos", "nature", "law")}
    | {*("death", "blood", "soul", "astral", "wrath", "dust", "lava", "mist", "mud", "smoke")}
    | {"steam", "sunfire"}
)


def _rune(name: str) -> str:
    return f"{name} rune" if name.lower() in _RUNES else name


def _spell_cost(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    runes = [
        f"{fields.get(f'rune{i}num') or '1'} {_rune(fields[f'rune{i}'])}"
        for i in range(1, 9)
        if fields.get(f"rune{i}")
    ]
    return _block("Runes required: " + ", ".join(runes)) if runes else ""


def _rune_req(r: Renderer, node: Template) -> str:
    return ", ".join(f"{n} {_rune(rune)}" for rune, n in r.named(node).items() if n)


def _store_line(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    name = fields.get("displayname") or fields.get("name")
    if not name:
        return ""
    detail = ", ".join(
        part
        for part in (
            f"stock {fields['stock']}" if fields.get("stock") else "",
            f"restock {fields['restock']}" if fields.get("restock") else "",
            f"sells for {fields['sell']}" if fields.get("sell") not in (None, "", "N/A") else "",
        )
        if part
    )
    return _block(f"- {name}: {detail}" if detail else f"- {name}")


def _location_monster(r: Renderer, node: Template) -> str:
    fields = r.named(node)
    if not fields.get("name"):
        return ""
    detail = ", ".join(
        part
        for part in (
            f"level {fields['levels']}" if fields.get("levels") else "",
            f"{fields['spawns']} spawns" if fields.get("spawns") else "",
            f"Slayer level {fields['slayerlevel']}" if fields.get("slayerlevel") else "",
            f"Slayer XP {fields['slayerxp']}" if fields.get("slayerxp") else "",
        )
        if part
    )
    return _block(f"- {fields['name']}: {detail}" if detail else f"- {fields['name']}")


def _build_handlers() -> dict[str, Handler]:
    handlers: dict[str, Handler] = {
        "scp": _scp,
        "gep": _gep,
        "gept": _gep,
        "ha": _high_alch,
        "coins": _coins,
        "nocoins": _first,
        "okay": _const("Yes"),
        "yes": _const("Yes"),
        "tick": _const("Yes"),
        "not okay": _const("No"),
        "no": _const("No"),
        "cross": _const("No"),
        "na": _const("N/A"),
        "n/a": _const("N/A"),
        "partial": _const("Partially"),
        "boostable": _yes_no("(boostable)", "(not boostable)"),
        "questreqstart": _yes_no("(needed to start the quest)", "(needed later in the quest)"),
        "colour": _colour,
        "color": _colour,
        "floornumber": _floor,
        "fairycode": _fairy,
        "key press": _key_press,
        "!": _const("|"),
        "=": _const("="),
        "formatnum": _first,
        "formatexpr": _formatexpr,
        "main": _main,
        "see also": _see_also,
        "*": _const("• "),
        "recipe": _recipe,
        "infobox bonuses": _bonuses,
        "quest details": _quest_details,
        "quest rewards": _quest_rewards,
        "itemspawnline": _item_spawn,
        "brimstone rarity": _const(
            "varies"
        ),  # a Lua formula of the monster's level; 305 is not a rate
        "(m)": _const("(members only)"),
        "(f)": _const("(free-to-play only)"),
        "efn": _footnote,
        "refn": _footnote,
        "raredroptable": _rare_drop_table,
        "gemdroptable": _table_lines("Gem drop table"),
        "needed": _needed,
        "checklist": _wrapper("", "1"),
        "plainlist": _wrapper("", "1"),
        "tabber": _tabs("tab{i}", "tab{i}content"),
        "switch infobox": _tabs("text{i}", "item{i}"),
        "multi infobox": _tabs("text{i}", "item{i}"),
        "clue info": _clue_info,
        "cryptics": _cryptic,
        "coordinates": _coordinates,
        "recommended equipment": _recommended_equipment,
        "equipment": _equipment_setup,
        "inventory": _inventory,
        "spell cost table": _spell_cost,
        "runereq": _rune_req,
        "storeline": _store_line,
        "locationmonstertableline": _location_monster,
        "locline": _loc_line,
        "costline": _key_values("Cost"),
        "teleportlocationline": _key_values("Teleport"),
    }
    for name in ("plink", "plinkt", "plinkp", "plinkit", "itemlink", "npclink", "pic", "ml"):
        handlers[name] = _first
    for name in ("dropsline", "dropslinereward", "dropslineclue", "dropslineskill"):
        handlers[name] = _drops_line
    for name, label in (
        ("herbdroplines", "Herb drop table"),
        ("uncommonseeddroplines", "Uncommon seed drop table"),
        ("rareseeddroplines", "Rare seed drop table"),
        ("generalseeddroplines", "General seed drop table"),
        ("allotmentseeddroplines", "Allotment seed drop table"),
        ("treeherbseeddroplines", "Tree and herb seed drop table"),
        ("usefulherbdroplines", "Useful herb drop table"),
    ):
        handlers[name] = _table_lines(label)
    return handlers


def render(text: str) -> str:
    """Convenience: render wikitext with a fresh renderer."""
    return Renderer().render(text)
