"""Deterministic parser for the OSRS Wiki money making guides (SPEC §8).

Semantics follow the wiki's own ``Module:Mmgtable`` (read on 2026-10-14):

* Items are ``Input{N}`` / ``Output{N}`` with ``Input{N}num`` (quantity; a number or a ``#expr``
  formula, default 1), optional ``Input{N}value`` (a fixed value or a price formula) and
  ``Input{N}isph``.
* With ``isperkill`` the quantities are per kill/action and are multiplied by ``kph`` unless the
  item is flagged ``isph`` (already per hour). Without it, quantities are per hour.
* Only OUTPUTS pay Grand Exchange tax, and only when valued by market price (a fixed ``value``
  override is never taxed). ``Mmgtable recurring`` guides give per-instance quantities and an
  ``Activity Time`` in minutes.

Prices are NEVER frozen here: guides store quantities plus price formulas (``gep("Item", n)``) and
are valued with live prices at query time. Values the wiki computes in Lua (``{{GDTValue}}``,
``{{SeedPackValue}}``, ``{{Average drop value}}`` ...) cannot be reproduced, so such guides are
marked ``unsupported`` instead of being valued wrongly.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from typing import Literal

import mwparserfromhell as mw
from mwparserfromhell.nodes import Comment, HTMLEntity, Template, Text, Wikilink
from mwparserfromhell.wikicode import Wikicode
from pydantic import BaseModel, Field

from app.expr import ExprError, calls_in, evaluate
from app.ingest.render import Renderer, flatten_parser_functions, normalize_name
from app.tools.items import ItemIndex

MAX_ITEMS = 75
SKILLS: frozenset[str] = frozenset(
    {
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
    }
)
NOT_QUESTS = {"quest points", "quest", "quests", "miniquest", "miniquests", "ironman", "ironmen"}
TRUE_WORDS = {"y", "yes", "true", "1"}
FALSE_WORDS = {"n", "no", "false", "0"}
SOFT_MARKERS = ("optional", "if you", "ironman", "ironmen", "alternatively", "not required")
# Item names the wiki computes as "the cheapest of these by current price" (Template:MinPrice).
# Transcribed from the wiki on 2026-10-14; the ranker picks the cheapest at query time.
CHOICE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "cheap food": ("Manta ray", "Tuna potato", "Anglerfish", "Dark crab", "Bluefin"),
}


class Unsupported(Exception):
    """A formula uses something we cannot evaluate (Lua-computed templates, unknown variables)."""


# --------------------------------------------------------------------------- model
class GuideItem(BaseModel):
    name: str
    # When set, the item is "the cheapest of these by current price" (e.g. {{Cheap food}}).
    cheapest_of: list[str] = Field(default_factory=list)
    item_id: int | None = None
    qty: float  # as written (per kill / per hour / per instance)
    is_per_hour: bool  # True when `qty` is already an hourly amount
    qty_per_hour: float | None = None
    fixed_value: float | None = None  # fixed coin value override (never taxed)
    value_expr: str | None = None  # price formula, e.g. gep("Fire rune", 3) + gep("Blood rune")
    note: str | None = None


class ExperienceLine(BaseModel):
    skill: str
    qty: float
    is_per_hour: bool
    qty_per_hour: float | None = None


class MoneyMakingGuide(BaseModel):
    page_title: str
    kind: Literal["standard", "recurring"] = "standard"
    activity: str
    category: str | None = None
    intensity: str | None = None
    location: str | None = None
    skill_category: str | None = None
    members_only: bool | None = None
    # Not a template parameter. Only the guides titled "(Ironman)" are known to be written for
    # ironmen; for every other guide it stays None (unknown), never False (SPEC §8).
    ironman_friendly: bool | None = None
    # Largest `{{Coins|N}}` in the guide's Item line ("10,000,000+ recommended"): the bankroll the
    # wiki says the method needs. A recommendation, not a hard requirement.
    capital: int | None = None
    # `Exclude = yes`: the wiki marks the guide obsolete (Category:Obsolete money making guides) and
    # keeps it out of its own rankings, so the ranker skips it too.
    obsolete: bool = False
    skill_requirements: dict[str, int] = Field(default_factory=dict)
    combat_level: int | None = None
    quest_points: int | None = None
    total_level: int | None = None
    quest_requirements: list[str] = Field(default_factory=list)
    requirements_text: list[str] = Field(default_factory=list)
    kph: float | None = None
    kph_label: str | None = None
    is_per_kill: bool = False
    activity_minutes: float | None = None
    recurrence: str | None = None
    items_in: list[GuideItem] = Field(default_factory=list)
    items_out: list[GuideItem] = Field(default_factory=list)
    experience: list[ExperienceLine] = Field(default_factory=list)
    source_hourly_profit: int | None = None  # wiki's own figure at a point in time; reference only
    issues: list[str] = Field(default_factory=list)  # informational
    unsupported: list[str] = Field(default_factory=list)  # reasons it cannot be valued exactly

    @property
    def rankable(self) -> bool:
        return not self.unsupported and bool(self.items_out)


# --------------------------------------------------------------------------- formula building
class _Formulas:
    """Turns wikitext fragments into evaluable expression text, tracking ``#vardefine`` state."""

    def __init__(self, renderer: Renderer) -> None:
        self.renderer = renderer
        self.variables: dict[str, float | None] = {}

    def to_expr(self, code: Wikicode, *, prices: bool) -> str:
        parts: list[str] = []
        for node in code.nodes:
            if isinstance(node, Text):
                parts.append(str(node.value))
            elif isinstance(node, Comment):
                continue
            elif isinstance(node, HTMLEntity):
                parts.append(" ")
            elif isinstance(node, Template):
                parts.append(self._template(node, prices=prices))
            else:
                raise Unsupported(f"unsupported syntax ({type(node).__name__}) in a formula")
        return "".join(parts).strip()

    def _positional(self, node: Template) -> list[Wikicode]:
        return [p.value for p in node.params if not p.showkey]

    def _template(self, node: Template, *, prices: bool) -> str:
        raw = str(node.name).strip()
        low = raw.lower()
        if low.startswith("#expr:"):
            return "(" + self.to_expr(mw.parse(raw[len("#expr:") :]), prices=prices) + ")"
        if low.startswith("#var:"):
            variable = raw.split(":", 1)[1].strip()
            value = self.variables.get(variable)
            if value is None:
                default = self._positional(node)
                if default:
                    return "(" + self.to_expr(default[0], prices=prices) + ")"
                raise Unsupported(f"unresolved variable {variable!r}")
            return f"({value!r})"
        if low.startswith(("#vardefine:", "#vardefineecho:")):
            variable = raw.split(":", 1)[1].strip()
            value = self.define(variable, node)
            if low.startswith("#vardefineecho:"):
                if value is None:
                    raise Unsupported(f"variable {variable!r} could not be evaluated")
                return f"({value!r})"
            return ""
        name = normalize_name(raw)
        if name in {"gep", "gept"} and prices:
            args = self._positional(node)
            if not args:
                raise Unsupported("empty {{GEP}}")
            item = self.renderer.inline(args[0]).replace('"', "")
            qty = self.to_expr(args[1], prices=prices) if len(args) > 1 else "1"
            return f'gep("{item}", {qty or "1"})'
        if name == "ha" and prices:
            args = self._positional(node)
            if not args:
                raise Unsupported("empty {{HA}}")
            return f'ha("{self.renderer.inline(args[0]).replace(chr(34), "")}")'
        if name in {"formatnum", "formatexpr"}:
            args = self._positional(node)
            if args:
                return "(" + self.to_expr(args[0], prices=prices) + ")"
        raise Unsupported(
            f"template {{{{{raw}}}}} is computed by the wiki and cannot be reproduced"
        )

    def define(self, variable: str, node: Template) -> float | None:
        args = self._positional(node)
        value: float | None = None
        if args:
            try:
                value = evaluate(self.to_expr(args[0], prices=False))
            except (Unsupported, ExprError):
                value = None
        self.variables[variable] = value
        return value

    def predefine(self, code: Wikicode) -> None:
        """Process every ``#vardefine`` of the page in document order."""
        for template in code.filter_templates(recursive=True):
            name = str(template.name).strip().lower()
            if name.startswith(("#vardefine:", "#vardefineecho:")):
                self.define(str(template.name).split(":", 1)[1].strip(), template)

    def number(self, code: Wikicode) -> float:
        return evaluate(self.to_expr(code, prices=False))


# --------------------------------------------------------------------------- helpers
def _truthy(text: str) -> bool | None:
    word = re.split(r"[\s<]", text.strip().lower(), maxsplit=1)[0]
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    return None


def table_params(template: Template) -> dict[str, Wikicode]:
    return {str(p.name).strip().lower(): p.value for p in template.params}


_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _soft_line(line: str) -> bool:
    lowered = line.lower()
    if any(marker in lowered for marker in SOFT_MARKERS):
        return True
    if "recommended" in lowered:
        # "{{SCP|Fishing|43}} (99 recommended)": the SCP is the requirement, the parenthesis the
        # recommendation. Anything else that says "recommended" is a recommendation, not a need.
        return re.search(r"\(\s*[^()]*recommended", lowered) is None
    return False


class _Requirements:
    def __init__(self, guide: MoneyMakingGuide, renderer: Renderer) -> None:
        self.guide = guide
        self.renderer = renderer

    def parse(self, code: Wikicode | None, label: str) -> None:
        if code is None:
            return
        # One bullet often packs several requirements separated by <br/>: each is judged on its own,
        # or a "recommended" in one of them would demote a hard requirement written next to it.
        segments = (piece for raw_line in str(code).split("\n") for piece in _BREAK.split(raw_line))
        for raw_line in segments:
            line = raw_line.strip().lstrip("*#: ").strip()
            parsed = mw.parse(line)
            text = self.renderer.inline(parsed)
            if not text or text.lower().strip(". ") in {"none", "n/a", "no", "nothing"}:
                continue
            self.guide.requirements_text.append(f"{label}: {text}")
            soft = _soft_line(line)
            for template in parsed.filter_templates(recursive=True):
                if normalize_name(str(template.name)) != "scp" or soft:
                    continue
                args = [self.renderer.inline(p.value) for p in template.params if not p.showkey]
                level_match = (
                    re.fullmatch(r"(\d{1,4})\+?", args[1].strip()) if len(args) > 1 else None
                )
                if level_match is None:
                    continue
                skill, level = args[0].strip().lower(), int(level_match.group(1))
                if skill in SKILLS:
                    current = self.guide.skill_requirements.get(skill.capitalize(), 0)
                    self.guide.skill_requirements[skill.capitalize()] = max(current, level)
                elif skill == "quest":
                    self.guide.quest_points = max(self.guide.quest_points or 0, level)
                elif skill == "combat":
                    self.guide.combat_level = max(self.guide.combat_level or 0, level)
                elif skill == "total":
                    self.guide.total_level = max(self.guide.total_level or 0, level)
            if label == "Quest" and not soft:
                for link in parsed.filter_wikilinks():
                    title = str(link.title).split("#", 1)[0].replace("_", " ").strip()
                    is_quest = title and title.lower() not in NOT_QUESTS and ":" not in title
                    if is_quest and title not in self.guide.quest_requirements:
                        self.guide.quest_requirements.append(title)


def find_table(code: Wikicode) -> Template | None:
    for template in code.filter_templates(recursive=True):
        if normalize_name(str(template.name)) in {"mmgtable", "mmgtable recurring"}:
            return template
    return None


# --------------------------------------------------------------------------- parser
def parse_money_making_guide(title: str, wikitext: str) -> MoneyMakingGuide | None:
    """Parse one ``Money making guide/...`` page; ``None`` if it has no guide table."""
    code = mw.parse(flatten_parser_functions(wikitext))
    table = find_table(code)
    if table is None:
        return None
    renderer = Renderer()
    formulas = _Formulas(renderer)
    formulas.predefine(code)
    params = table_params(table)
    recurring = normalize_name(str(table.name)) == "mmgtable recurring"

    def text(key: str) -> str | None:
        value = params.get(key)
        rendered = renderer.inline(value) if value is not None else ""
        return rendered or None

    guide = MoneyMakingGuide(
        page_title=title,
        kind="recurring" if recurring else "standard",
        activity=text("activity") or title.rsplit("/", 1)[-1],
        category=text("category"),
        intensity=text("intensity"),
        location=text("location"),
        skill_category=text("skillcategory"),
        kph_label=text("kph name"),
        recurrence=text("recurrence time"),
    )
    members = text("members")
    guide.members_only = _truthy(members) if members else None
    exclude = text("exclude")
    guide.obsolete = bool(exclude and _truthy(exclude))
    if "(ironman)" in title.lower():
        guide.ironman_friendly = True
    guide.capital = _capital(params.get("item"))

    # -- how quantities become hourly amounts
    per_kill_text = text("isperkill")
    guide.is_per_kill = bool(per_kill_text and _truthy(per_kill_text))
    if "kph" in params:
        try:
            guide.kph = formulas.number(params["kph"])
        except (Unsupported, ExprError) as exc:
            guide.unsupported.append(f"kph: {exc}")
    if guide.is_per_kill and guide.kph is None and not guide.unsupported:
        guide.issues.append("isperkill without a numeric kph: the wiki defaults to 1")
        guide.kph = 1.0
    if recurring:
        try:
            guide.activity_minutes = formulas.number(params["activity time"])
            if guide.activity_minutes <= 0:
                raise ExprError("activity time must be positive")
        except (KeyError, Unsupported, ExprError) as exc:
            guide.unsupported.append(f"activity time: {exc}")

    # -- requirements
    reqs = _Requirements(guide, renderer)
    for key, label in (
        ("skill", "Skill"),
        ("quest", "Quest"),
        ("item", "Item"),
        ("other", "Other"),
    ):
        reqs.parse(params.get(key), label)

    # -- items and experience
    guide.items_in = _items(guide, params, "input", formulas, renderer)
    guide.items_out = _items(guide, params, "output", formulas, renderer)
    guide.experience = _experience(guide, params, formulas, renderer)
    if not guide.items_out and not guide.unsupported:
        guide.unsupported.append("no outputs: nothing to value")
    return guide


def _capital(code: Wikicode | None) -> int | None:
    """The biggest ``{{Coins|N}}`` amount in a parameter, or ``None``."""
    if code is None:
        return None
    amounts: list[int] = []
    for template in code.filter_templates(recursive=True):
        if normalize_name(str(template.name)) != "coins":
            continue
        amount = str(template.params[0].value).strip() if template.params else ""
        if re.fullmatch(
            r"\d[\d,]*", amount
        ):  # a formula ({{GEP|...}}*16200) depends on live prices
            amounts.append(int(amount.replace(",", "")))
    return max(amounts) if amounts else None


def _per_hour(guide: MoneyMakingGuide, qty: float, is_per_hour: bool) -> float | None:
    if guide.kind == "recurring":
        return qty * 60 / guide.activity_minutes if guide.activity_minutes else None
    if guide.is_per_kill and not is_per_hour:
        return qty * (guide.kph or 1.0)
    return qty


def _items(
    guide: MoneyMakingGuide,
    params: dict[str, Wikicode],
    prefix: str,
    formulas: _Formulas,
    renderer: Renderer,
) -> list[GuideItem]:
    items: list[GuideItem] = []
    for i in range(1, MAX_ITEMS + 1):
        name_code = params.get(f"{prefix}{i}")
        raw_name = str(name_code).strip() if name_code is not None else ""
        if not raw_name:
            break  # the wiki module stops at the first missing/empty entry
        key = f"{prefix.capitalize()}{i}"
        name = renderer.inline(name_code) if name_code is not None else ""
        choices: list[str] = []
        if not name:
            templates = name_code.filter_templates(recursive=False) if name_code is not None else []
            known = normalize_name(str(templates[0].name)) if len(templates) == 1 else ""
            if known in CHOICE_TEMPLATES:
                choices = list(CHOICE_TEMPLATES[known])
                name = known.capitalize()
            else:
                # The wiki computes this item name (a template or variable): we cannot know it, and
                # silently skipping it would misstate profit.
                guide.unsupported.append(f"{key} name is computed by the wiki ({raw_name[:50]})")
                continue
        isph_text = (
            renderer.inline(params[f"{prefix}{i}isph"]) if f"{prefix}{i}isph" in params else ""
        )
        is_per_hour = guide.kind == "standard" and (
            (not guide.is_per_kill) or bool(_truthy(isph_text))
        )

        qty = 1.0
        if f"{prefix}{i}num" in params:
            try:
                qty = formulas.number(params[f"{prefix}{i}num"])
            except (Unsupported, ExprError) as exc:
                guide.unsupported.append(f"{key} quantity ({name}): {exc}")

        fixed_value: float | None = None
        value_expr: str | None = None
        if f"{prefix}{i}value" in params:
            try:
                expression = formulas.to_expr(params[f"{prefix}{i}value"], prices=True)
            except Unsupported as exc:
                guide.unsupported.append(f"{key} value ({name}): {exc}")
                expression = ""
            if expression:
                try:
                    if calls_in(expression):
                        # a bare {{GEP|<the item itself>}} is just the live price, i.e. no override
                        own_price = f'gep("{name}", 1)'.replace(" ", "")
                        if expression.replace(" ", "") != own_price:
                            value_expr = expression
                    else:
                        fixed_value = evaluate(expression)
                except ExprError as exc:
                    guide.unsupported.append(f"{key} value ({name}): {exc}")

        note = renderer.inline(params[f"{prefix}{i}note"]) if f"{prefix}{i}note" in params else ""
        items.append(
            GuideItem(
                name=name,
                cheapest_of=choices,
                qty=qty,
                is_per_hour=is_per_hour,
                qty_per_hour=_per_hour(guide, qty, is_per_hour),
                fixed_value=fixed_value,
                value_expr=value_expr,
                note=note or None,
            )
        )
    return items


def _experience(
    guide: MoneyMakingGuide,
    params: dict[str, Wikicode],
    formulas: _Formulas,
    renderer: Renderer,
) -> list[ExperienceLine]:
    lines: list[ExperienceLine] = []
    for i in range(1, MAX_ITEMS + 1):
        skill_code = params.get(f"experience{i}")
        skill = renderer.inline(skill_code) if skill_code is not None else ""
        if not skill:
            break
        isph_text = (
            renderer.inline(params[f"experience{i}isph"]) if f"experience{i}isph" in params else ""
        )
        is_per_hour = guide.kind == "standard" and (
            (not guide.is_per_kill) or bool(_truthy(isph_text))
        )
        qty = 0.0
        if f"experience{i}num" in params:
            try:
                qty = formulas.number(params[f"experience{i}num"])
            except (Unsupported, ExprError):
                guide.issues.append(f"experience{i} ({skill}) quantity could not be evaluated")
        lines.append(
            ExperienceLine(
                skill=skill,
                qty=qty,
                is_per_hour=is_per_hour,
                qty_per_hour=_per_hour(guide, qty, is_per_hour),
            )
        )
    return lines


def link_titles(code: Wikicode) -> list[str]:
    """Titles of the wikilinks in ``code`` (anchors and underscores normalised)."""
    return [
        str(link.title).split("#", 1)[0].replace("_", " ").strip()
        for link in code.filter_wikilinks()
        if isinstance(link, Wikilink)
    ]


# --------------------------------------------------------------------------- post-parse steps
_LIST_ROW = re.compile(
    r'<tr><td><a [^>]*title="(Money making guide/[^"]+)"[^>]*>[^<]*</a></td>'
    r'<td data-sort-value="(-?[\d.]+)"'
)


def parse_reference_profits(rendered_html: str) -> dict[str, int]:
    """``{page title: hourly profit}`` from the rendered *Money making guide* list table.

    The wikitext has no profit (Lua computes it with live prices), but the list page shows it with
    the exact value in ``data-sort-value``. Reference only: it is the wiki's figure at one moment,
    never used to rank (see SPEC §8, ``source_hourly_profit``).
    """
    profits: dict[str, int] = {}
    for match in _LIST_ROW.finditer(rendered_html):
        profits.setdefault(html.unescape(match.group(1)), round(float(match.group(2))))
    return profits


def attach_reference_profits(guides: Iterable[MoneyMakingGuide], profits: Mapping[str, int]) -> int:
    """Fill ``source_hourly_profit``; returns how many guides got one."""
    attached = 0
    for guide in guides:
        profit = profits.get(guide.page_title)
        if profit is not None:
            guide.source_hourly_profit = profit
            attached += 1
    return attached


def resolve_item_ids(guide: MoneyMakingGuide, index: ItemIndex) -> None:
    """Fill ``item_id`` from ``/mapping`` and flag guides that contain an item we cannot price.

    An item needs a GE price unless the guide gives it a fixed value or a price formula. If such
    an item is not in ``/mapping`` (untradeable, renamed...) valuing it at 0 would misstate the
    profit, so the guide is marked unsupported instead.
    """
    for item in (*guide.items_in, *guide.items_out):
        needs_price = item.fixed_value is None and item.value_expr is None
        if item.cheapest_of:
            resolved = [index.resolve(name) for name in item.cheapest_of]
            if needs_price and not any(r.status == "exact" for r in resolved):
                guide.unsupported.append(f"none of {item.cheapest_of} is in /mapping")
            continue
        resolution = index.resolve(item.name)
        if resolution.status == "exact" and resolution.item is not None:
            item.item_id = resolution.item.id
        elif needs_price and item.name:
            guide.unsupported.append(f"no Grand Exchange price for '{item.name}' (not in /mapping)")
