"""Principle 1 as a check: a large number in an answer must come from a real source.

The model is told never to state prices or profits that did not come from a tool, and never to do
arithmetic on them. Telling is not enough, so after generation every number of 1,000 or more in the
answer is looked up in what the model was given (tool results, the player's own words, the wiki
passages). A number with no source, within rounding, is "ungrounded": the caller asks for a rewrite
and, failing that, drops the sentence.

Numbers are read in the formats players use: ``1,500,000``, ``1.500.000``, ``1.5M``, ``1,5 M``,
``2.3 millones``, ``300k``. Small numbers (levels, quantities, percentages) are not checked, nor are
years.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

FLOOR = 1_000
EXACT_TOLERANCE = 0.005  # a figure quoted as written may differ by half a percent (float noise)
ROUNDED_TOLERANCE = 0.03  # "2.96M" for 2,961,443: abbreviated figures are rounded
_SUFFIXES = {
    "k": 1e3,
    "mil": 1e3,
    "m": 1e6,
    "million": 1e6,
    "millon": 1e6,
    "millón": 1e6,
    "millones": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}
_NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2}(?!\d))?(?!\d)|\d+(?:[.,]\d+)?)"
    r"(?:\s?(millones|millón|million|billion|mil|bn|k|m|b)\b)?",
    re.IGNORECASE,
)


def _to_float(token: str) -> float:
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):  # 1.234,5
            token = token.replace(".", "").replace(",", ".")
        else:  # 1,234.5
            token = token.replace(",", "")
    elif "," in token or "." in token:
        sep = "," if "," in token else "."
        parts = token.split(sep)
        thousands = len(parts) > 2 or (len(parts[1]) == 3 and parts[0] not in ("0", ""))
        token = token.replace(sep, "") if thousands else token.replace(sep, ".")
    return float(token)


def numbers_in(text: str) -> list[tuple[str, float, bool]]:
    """``(as written, value, abbreviated)`` for every number in ``text``."""
    found: list[tuple[str, float, bool]] = []
    for match in _NUMBER.finditer(text):
        raw, suffix = match.group(1), match.group(2)
        try:
            value = _to_float(raw)
        except ValueError:
            continue
        if suffix:
            value *= _SUFFIXES[suffix.lower()]
        found.append((match.group(0).strip(), value, bool(suffix)))
    return found


def _is_year(value: float, abbreviated: bool) -> bool:
    return not abbreviated and 1900 <= value <= 2100 and value == int(value)


def ungrounded_numbers(answer: str, sources: Iterable[str], *, floor: int = FLOOR) -> list[str]:
    """Numbers of ``floor`` or more in ``answer`` that no source contains (within rounding)."""
    known = [value for text in sources for _, value, _ in numbers_in(text)]
    bad: list[str] = []
    for written, value, abbreviated in numbers_in(answer):
        if value < floor or _is_year(value, abbreviated):
            continue
        tolerance = ROUNDED_TOLERANCE if abbreviated else EXACT_TOLERANCE
        if not any(abs(value - v) <= tolerance * max(value, v) for v in known):
            bad.append(written)
    return bad


_PROFIT_RATE = re.compile(
    r"\d[\d.,]*\s*(?:k|m|mil|millones|millón|million)?\s*(?:gp|coins|gold|monedas)?\s*"
    r"(?:/|per|por|an?)\s*(?:h|hr|hrs|hour|hora)\b",
    re.IGNORECASE,
)


def profit_rate_mentions(text: str) -> list[str]:
    """Figures phrased as a rate per hour ("30,000 gp/h", "1.2M por hora"): what an ironman's
    answer must not contain, wherever the figure came from."""
    return [match.group(0).strip() for match in _PROFIT_RATE.finditer(text)]


def drop_sentences_with(answer: str, numbers: Iterable[str]) -> str:
    """Last resort: remove every sentence or line that states one of the ungrounded numbers."""
    bad = [n for n in numbers if n]
    kept: list[str] = []
    for line in answer.split("\n"):
        pieces = re.split(r"(?<=[.!?])\s+", line)
        survivors = [piece for piece in pieces if not any(n in piece for n in bad)]
        if survivors:
            kept.append(" ".join(survivors))
    return "\n".join(kept).strip()
