"""A small, safe arithmetic evaluator (no ``eval``) for the formulas found in wiki guides.

It understands the subset of MediaWiki's ``{{#expr:}}`` that money-making guides use, plus calls to
host-provided functions such as ``gep("Fire rune", 3)`` (live price of 3 units) or
``ha("Rune scimitar")`` (high-alchemy value). Money-making guides are stored with such formulas
as text and valued at query time with live prices, so the evaluation must be total and safe.

Grammar (lowest to highest precedence)::

    expr     := additive [ "round" additive ]
    additive := term { ("+" | "-") term }
    term     := power { ("*" | "/" | "mod") power }
    power    := unary [ "^" power ]               (right associative)
    unary    := ("+" | "-") unary | primary
    primary  := NUMBER | NAME | NAME "(" [expr {"," expr}] ")" | STRING | "(" expr ")"
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

MAX_LENGTH = 4_000
MAX_DEPTH = 100

Number = float
Value = Number | str

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
      | (?P<str>"[^"]*"|'[^']*')
      | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
      | (?P<op>[-+*/^(),])
    )""",
    re.VERBOSE,
)

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
BUILTINS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "ceil": lambda x: float(math.ceil(x)),
    "floor": lambda x: float(math.floor(x)),
    "trunc": lambda x: float(math.trunc(x)),
    "int": lambda x: float(math.trunc(x)),
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


class ExprError(ValueError):
    """The formula is malformed, uses something unsupported, or cannot be evaluated."""


@dataclass(frozen=True)
class _Token:
    kind: str  # num | str | name | op | end
    text: str


def _tokenize(source: str) -> list[_Token]:
    if len(source) > MAX_LENGTH:
        raise ExprError(f"expression too long ({len(source)} chars)")
    tokens: list[_Token] = []
    pos = 0
    while pos < len(source):
        if source[pos:].strip() == "":
            break
        match = _TOKEN.match(source, pos)
        if match is None:
            raise ExprError(f"unexpected character {source[pos:].lstrip()[:1]!r}")
        kind = match.lastgroup or ""
        tokens.append(_Token(kind, match.group(kind)))
        pos = match.end()
    tokens.append(_Token("end", ""))
    return tokens


class _Parser:
    def __init__(
        self,
        tokens: list[_Token],
        variables: Mapping[str, float],
        funcs: Mapping[str, Callable[..., float]],
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.variables = variables
        self.funcs = funcs
        self.depth = 0

    # -- helpers
    @property
    def current(self) -> _Token:
        return self.tokens[self.pos]

    def _advance(self) -> _Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _is_op(self, text: str) -> bool:
        return self.current.kind == "op" and self.current.text == text

    def _is_word(self, text: str) -> bool:
        return self.current.kind == "name" and self.current.text.lower() == text

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExprError("expression nested too deeply")

    # -- grammar
    def parse(self) -> Value:
        value = self._expr()
        if self.current.kind != "end":
            raise ExprError(f"unexpected {self.current.text!r} after expression")
        return value

    def _expr(self) -> Value:
        self._enter()
        left = self._additive()
        while self._is_word("round"):
            self._advance()
            right = self._additive()
            left = round(_num(left), int(_num(right)))
        self.depth -= 1
        return left

    def _additive(self) -> Value:
        left = self._term()
        while self.current.kind == "op" and self.current.text in "+-":
            op = self._advance().text
            right = self._term()
            left = _num(left) + _num(right) if op == "+" else _num(left) - _num(right)
        return left

    def _term(self) -> Value:
        left = self._power()
        while (self.current.kind == "op" and self.current.text in "*/") or self._is_word("mod"):
            op = self._advance().text.lower()
            right = self._power()
            a, b = _num(left), _num(right)
            if op == "*":
                left = a * b
            elif b == 0:
                raise ExprError("division by zero")
            else:
                left = a / b if op == "/" else math.fmod(a, b)
        return left

    def _power(self) -> Value:
        base = self._unary()
        if self._is_op("^"):
            self._advance()
            exponent = self._power()  # right associative
            try:
                return _num(base) ** _num(exponent)
            except (OverflowError, ZeroDivisionError, ValueError) as exc:
                raise ExprError("invalid power") from exc
        return base

    def _unary(self) -> Value:
        if self._is_op("-"):
            self._advance()
            return -_num(self._unary())
        if self._is_op("+"):
            self._advance()
            return _num(self._unary())
        return self._primary()

    def _primary(self) -> Value:
        token = self._advance()
        if token.kind == "num":
            return float(token.text)
        if token.kind == "str":
            return token.text[1:-1]
        if token.kind == "op" and token.text == "(":
            value = self._expr()
            if not self._is_op(")"):
                raise ExprError("missing closing parenthesis")
            self._advance()
            return value
        if token.kind == "name":
            name = token.text
            if self._is_op("("):
                return self._call(name)
            lowered = name.lower()
            if name in self.variables:
                return self.variables[name]
            if lowered in CONSTANTS:
                return CONSTANTS[lowered]
            raise ExprError(f"unknown name {name!r}")
        raise ExprError(f"unexpected {token.text or 'end of expression'!r}")

    def _call(self, name: str) -> Value:
        function = (
            self.funcs.get(name) or self.funcs.get(name.lower()) or BUILTINS.get(name.lower())
        )
        if function is None:
            raise ExprError(f"unknown function {name!r}")
        self._advance()  # "("
        args: list[Value] = []
        if not self._is_op(")"):
            args.append(self._expr())
            while self._is_op(","):
                self._advance()
                args.append(self._expr())
        if not self._is_op(")"):
            raise ExprError("missing closing parenthesis in call")
        self._advance()
        try:
            result: Any = function(*args)
        except ExprError:
            raise
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ExprError(f"bad call to {name}(): {exc}") from exc
        return _num(result)


def _num(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExprError(f"expected a number, got {value!r}")
    return float(value)


def evaluate(
    source: str,
    *,
    variables: Mapping[str, float] | None = None,
    funcs: Mapping[str, Callable[..., float]] | None = None,
) -> float:
    """Evaluate ``source`` to a float; raises ``ExprError`` on anything it cannot handle."""
    if not source.strip():
        raise ExprError("empty expression")
    result = _Parser(_tokenize(source), variables or {}, funcs or {}).parse()
    if isinstance(result, str):
        raise ExprError("expression evaluates to text, not a number")
    if math.isnan(result) or math.isinf(result):
        raise ExprError("expression is not finite")
    return result


def calls_in(source: str) -> set[str]:
    """Names of the functions a formula calls (lets callers tell constants from price formulas)."""
    tokens = _tokenize(source)
    return {
        tokens[i].text.lower()
        for i in range(len(tokens) - 1)
        if tokens[i].kind == "name" and tokens[i + 1].kind == "op" and tokens[i + 1].text == "("
    }
