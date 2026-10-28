from __future__ import annotations

import math

import pytest

from app.expr import ExprError, calls_in, evaluate


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("42", 42),
        (".525", 0.525),
        ("1.", 1),
        ("2+3*4", 14),  # precedence
        ("(2+3)*4", 20),
        ("10/4", 2.5),
        ("2^3^2", 512),  # right associative: 2^(3^2)
        ("2*3^2", 18),
        ("-3+5", 2),
        ("--3", 3),
        ("+4", 4),
        ("7 mod 3", 1),
        ("-7 mod 3", -1),  # fmod semantics, like #expr
        ("1234.567 round 1", 1234.6),
        ("1234.567 round 0", 1235),
        ("1234 round -2", 1200),
        ("  2  *  ( 3 + 4 ) ", 14),
        ("4*0.7852", 3.1408),
        ("4*5*.525*(60/95)", 4 * 5 * 0.525 * (60 / 95)),
        ("1e3", 1000),
        ("pi", math.pi),
        ("floor(3.9) + ceil(3.1) + abs(-2) + trunc(-3.9)", 3 + 4 + 2 - 3),
        ("min(3, 1, 2) + max(3, 1, 2)", 4),
        ("sqrt(16)", 4),
    ],
)
def test_arithmetic(source: str, expected: float) -> None:
    assert evaluate(source) == pytest.approx(expected)


def test_wiki_style_guide_quantities() -> None:
    # Real shapes from money-making guides: nested fractions, long sums.
    assert evaluate("500*(1/128)") == pytest.approx(3.90625)
    assert evaluate("((1/3)*4*(5/6)) + ((1/4)*2*(1/2))") == pytest.approx(1 / 3 * 4 * 5 / 6 + 0.25)
    assert evaluate("8 * 1/20") == pytest.approx(0.4)


def test_variables_and_host_functions() -> None:
    assert evaluate("regchance*100", variables={"regchance": 0.9767}) == pytest.approx(97.67)
    prices = {"Fire rune": 5.0, "Blood rune": 300.0}
    funcs = {"gep": lambda name, qty=1: prices[name] * qty, "ha": lambda name: 72_000.0}
    assert (
        evaluate('gep("Fire rune", 3) + gep("Blood rune") + ha("Rune scimitar")', funcs=funcs)
        == 72_315
    )
    assert evaluate("GEP('Fire rune', 2)", funcs=funcs) == 10  # function names are case-insensitive
    assert evaluate('gep("Fire rune")*0.5 + 1', funcs=funcs) == 3.5


def test_calls_in_distinguishes_constant_formulas_from_price_formulas() -> None:
    assert calls_in("4*(3/5)") == set()
    assert calls_in('GEP("Fire rune", 3) + ha("Dragon bones") * 2') == {"gep", "ha"}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("1/0", "division by zero"),
        ("5 mod 0", "division by zero"),
        ("2+", "unexpected"),
        ("(1+2", "closing parenthesis"),
        ("1 2", "after expression"),
        ("2 $ 3", "unexpected character"),
        ("foo", "unknown name"),
        ("foo(1)", "unknown function"),
        ("gep('x')", "unknown function"),  # host functions must be provided explicitly
        ('"text"', "text, not a number"),
        ("sqrt(-1)", "bad call"),
        ("9^9^9^9", "invalid power|not finite|invalid"),
        ("1" * 5_000, "too long"),
        ("(" * 300 + "1" + ")" * 300, "nested too deeply"),
    ],
)
def test_malformed_or_unsafe_input_raises_exprerror(source: str, message: str) -> None:
    with pytest.raises(ExprError, match=message):
        evaluate(source)


def test_cannot_reach_python_internals() -> None:
    for hostile in (
        "__import__('os').system('true')",
        "().__class__",
        "open('/etc/passwd')",
        "a.b",
    ):
        with pytest.raises(ExprError):
            evaluate(hostile)


def test_function_errors_are_wrapped() -> None:
    def boom(name: str) -> float:
        raise KeyError(name)

    with pytest.raises(
        KeyError
    ):  # only arithmetic/value errors are wrapped; lookups are the caller's
        evaluate('gep("x")', funcs={"gep": boom})
    with pytest.raises(ExprError, match="bad call"):
        evaluate(
            "gep(1, 2, 3)", funcs={"gep": lambda a: a}
        )  # wrong arity -> TypeError -> ExprError
