"""Milestone 1 smoke test (SPEC §13.1).

1. Classifies sample phrases with Jev through the OpenRouter Decisions API. Each intent has an
   English and a Spanish phrase, so Jev's weaker non-English accuracy (documented by TypeSafe)
   is measured from day one rather than assumed.
2. Fetches the current prices of 3 items from the Wiki prices API (no key needed).

Run from ``backend/``:

    uv run python -m scripts.m1_smoke                # needs OPEN_ROUTER_KEY in env or .env
    uv run python -m scripts.m1_smoke --prices-only  # no key needed

The key is read from the environment/.env here because this is a developer script (SPEC §11);
the API server never does that. Paid calls are tracked and capped (``--max-spend``).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

import httpx

from app.clients.jev import JevClient
from app.clients.prices import PricesClient
from app.config import dev_api_key, get_settings
from app.devtools import BudgetExceeded, SpendGuard
from app.errors import AgentError
from app.graph.intents import INTENT_CRITERIA, ROUTER_INSTRUCTIONS

# (message, expected intent, language)
SAMPLES: list[tuple[str, str, str]] = [
    ("How do I start Monkey Madness II and what do I need first?", "quest", "en"),
    ("¿Qué necesito para empezar Dragon Slayer II?", "quest", "es"),
    ("What's a good money maker for a mid-level account?", "moneymaking", "en"),
    ("¿Qué moneymaker me conviene con 80 de Slayer y 1M de GP?", "moneymaking", "es"),
    ("Is flipping dragon bones profitable right now?", "flipping", "en"),
    ("¿Qué ítems puedo flipear en el GE con 5M de GP?", "flipping", "es"),
    ("What is the price of an abyssal whip?", "price_lookup", "en"),
    ("¿A cuánto está la Abyssal whip en el GE?", "price_lookup", "es"),
    ("What's the best way to train Prayer?", "wiki_general", "en"),
    ("¿Cómo funciona el special attack del dragon dagger?", "wiki_general", "es"),
    ("What's the weather like today in Buenos Aires?", "out_of_domain", "en"),
    ("¿Me ayudás a escribir un script en Python?", "out_of_domain", "es"),
]

PRICE_ITEMS = ["Abyssal whip", "Dragon bones", "Old school bond"]


def _fmt_ts(ts: int | None) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%H:%M:%S") if ts else "-"


async def run_jev(http: httpx.AsyncClient, key: str, max_spend: float) -> bool:
    jev = JevClient(key, http=http)
    guard = SpendGuard(max_spend)
    print(f"\n== Jev routing ({get_settings().jev_model}) ==")
    hits = {"en": [0, 0], "es": [0, 0]}
    served_by = ""
    tokens = 0

    async def classify(message: str):
        state = {
            "user_message": message,
            "standalone_question": "(same as user_message)",
            "conversation_summary": "(none: first message)",
            "player_summary": "(unknown)",
        }
        return await jev.choose(state, ROUTER_INSTRUCTIONS, INTENT_CRITERIA)

    results = await asyncio.gather(*(classify(m) for m, _, _ in SAMPLES), return_exceptions=True)
    try:
        for (message, expected, lang), result in zip(SAMPLES, results, strict=True):
            if isinstance(result, BaseException):
                print(f"  [{lang}] ERROR for {message!r}: {result}")
                if isinstance(result, AgentError) and result.http_status in (401, 402):
                    return False
                hits[lang][1] += 1
                continue
            answer, decision = result
            guard.add(decision.usage.cost)
            tokens += decision.usage.input_tokens
            served_by = decision.model
            ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])[:2]
            top2 = ", ".join(f"{name}={prob:.2f}" for name, prob in ranked)
            ok = answer.choice == expected
            hits[lang][0] += ok
            hits[lang][1] += 1
            conf = f"{answer.confidence:.2f}" if answer.confidence is not None else " n/a"
            print(
                f"  [{lang}] {'OK ' if ok else 'MISS'} expected={expected:<13} "
                f"got={answer.choice:<13} conf={conf} top2: {top2:<34}| {message}"
            )
    except BudgetExceeded as exc:
        print(f"  STOPPED: {exc}")
        return False
    for lang, (ok, n) in hits.items():
        print(f"  accuracy[{lang}]: {ok}/{n}")
    print(f"  served by: {served_by or '?'} | input tokens: {tokens} | {guard.summary()}")
    return True


async def run_prices(http: httpx.AsyncClient) -> None:
    print("\n== Current prices (Wiki prices API) ==")
    prices = PricesClient(http=http)
    mapping, latest, hourly = await asyncio.gather(
        prices.mapping(), prices.latest(), prices.one_hour()
    )
    by_name = {m.name.lower(): m for m in mapping.data.values()}
    print(f"  mapping: {len(mapping.data)} items | latest: {len(latest.data)} items")
    for name in PRICE_ITEMS:
        meta = by_name.get(name.lower())
        if meta is None:
            print(f"  {name}: not found in /mapping")
            continue
        p = latest.data.get(meta.id)
        h = hourly.data.get(meta.id)
        vol = f"{h.high_volume + h.low_volume:,}" if h else "n/a"
        if p is None:
            print(f"  {meta.name} (id {meta.id}): no price data")
            continue
        print(
            f"  {meta.name} (id {meta.id}): "
            f"buy-now(high)={p.high:,} @{_fmt_ts(p.high_time)}Z | "
            f"sell-now(low)={p.low:,} @{_fmt_ts(p.low_time)}Z | "
            f"1h volume={vol} | buy limit={meta.limit}"
        )
    print(f"  fetched at {latest.fetched_at:%Y-%m-%d %H:%M:%S}Z (stale={latest.stale})")


async def main(prices_only: bool, max_spend: float) -> int:
    async with httpx.AsyncClient() as http:
        await run_prices(http)
        if prices_only:
            return 0
        key = dev_api_key()
        if not key:
            print(
                "\nNo OpenRouter key found (OPEN_ROUTER_KEY in the environment or .env): "
                "skipping the Jev part.\n"
                "Re-run with the key, or use --prices-only to silence this.",
                file=sys.stderr,
            )
            return 2
        return 0 if await run_jev(http, key, max_spend) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--max-spend", type=float, default=0.25, help="USD cap for this run")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.prices_only, args.max_spend)))
