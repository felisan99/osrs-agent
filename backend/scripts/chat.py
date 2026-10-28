"""Talk to the agent from the terminal (M5 acceptance: a conversation whose turn 2 needs turn 1).

    uv run python -m scripts.chat --profile main_midgame \\
        --say "How do I start Monkey Madness II?" --say "¿y si no tengo el m'speak amulet?"
    uv run python -m scripts.chat                # interactive; an empty line ends it

The runs use the development key from ``OPEN_ROUTER_KEY`` (never the server's: the server has none)
and stop at ``--max-spend``. ``--profile`` loads a player snapshot from ``eval/questions.yaml``;
without it the agent has no information about the character, like a plugin that sent none.
Every turn prints what the graph decided: intent and probabilities, searches, tools, cost.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid

import yaml

from app.api.schemas import PlayerState
from app.config import dev_api_key, get_settings
from app.devtools import BudgetExceeded, SpendGuard
from app.errors import AgentError
from app.service import AgentService
from eval.cases import QUESTIONS_PATH


def load_profile(name: str | None) -> PlayerState | None:
    if not name:
        return None
    profiles = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))["profiles"]
    if name not in profiles:
        raise SystemExit(f"unknown profile {name!r}; choose from {', '.join(profiles)}")
    return PlayerState.model_validate(profiles[name])


async def main(args: argparse.Namespace) -> int:
    key = dev_api_key()
    if not key:
        print("No OPEN_ROUTER_KEY in the environment or .env", file=sys.stderr)
        return 2
    player = load_profile(args.profile)
    thread = args.thread or f"chat-{uuid.uuid4().hex[:12]}"
    guard = SpendGuard(args.max_spend)
    messages = list(args.say)
    print(f"thread {thread} | player: {args.profile or 'none'}\n")
    async with AgentService.open(get_settings()) as service:
        turn = 0
        while True:
            if messages:
                text = messages.pop(0)
                print(f"you> {text}")
            elif args.say:
                break
            else:
                try:
                    text = (await asyncio.to_thread(input, "you> ")).strip()
                except EOFError:
                    break
                if not text:
                    break
            started = time.perf_counter()
            try:
                result = await service.chat(
                    thread_id=thread, message=text, api_key=key, player_state=player
                )
            except AgentError as exc:
                print(f"[error {exc.code}] {exc.message}\n")
                return 1
            cost = result.usage.get("llm_cost", 0.0) + result.usage.get("jev_cost", 0.0)
            try:
                guard.add(cost)
            except BudgetExceeded as exc:
                print(f"STOPPED: {exc}")
                return 1
            turn += 1
            top = sorted(result.route_probs.items(), key=lambda kv: -kv[1])[:3]
            print(f"\nagent> {result.answer}\n")
            if result.sources:
                print("  sources: " + " | ".join(s["title"] for s in result.sources))
            print(
                f"  [{result.intent}; route {', '.join(f'{k} {v:.2f}' for k, v in top)}; "
                f"searches {result.retrieval_iterations}; tools {result.tools_called or '-'}; "
                f"llm calls {int(result.usage.get('llm_calls', 0))}, "
                f"jev calls {int(result.usage.get('jev_calls', 0))}; ${cost:.5f}; "
                f"{time.perf_counter() - started:.1f}s]\n"
            )
    print(guard.summary())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="player snapshot from eval/questions.yaml")
    parser.add_argument("--thread", help="reuse a thread id to continue a conversation")
    parser.add_argument("--say", action="append", default=[], help="a message (repeatable)")
    parser.add_argument("--max-spend", type=float, default=0.30, help="USD cap for this run")
    sys.exit(asyncio.run(main(parser.parse_args())))
