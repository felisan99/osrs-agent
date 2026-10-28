"""Retrieval benchmark (docs/rag-quality.md): is the right chunk among the top-k candidates?

Runs ``eval/retrieval.yaml`` through the real retriever (embedder + Postgres + hybrid search). No
LLM, no Jev, no key and no cost: a run takes seconds, so chunking and embedder changes can be
judged with numbers instead of impressions.

    uv run python -m eval.retrieval_bench                      # table + failures
    uv run python -m eval.retrieval_bench --label e5-small --json   # also saved to data/eval/
    uv run python -m eval.retrieval_bench --only drops-moss-giant-list --verbose
    uv run python -m eval.retrieval_bench --check-gold         # labels vs the cached wikitext
    uv run python -m eval.retrieval_bench --compare a.json b.json

The database and the embedder come from the usual settings (``SUPABASE_DB_URL``,
``EMBEDDING_MODEL``), so an experiment is a different environment, not different code.

Three hit levels are reported (rank of the first matching candidate):

* ``strict``: right page AND right section AND the chunk text holds the answer;
* ``content``: right page and the chunk text holds the answer, whatever its section;
* ``page``: any chunk of the right page.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import get_settings
from app.db import MissingDatabaseUrl, async_connection, require_db_url
from app.rag.embed import Embedder
from app.rag.retrieve import Passage, Retriever
from app.rag.titles import TitleIndex

CASES_PATH = Path(__file__).with_name("retrieval.yaml")
KS = (1, 3, 5, 10, 20)


class Gold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str | None = None
    page_re: str | None = None
    section: str | None = None
    has: list[str] = Field(default_factory=list)
    source: list[str] = Field(default_factory=list)

    @field_validator("page_re", "section")
    @classmethod
    def _regex(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value

    @field_validator("has", "source")
    @classmethod
    def _regexes(cls, values: list[str]) -> list[str]:
        for value in values:
            re.compile(value)
        return values

    @model_validator(mode="after")
    def _one_page(self) -> Gold:
        if (self.page is None) == (self.page_re is None):
            raise ValueError("a gold entry needs exactly one of page / page_re")
        return self

    def page_matches(self, title: str) -> bool:
        if self.page is not None:
            return title == self.page
        return re.search(self.page_re or "", title) is not None

    def content_matches(self, passage: Passage) -> bool:
        return self.page_matches(passage.page_title) and all(
            re.search(pattern, passage.content) for pattern in self.has
        )

    def strict_matches(self, passage: Passage) -> bool:
        if self.section is not None and re.search(self.section, passage.section) is None:
            return False
        return self.content_matches(passage)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    q: str
    gold: list[Gold] = Field(min_length=1)
    notes: str = ""


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = [Case.model_validate(item) for item in raw["cases"]]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return cases


def rank_of(passages: list[Passage], test: Any) -> int | None:
    return next((i for i, p in enumerate(passages, 1) if test(p)), None)


@dataclass
class Result:
    case: Case
    passages: list[Passage]
    strict: int | None
    content: int | None
    page: int | None
    seconds: float


def evaluate(case: Case, passages: list[Passage], seconds: float) -> Result:
    return Result(
        case,
        passages,
        rank_of(passages, lambda p: any(g.strict_matches(p) for g in case.gold)),
        rank_of(passages, lambda p: any(g.content_matches(p) for g in case.gold)),
        rank_of(passages, lambda p: any(g.page_matches(p.page_title) for g in case.gold)),
        seconds,
    )


def hit_rate(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def label(passage: Passage) -> str:
    return passage.page_title + (f" > {passage.section}" if passage.section else " (lead)")


def print_summary(results: list[Result], k: int) -> None:
    print(f"\n{len(results)} cases, top {k} candidates per question\n")
    header = "".join(f"{'@' + str(n):>7}" for n in KS if n <= k)
    print(f"{'level':10}{header}{'MRR':>8}")
    for name in ("strict", "content", "page"):
        ranks = [getattr(r, name) for r in results]
        cells = "".join(f"{hit_rate(ranks, n):>7.0%}" for n in KS if n <= k)
        print(f"{name:10}{cells}{mrr(ranks):>8.3f}")
    print(f"\n{'kind':14}{'n':>4}{'strict@5':>10}{'strict@' + str(k):>11}{'MRR':>8}")
    by_kind: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        by_kind[result.case.kind].append(result)
    for kind, group in by_kind.items():
        ranks = [r.strict for r in group]
        print(
            f"{kind:14}{len(group):>4}{hit_rate(ranks, 5):>10.0%}{hit_rate(ranks, k):>11.0%}"
            f"{mrr(ranks):>8.3f}"
        )


def print_failures(results: list[Result], k: int, verbose: bool) -> None:
    missed = [r for r in results if r.strict is None or r.strict > 5 or verbose]
    if not missed:
        return
    print(f"\ncases whose strict hit is outside the top 5{' (all, --verbose)' if verbose else ''}:")
    for result in missed:
        where = f"rank {result.strict}" if result.strict else f"not in the top {k}"
        extra = f" | content hit: {result.content}" if result.strict != result.content else ""
        print(f"  [{result.case.id}] {where}{extra} | page hit: {result.page}")
        print(f"      {result.case.q}")
        for i, passage in enumerate(result.passages[:3], 1):
            ranks = (passage.vector_rank, passage.keyword_rank, passage.title_rank)
            via = "/".join(f"{tag}{rank or '-'}" for tag, rank in zip("vkt", ranks, strict=True))
            print(f"      {i}. {label(passage)}  ({via})")


def save(results: list[Result], path: Path, tag: str, k: int, model: str, database: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": tag,
        "created": datetime.now(UTC).isoformat(),
        "model": model,
        "database": database,
        "k": k,
        "cases": [
            {
                "id": r.case.id,
                "kind": r.case.kind,
                "strict": r.strict,
                "content": r.content,
                "page": r.page,
                "top": [label(p) for p in r.passages[:5]],
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path}")


def compare(first: Path, second: Path) -> None:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (first, second))
    by_id = {c["id"]: c for c in b["cases"]}
    print(f"A = {a['label']} ({a['model']})   B = {b['label']} ({b['model']})\n")
    better, worse = [], []
    for case in a["cases"]:
        other = by_id.get(case["id"])
        if other is None:
            continue
        ra = case["strict"] or 10**6
        rb = other["strict"] or 10**6
        if rb < ra:
            better.append((case["id"], case["strict"], other["strict"]))
        elif rb > ra:
            worse.append((case["id"], case["strict"], other["strict"]))
    for title, rows in (("better in B", better), ("worse in B", worse)):
        print(f"{title}: {len(rows)}")
        for case_id, ra, rb in rows:
            print(f"  {case_id:42} strict rank {ra or '-'} -> {rb or '-'}")
    for name, data in (("A", a), ("B", b)):
        ranks = [c["strict"] for c in data["cases"]]
        cells = " ".join(f"@{n} {hit_rate(ranks, n):.0%}" for n in KS if n <= data["k"])
        print(f"{name}: strict {cells} MRR {mrr(ranks):.3f}")


def cached_pages() -> dict[str, str]:
    """Title -> raw wikitext for every cached page (the first category that lists it wins)."""
    pages: dict[str, str] = {}
    pattern = str(Path(get_settings().data_dir) / "wiki" / "*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                pages.setdefault(record["title"], record["wikitext"])
    return pages


def check_gold(cases: list[Case]) -> int:
    pages = cached_pages()
    problems: list[str] = []
    for case in cases:
        for gold in case.gold:
            titles = [t for t in pages if gold.page_matches(t)]
            if not titles:
                problems.append(f"[{case.id}] no cached page matches {gold.page or gold.page_re!r}")
                continue
            for pattern in gold.source:
                if not any(re.search(pattern, pages[t], re.S) for t in titles):
                    problems.append(
                        f"[{case.id}] {titles[0]!r}: source /{pattern}/ not in the wikitext"
                    )
    for problem in problems:
        print(problem)
    print(
        f"{len(cases)} cases checked against {len(pages)} cached pages: "
        f"{'OK' if not problems else f'{len(problems)} problem(s)'}"
    )
    return 1 if problems else 0


async def run(args: argparse.Namespace, cases: list[Case]) -> int:
    settings = get_settings()
    try:
        url = require_db_url(settings)
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    embedder = Embedder(settings)
    _ = embedder.model  # load once, before timing
    titles = await TitleIndex.load(lambda: async_connection(url))
    retriever = Retriever(lambda: async_connection(url), embedder.embed_query, settings, titles)
    results: list[Result] = []
    for case in cases:
        started = time.perf_counter()
        passages = await retriever.search(case.q, k=args.k)
        results.append(evaluate(case, passages, time.perf_counter() - started))
    print(f"model: {settings.embedding_model} | database: {url.rsplit('/', 1)[-1]}")
    print_summary(results, args.k)
    print_failures(results, args.k, args.verbose)
    if args.json:
        tag = args.label or settings.embedding_model.rsplit("/", 1)[-1]
        out = Path(settings.data_dir) / "eval" / f"retrieval-{tag}.json"
        save(results, out, tag, args.k, settings.embedding_model, url.rsplit("/", 1)[-1])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--k", type=int, default=get_settings().crag_topk, help="candidates")
    parser.add_argument("--label", help="name for the saved JSON (default: the embedding model)")
    parser.add_argument("--json", action="store_true", help="save the ranks to data/eval/")
    parser.add_argument("--verbose", action="store_true", help="list every case, not just misses")
    parser.add_argument("--check-gold", action="store_true", help="validate labels, no database")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return 0
    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c.id in args.only]
    if args.check_gold:
        return check_gold(cases)
    return asyncio.run(run(args, cases))


if __name__ == "__main__":
    sys.exit(main())
