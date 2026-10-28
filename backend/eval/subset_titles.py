"""The pages of an experiment index (docs/rag-quality.md): a faithful, smaller copy of the corpus.

Embedding the whole wiki with a large model takes hours; comparing embedding models does not need
it. This writes the titles of a subset that keeps what makes retrieval hard:

* every page the benchmark's gold labels point to;
* the pages the current retriever returns for each benchmark question (the real confusions:
  "Moss giant" vs "Mossy key" vs "Bryophyta");
* a seeded random sample of everything else, stratified by page type.

    uv run python -m eval.subset_titles --out data/eval/subset-titles.txt
    SUPABASE_DB_URL=postgresql://127.0.0.1:5432/osrs_exp_x EMBEDDING_MODEL=... EMBEDDING_DIM=... \\
        uv run python -m scripts.ingest load --scope v0 --titles-file data/eval/subset-titles.txt

Run it against a full index (it asks that index for the hard negatives). The same file is then used
for every model, so they are compared on the same corpus. A subset has fewer distractors than the
full corpus, so its absolute numbers are higher; only the comparison between models is meaningful.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.db import MissingDatabaseUrl, async_connection, require_db_url
from app.rag.embed import Embedder
from app.rag.retrieve import Retriever
from app.rag.titles import TitleIndex
from eval.retrieval_bench import CASES_PATH, load_cases

SAMPLE_PAGES = {"item": 1200, "npc": 250, "other": 500, "quest": 60, "skill_guide": 50, "mmg": 150}
HARD_NEGATIVES_K = 25  # candidates asked per question


def cached_pages(data_dir: Path) -> dict[str, str]:
    """Title -> page type of every cached page (the first category that lists a page wins)."""
    pages: dict[str, str] = {}
    for path in sorted((data_dir / "wiki").glob("*.jsonl")):
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                pages.setdefault(record["title"], record["page_type"])
    return pages


async def hard_negatives(cases_path: Path, k: int) -> set[str]:
    settings = get_settings()
    url = require_db_url(settings)
    embedder = Embedder(settings)
    titles = await TitleIndex.load(lambda: async_connection(url))
    retriever = Retriever(lambda: async_connection(url), embedder.embed_query, settings, titles)
    found: set[str] = set()
    for case in load_cases(cases_path):
        found.update(p.page_title for p in await retriever.search(case.q, k=k))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/eval/subset-titles.txt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    settings = get_settings()
    try:
        pages = cached_pages(Path(settings.data_dir))
        hard = asyncio.run(hard_negatives(args.cases, HARD_NEGATIVES_K))
    except MissingDatabaseUrl as exc:
        print(exc, file=sys.stderr)
        return 2
    gold: set[str] = set()
    for case in load_cases(args.cases):
        for entry in case.gold:
            gold.update(t for t in pages if entry.page_matches(t))
    chosen = gold | {t for t in hard if t in pages}
    rng = random.Random(args.seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for title, page_type in sorted(pages.items()):
        if title not in chosen:
            by_type[page_type].append(title)
    for page_type, titles in by_type.items():
        rng.shuffle(titles)
        chosen.update(titles[: SAMPLE_PAGES.get(page_type, 0)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(chosen)) + "\n", encoding="utf-8")
    print(
        f"{len(chosen)} pages -> {args.out} (gold {len(gold)}, hard negatives "
        f"{len(chosen & hard) - len(chosen & gold & hard)}, of {len(pages)} cached)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
