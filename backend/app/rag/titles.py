"""Page-title matching for retrieval (SPEC §7 ``retrieve``).

Most questions about this game name a page: a quest, an item, a boss, an NPC. Embeddings and
keyword search both find that page's chunks only indirectly (a "requirements" chunk of Monkey
Madness II is a list of quest and skill names, which looks like nothing in particular). Spotting
the page name in the question gives retrieval a third, high-precision signal: the chunks of the
named page, ordered by how close they are to the question.

Matching is exact on normalised text: lowercase, accents and apostrophes removed, punctuation
treated as a space, and the digits 1-10 read as roman numerals ("Monkey Madness 2" is "Monkey
Madness II"; both the titles and the question go through the same folding). The longest title
wins and consumed words cannot match again, so
"Monkey Madness II" does not also select "Monkey". A trailing plural is tried when the plural
form is not a title ("gargoyles" -> "Gargoyle"). Single-word titles must be distinctive: short
words and generic game vocabulary ("Combat", "Items") would match almost every question.

A boss named in a question brings its strategy page along (``guide_pages``: "Vorkath" ->
"Vorkath/Strategies"): the main page holds the stats and drop tables, the strategy page the
gear, inventories and mechanics, and "what gear for Vorkath?" never says the word "strategies".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager

import psycopg

ConnectionFactory = Callable[[], AbstractAsyncContextManager[psycopg.AsyncConnection]]

MAX_NGRAM = 6  # "Desert Treasure II - The Fallen Empire" is the longest everyday quest name
MAX_TITLES = 8  # a question naming more pages than this is not about any one of them
MIN_SINGLE_WORD = 5
GENERIC_SINGLE_WORDS = frozenset(
    {
        *("combat", "skills", "skill", "items", "item", "level", "levels", "world", "bank", "game"),
        *("quest", "quests", "players", "player", "guide", "money", "making", "training", "update"),
        *("boss", "bosses", "monsters", "monster", "experience", "equipment", "weapons", "weapon"),
        *("armour", "armor", "members", "membership", "prices", "price", "trading"),
    }
)

GUIDE_SUBPAGES = frozenset({"strategies"})  # "<Page>/Strategies"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ROMAN = {
    "1": "i",
    "2": "ii",
    "3": "iii",
    "4": "iv",
    "5": "v",
    "6": "vi",
    "7": "vii",
    "8": "viii",
    "9": "ix",
    "10": "x",
}


def tokens(text: str) -> list[str]:
    """Lowercase words without accents or apostrophes: "Cook's Assistant" -> cooks assistant.

    A word that is just the digits 1-10 becomes its roman numeral (quests and bosses are numbered
    that way: "Dragon Slayer 2" is "Dragon Slayer II").
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().replace("'", "")
    return [_ROMAN.get(word, word) for word in _NON_ALNUM.sub(" ", folded).split()]


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


class TitleIndex:
    def __init__(self, titles: Iterable[str]) -> None:
        self._by_key: dict[str, list[str]] = {}
        self._guides: dict[str, list[str]] = {}
        for title in titles:
            parent, _, tail = title.partition("/")
            if tail and tail.lower() in GUIDE_SUBPAGES:
                self._guides.setdefault(parent, []).append(title)
            words = tokens(title)
            if not words:
                continue
            if len(words) == 1 and (
                len(words[0]) < MIN_SINGLE_WORD or words[0] in GENERIC_SINGLE_WORDS
            ):
                continue
            bucket = self._by_key.setdefault(" ".join(words), [])
            if title not in bucket:
                bucket.append(title)

    def __len__(self) -> int:
        return len(self._by_key)

    def _lookup(self, words: list[str]) -> list[str]:
        exact = self._by_key.get(" ".join(words))
        if exact:
            return exact
        return self._by_key.get(" ".join([*words[:-1], singular(words[-1])]), [])

    def guide_pages(self, title: str) -> list[str]:
        """The strategy pages of ``title`` ("Vorkath" -> ["Vorkath/Strategies"]), if any."""
        return list(self._guides.get(title, []))

    def match(self, *texts: str) -> list[str]:
        """Titles of the pages named in the texts, longest names first (at most ``MAX_TITLES``)."""
        found: list[str] = []
        for text in texts:
            words = tokens(text)
            used = [False] * len(words)
            for n in range(min(MAX_NGRAM, len(words)), 0, -1):
                for start in range(len(words) - n + 1):
                    if any(used[start : start + n]):
                        continue
                    titles = self._lookup(words[start : start + n])
                    if not titles:
                        continue
                    found += [t for t in titles if t not in found]
                    used[start : start + n] = [True] * n
        return found[:MAX_TITLES]

    @classmethod
    async def load(cls, connect: ConnectionFactory) -> TitleIndex:
        """Index every page title currently in ``documents`` (call again after an ingestion)."""
        async with connect() as conn:
            cursor = await conn.execute("select distinct page_title from documents")
            return cls(row[0] for row in await cursor.fetchall())
