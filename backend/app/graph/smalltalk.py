"""Greetings, thanks and goodbyes: answered with a fixed message, no search and no model call.

Jev's router has no "hello" intent and a bare greeting is not an OSRS question, so it used to land
on ``out_of_domain`` and be refused as if it were a request to automate the game. A message made
*only* of these words is recognised here, before anything else runs; "thanks, and what about the
whip?" has other words in it and goes through the normal path.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-záéíóúüñ']+")

_GREETING = {
    "en": {
        "hello",
        "hi",
        "hey",
        "heya",
        "howdy",
        "yo",
        "greetings",
        "sup",
        "morning",
        "afternoon",
        "evening",
    },
    "es": {
        "hola",
        "holaa",
        "buenas",
        "ey",
        "che",
        "saludos",
        "buenos",
        "buen",
        "dia",
        "dias",
        "días",
        "tardes",
        "noches",
    },
}
_THANKS = {
    "en": {
        "thanks",
        "thank",
        "thx",
        "ty",
        "cheers",
        "ok",
        "okay",
        "great",
        "perfect",
        "cool",
        "nice",
    },
    "es": {"gracias", "genial", "dale", "listo", "perfecto", "buenisimo", "buenísimo", "ok", "mil"},
}
_BYE = {
    "en": {"bye", "goodbye", "later", "cya"},
    "es": {"chau", "chao", "adios", "adiós", "hasta", "luego"},
}
_FILLER = {
    "en": {
        "there",
        "all",
        "everyone",
        "good",
        "a",
        "lot",
        "you",
        "so",
        "much",
        "very",
        "see",
        "ya",
        "u",
    },
    "es": {
        "a",
        "todos",
        "muchas",
        "muchísimas",
        "muy",
        "bien",
        "que",
        "qué",
        "tal",
        "buen",
        "día",
        "dia",
        "te",
        "veo",
        "vemos",
        "me",
    },
}
MAX_WORDS = 5


def classify_smalltalk(text: str) -> tuple[str, str] | None:
    """``(kind, language)`` when the whole message is a greeting / thanks / goodbye, else ``None``.

    ``kind`` is "greeting", "thanks" or "bye"; the language comes from the words used.
    """
    words = _TOKEN.findall(text.lower())
    if not words or len(words) > MAX_WORDS:
        return None
    for language in ("es", "en"):
        vocabulary = _GREETING[language] | _THANKS[language] | _BYE[language] | _FILLER[language]
        # "ok" and "que" exist in both: the first language that covers every word wins
        if not all(w in vocabulary for w in words):
            continue
        for kind, table in (("bye", _BYE), ("thanks", _THANKS), ("greeting", _GREETING)):
            if any(w in table[language] for w in words):
                # "ok"/"mil" alone are acknowledgements, fine as "thanks"
                return kind, language
    return None
