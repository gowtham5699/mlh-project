"""Small deterministic fluency heuristics used by the evaluation script."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def repetition_ratio(text: str) -> float:
    """Return the fraction of word tokens that repeat an earlier token."""
    tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text)]
    if not tokens:
        return 0.0
    return (len(tokens) - len(set(tokens))) / len(tokens)


PIRATE_MARKERS = frozenset(
    {
        "arr", "arrr", "arrrr", "ahoy", "matey", "mateys", "ye", "yer", "aye", "avast",
        "be", "hearty", "hearties", "landlubber", "landlubbers", "scallywag", "booty",
        "plunder", "doubloons", "buccaneer", "captain", "crew", "ship", "sail", "seas",
        "sea", "treasure", "shiver", "timbers", "yo", "ho", "nay", "ol'", "o'", "aboard",
    }
)


def pirate_score(text: str) -> float:
    """Return pirate-marker words per 100 word tokens (style-shift signal)."""
    tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text)]
    if not tokens:
        return 0.0
    hits = sum(token in PIRATE_MARKERS for token in tokens)
    return 100.0 * hits / len(tokens)


def non_ascii_ratio(text: str) -> float:
    """Return the fraction of characters outside printable ASCII."""
    if not text:
        return 0.0
    return sum(ord(character) > 127 for character in text) / len(text)
