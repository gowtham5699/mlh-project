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


def non_ascii_ratio(text: str) -> float:
    """Return the fraction of characters outside printable ASCII."""
    if not text:
        return 0.0
    return sum(ord(character) > 127 for character in text) / len(text)
