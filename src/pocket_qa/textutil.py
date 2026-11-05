"""Token, number, and polarity helpers shared by the offline checkers."""

from __future__ import annotations

import re

# Closed-class glue. Polarity words (not, never, more, less, before, after)
# stay out of this set so contradiction checks can still see them.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "for",
        "from",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "should",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|dont|don't|didn't|didnt|won't|wont|"
    r"isn't|isnt|aren't|arent|wasn't|wasnt|weren't|werent|no longer|unable|failed to)\b",
    re.IGNORECASE,
)

# A small closed lexicon. This is not a paraphrase model. Pairs that are not
# listed are invisible to the offline contradiction check.
ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("accept", "reject"),
    ("accepted", "rejected"),
    ("approve", "deny"),
    ("approved", "denied"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("higher", "lower"),
    ("include", "exclude"),
    ("pass", "fail"),
    ("passed", "failed"),
    ("buy", "sell"),
    ("start", "stop"),
    ("always", "never"),
    ("earlier", "later"),
    ("above", "below"),
    ("win", "lose"),
    ("add", "remove"),
    ("before", "after"),
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def content_tokens(text: str) -> list[str]:
    tokens = _WORD_RE.findall(text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def raw_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def numbers(text: str) -> set[str]:
    found: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace(",", "").replace("%", "")
        if cleaned:
            found.add(cleaned)
    return found


def has_negation(text: str) -> bool:
    return NEGATION_RE.search(text) is not None


def antonym_conflict(claim: str, evidence: str) -> str | None:
    """Return a short description when a listed antonym pair is split across texts."""
    claim_words = raw_words(claim)
    evidence_words = raw_words(evidence)
    for left, right in ANTONYM_PAIRS:
        claim_has_left = left in claim_words
        claim_has_right = right in claim_words
        evidence_has_left = left in evidence_words
        evidence_has_right = right in evidence_words
        if claim_has_left and claim_has_right:
            continue
        if evidence_has_left and evidence_has_right:
            continue
        crossed = (claim_has_left and evidence_has_right) or (
            claim_has_right and evidence_has_left
        )
        if crossed:
            return f"{left}/{right}"
    return None


def token_recall(needles: set[str], haystack: set[str]) -> float:
    if not needles:
        return 0.0
    return len(needles & haystack) / len(needles)
