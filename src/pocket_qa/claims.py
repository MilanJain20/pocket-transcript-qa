"""Split a summary into atomic claims.

The splitter is deterministic and conservative. It breaks markdown blocks and
sentences. It does not split on 'and', because that cuts real noun phrases
('Alice and Bob') as often as it cuts compound claims.
"""

from __future__ import annotations

import re

from pocket_qa.textutil import content_tokens, numbers

_ABBREVIATIONS = ("Mr.", "Ms.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "e.g.", "i.e.")
_DOT = "\u2219"


def _protect(text: str) -> str:
    protected = re.sub(r"(?<=\d)\.(?=\d)", _DOT, text)
    for abbr in _ABBREVIATIONS:
        protected = re.sub(re.escape(abbr), abbr[:-1] + _DOT, protected, flags=re.IGNORECASE)
    return protected


def _restore(text: str) -> str:
    return text.replace(_DOT, ".")


def _sentences(line: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _protect(line))
    sentences: list[str] = []
    for part in parts:
        for piece in re.split(r";\s+", _restore(part)):
            cleaned = piece.strip(" \t-*")
            if cleaned:
                sentences.append(cleaned)
    return sentences


def split_claims(summary: str) -> list[str]:
    """Return claim strings in reading order."""
    claims: list[str] = []
    for raw_line in summary.replace("\r\n", "\n").split("\n"):
        line = re.sub(r"^#{1,6}\s+", "", raw_line).strip()
        line = re.sub(r"^[-*]\s+", "", line).strip()
        if not line:
            continue
        for sentence in _sentences(line):
            if len(content_tokens(sentence)) >= 3 or numbers(sentence):
                if sentence[-1] not in ".!?":
                    sentence = f"{sentence}."
                claims.append(sentence)
    return claims
