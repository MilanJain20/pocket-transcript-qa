"""Speaker-label consistency.

Checks mid-sentence speaker flips, ping-pong turns that are not backchannels,
leave-one-out voice-embedding mismatches, and summary sentences that attribute
a statement to the wrong named speaker. Passive attributions ('requested by
Alice') are not parsed. That miss is covered by the benchmark.
"""

from __future__ import annotations

import math
import re

from pocket_qa.models import Segment, SpeakerFlag, SpeakerReport
from pocket_qa.textutil import content_tokens, token_recall

BACKCHANNELS = frozenset(
    {
        "yeah",
        "yes",
        "yep",
        "ok",
        "okay",
        "right",
        "sure",
        "no",
        "mm",
        "hmm",
        "uh",
        "thanks",
        "thank",
        "you",
        "hi",
        "hello",
        "bye",
        "goodbye",
    }
)

_ATTRIBUTION = re.compile(
    r"\b([A-Z][a-z]+)\s+"
    r"(?:said|says|asked|noted|mentioned|claimed|argued|told|explained|added|"
    r"suggested|insisted|pointed out)\s+"
    r"(?:that\s+)?(.+)"
)
_ACCORDING = re.compile(r"\bAccording to ([A-Z][a-z]+),\s+(.+)", re.IGNORECASE)
_CONTINUATION = ("and ", "but ", "or ", "so ", "because ", "which ")


def check_speakers(segments: list[Segment], summary: str) -> SpeakerReport:
    flags: list[SpeakerFlag] = []
    flags.extend(_mid_sentence_flips(segments))
    flags.extend(_implausible_turns(segments))
    flags.extend(_embedding_mismatches(segments))
    flags.extend(_wrong_attributions(segments, summary))
    return SpeakerReport(flags=flags, has_issue=bool(flags))


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.?!][\"')\]]*$", text.strip()))


def _is_continuation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0].islower():
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(prefix) for prefix in _CONTINUATION)


def _mid_sentence_flips(segments: list[Segment]) -> list[SpeakerFlag]:
    flags: list[SpeakerFlag] = []
    for previous, current in zip(segments, segments[1:], strict=False):
        if previous.speaker == current.speaker:
            continue
        if _ends_sentence(previous.text):
            continue
        if not _is_continuation(current.text):
            continue
        flags.append(
            SpeakerFlag(
                kind="mid_sentence_flip",
                segment_id=current.id,
                message=(
                    f"Speaker changes from {previous.speaker_name or previous.speaker} "
                    f"to {current.speaker_name or current.speaker} inside one sentence."
                ),
                evidence=f"{previous.text.strip()} || {current.text.strip()}",
            )
        )
    return flags


def _is_backchannel(text: str) -> bool:
    tokens = re.findall(r"[a-z']+", text.lower())
    return 0 < len(tokens) <= 2 and all(token in BACKCHANNELS for token in tokens)


def _implausible_turns(segments: list[Segment]) -> list[SpeakerFlag]:
    flags: list[SpeakerFlag] = []
    run: list[Segment] = []

    def close() -> None:
        if len(run) < 6:
            return
        speakers = [segment.speaker for segment in run]
        alternates = all(speakers[index] != speakers[index - 1] for index in range(1, len(speakers)))
        contentful = [segment for segment in run if not _is_backchannel(segment.text)]
        if alternates and len(contentful) >= 4:
            flags.append(
                SpeakerFlag(
                    kind="implausible_turns",
                    segment_id=run[0].id,
                    message=(
                        f"{len(run)} consecutive one- or two-word turns alternate speakers "
                        "and are not backchannels."
                    ),
                    evidence=" | ".join(segment.text.strip() for segment in run),
                )
            )

    for segment in segments:
        micro = 0 < len(segment.text.split()) <= 2
        if micro and (not run or run[-1].speaker != segment.speaker):
            run.append(segment)
            continue
        close()
        run = [segment] if micro else []
    close()
    return flags


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _centroid(vectors: list[list[float]]) -> list[float]:
    width = len(vectors[0])
    totals = [0.0] * width
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]


def _embedding_mismatches(segments: list[Segment]) -> list[SpeakerFlag]:
    grouped: dict[str, list[tuple[int, list[float]]]] = {}
    for index, segment in enumerate(segments):
        if segment.embedding:
            grouped.setdefault(segment.speaker, []).append((index, segment.embedding))
    flags: list[SpeakerFlag] = []
    for speaker, rows in grouped.items():
        if len(rows) < 3:
            continue
        for position, (index, vector) in enumerate(rows):
            others = [item for item_position, item in rows if item_position != position]
            if len(others) < 2:
                continue
            own = _cosine(vector, _centroid(others))
            best_other = -1.0
            best_name = ""
            for other_speaker, other_rows in grouped.items():
                if other_speaker == speaker or len(other_rows) < 2:
                    continue
                similarity = _cosine(vector, _centroid([item for _, item in other_rows]))
                if similarity > best_other:
                    best_other = similarity
                    best_name = other_speaker
            if best_name and own < 0.75 and best_other - own > 0.35:
                segment = segments[index]
                flags.append(
                    SpeakerFlag(
                        kind="voice_profile_mismatch",
                        segment_id=segment.id,
                        message=(
                            f"Segment labeled {speaker} is closer to the {best_name} "
                            f"voice cluster (cosine {best_other:.2f}) than to its own "
                            f"leave-one-out centroid (cosine {own:.2f})."
                        ),
                        evidence=segment.text.strip(),
                    )
                )
    return flags


def _speaker_corpus(segments: list[Segment]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for segment in segments:
        name = segment.speaker_name or ""
        if not name or name.upper().startswith("SPEAKER_"):
            continue
        grouped.setdefault(name, []).append(segment.text)
    return {name: " ".join(parts) for name, parts in grouped.items()}


def _attributions(summary: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for line in summary.splitlines() or [summary]:
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            if not sentence:
                continue
            match = _ACCORDING.search(sentence) or _ATTRIBUTION.search(sentence)
            if not match:
                continue
            name, proposition = match.group(1), match.group(2)
            found.append((name, proposition.strip()))
    return found


def _wrong_attributions(segments: list[Segment], summary: str) -> list[SpeakerFlag]:
    corpus = _speaker_corpus(segments)
    if len(corpus) < 2 or not summary.strip():
        return []
    known = {name.lower(): name for name in corpus}
    flags: list[SpeakerFlag] = []
    for attributed, proposition in _attributions(summary):
        prop_tokens = set(content_tokens(proposition))
        if len(prop_tokens) < 2:
            continue
        scores = {
            name: token_recall(prop_tokens, set(content_tokens(text)))
            for name, text in corpus.items()
        }
        best_name, best_score = max(scores.items(), key=lambda item: item[1])
        attributed_score = scores.get(known.get(attributed.lower(), ""), 0.0)
        if attributed.lower() not in known:
            attributed_score = 0.0
        if best_score < 0.5:
            continue
        if best_name.lower() == attributed.lower():
            continue
        if attributed_score + 0.25 >= best_score:
            continue
        flags.append(
            SpeakerFlag(
                kind="wrong_speaker_attribution",
                segment_id=None,
                message=(
                    f"The summary attributes this to {attributed}, but the words "
                    f"line up with {best_name}."
                ),
                evidence=proposition,
            )
        )
    return flags
