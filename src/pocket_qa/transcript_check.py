"""Surface cues that a transcript span was not spoken.

The offline checker looks for published Whisper-style filler hallucinations,
consecutive repetition, impossible word rates, broken timestamps, and text
sitting on near-silent audio. A fluent invented sentence with normal timing
and no audio is invisible to this checker. That gap is intentional and is
measured on the synthetic benchmark.
"""

from __future__ import annotations

import re

from pocket_qa.audio import SILENCE_RMS, segment_rms
from pocket_qa.models import FabricationFlag, Segment, TranscriptReport

# Phrases widely reported as Whisper hallucinations on silence or music.
# Matching is literal and case-insensitive. "Thanks for your time" does not match.
WHISPER_FILLERS: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "subtitles by the amara.org community",
    "amara.org",
    "transcribed by https",
    "mbc news",
    "thank you for watching and don't forget to subscribe",
)

# Above this, a contentful span is faster than plausible conversational speech.
MAX_WORDS_PER_SECOND = 5.5
MIN_WORDS_FOR_RATE = 8

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def check_transcript(
    segments: list[Segment],
    audio_path: str | None = None,
) -> TranscriptReport:
    flags: list[FabricationFlag] = []
    flags.extend(_filler_flags(segments))
    flags.extend(_repetition_flags(segments))
    flags.extend(_timing_flags(segments))
    if audio_path:
        flags.extend(_silence_flags(segments, audio_path))
    ids = sorted({flag.segment_id for flag in flags})
    return TranscriptReport(
        flags=flags,
        segment_count=len(segments),
        fabricated_segment_ids=ids,
        has_issue=bool(flags),
    )


def _filler_flags(segments: list[Segment]) -> list[FabricationFlag]:
    flags: list[FabricationFlag] = []
    for segment in segments:
        lowered = segment.text.lower()
        for phrase in WHISPER_FILLERS:
            if phrase in lowered:
                flags.append(
                    FabricationFlag(
                        segment_id=segment.id or "",
                        kind="filler_hallucination",
                        message=f"Segment contains a known non-speech hallucination phrase: {phrase!r}.",
                        evidence=segment.text.strip(),
                    )
                )
                break
    return flags


def _repetition_flags(segments: list[Segment]) -> list[FabricationFlag]:
    flags: list[FabricationFlag] = []
    normalized = [_normalize(segment.text) for segment in segments]
    index = 0
    while index < len(segments):
        current = normalized[index]
        word_count = len(current.split())
        if word_count < 3:
            index += 1
            continue
        end = index + 1
        while end < len(segments) and normalized[end] == current:
            end += 1
        run = end - index
        if run >= 3:
            for segment in segments[index:end]:
                flags.append(
                    FabricationFlag(
                        segment_id=segment.id or "",
                        kind="repetition_loop",
                        message=f"The same sentence repeats {run} times in a row.",
                        evidence=segment.text.strip(),
                    )
                )
            index = end
            continue
        index += 1

    for segment in segments:
        flags.extend(_internal_loop(segment))
    return flags


def _internal_loop(segment: Segment) -> list[FabricationFlag]:
    words = _normalize(segment.text).split()
    if len(words) < 8:
        return []
    for size in range(3, 9):
        if len(words) < size * 4:
            continue
        phrase = words[:size]
        repeats = 1
        cursor = size
        while cursor + size <= len(words) and words[cursor : cursor + size] == phrase:
            repeats += 1
            cursor += size
        if repeats >= 4 and cursor >= len(words) - size:
            return [
                FabricationFlag(
                    segment_id=segment.id or "",
                    kind="repetition_loop",
                    message="A short phrase loops inside the segment.",
                    evidence=segment.text.strip(),
                )
            ]
    return []


def _timing_flags(segments: list[Segment]) -> list[FabricationFlag]:
    flags: list[FabricationFlag] = []
    for segment in segments:
        duration = segment.end - segment.start
        word_count = len(_words(segment.text))
        if duration <= 0:
            flags.append(
                FabricationFlag(
                    segment_id=segment.id or "",
                    kind="invalid_timestamp",
                    message="Segment end time is not after its start time.",
                    evidence=f"{segment.start:.2f} -> {segment.end:.2f}: {segment.text.strip()}",
                )
            )
            continue
        if word_count >= MIN_WORDS_FOR_RATE:
            rate = word_count / duration
            if rate > MAX_WORDS_PER_SECOND:
                flags.append(
                    FabricationFlag(
                        segment_id=segment.id or "",
                        kind="word_rate",
                        message=(
                            f"Word rate is {rate:.1f} words/second, above the "
                            f"{MAX_WORDS_PER_SECOND:.1f} conversational ceiling."
                        ),
                        evidence=segment.text.strip(),
                    )
                )
    return flags


def _silence_flags(segments: list[Segment], audio_path: str) -> list[FabricationFlag]:
    flags: list[FabricationFlag] = []
    for segment in segments:
        # Four raw words is enough to be a sentence. Content-token counts hide
        # short lines like "We will launch in March." behind stopwords.
        if len(_words(segment.text)) < 4:
            continue
        energy = segment_rms(audio_path, segment.start, segment.end)
        if energy is None:
            continue
        if energy < SILENCE_RMS:
            flags.append(
                FabricationFlag(
                    segment_id=segment.id or "",
                    kind="silent_audio",
                    message=(
                        f"Transcript text sits on near-silent audio "
                        f"(RMS {energy:.4f} < {SILENCE_RMS:.2f})."
                    ),
                    evidence=segment.text.strip(),
                )
            )
    return flags
