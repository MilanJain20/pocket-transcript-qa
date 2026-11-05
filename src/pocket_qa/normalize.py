"""Assign stable segment ids before the checkers run."""

from __future__ import annotations

from pocket_qa.models import EvaluateRequest, Segment
from pocket_qa.paths import resolve_audio_id


def prepare_request(request: EvaluateRequest) -> tuple[list[Segment], str | None]:
    if not request.transcript:
        raise ValueError("A transcript with at least one segment is required.")
    segments: list[Segment] = []
    for index, segment in enumerate(request.transcript):
        if not segment.text.strip():
            raise ValueError(f"Segment {index} has empty text.")
        if not segment.speaker.strip():
            raise ValueError(f"Segment {index} is missing a speaker label.")
        segments.append(segment.model_copy(update={"id": segment.id or f"s{index}"}))
    audio_path = request.audio_path
    if request.audio_id:
        resolved = resolve_audio_id(request.audio_id)
        audio_path = str(resolved) if resolved else None
    return segments, audio_path
