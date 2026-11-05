"""Convert documented Pocket payloads into an EvaluateRequest."""

from __future__ import annotations

from pocket_qa.models import EvaluateRequest, Segment
from pocket_qa.pocket.schemas import (
    PocketRecording,
    PocketSummarization,
    PocketTranscriptSegment,
    PocketWebhookPayload,
)


def _segment_from_pocket(segment: PocketTranscriptSegment, index: int) -> Segment | None:
    text = (segment.text or segment.original_text or "").strip()
    if not text:
        return None
    speaker = segment.speaker or segment.speaker_name or f"SPEAKER_{index:02d}"
    return Segment(
        id=f"s{index}",
        speaker=speaker,
        speaker_name=segment.speaker_name,
        text=text,
        start=segment.start,
        end=segment.end,
    )


def _segments_from_transcript(transcript: object) -> list[Segment]:
    raw: list[PocketTranscriptSegment] = []
    if isinstance(transcript, list):
        raw = transcript
    elif transcript is not None and hasattr(transcript, "segments"):
        raw = list(transcript.segments)
    segments: list[Segment] = []
    for index, item in enumerate(raw):
        converted = _segment_from_pocket(item, index)
        if converted is not None:
            segments.append(converted)
    return segments


def summary_from_summarizations(summarizations: dict[str, PocketSummarization] | None) -> str:
    """Prefer v2.summary.markdown, which is the shape in the public webhook example.

    The GET recording example shows `summarizations: null`, so a missing summary
    is returned as an empty string rather than guessed from another field.
    """
    if not summarizations:
        return ""
    items = list(summarizations.values())
    items.sort(key=lambda item: item.updated_at or "", reverse=True)
    for item in items:
        if item.v2 is None or item.v2.summary is None:
            continue
        body = item.v2.summary
        if body.markdown and body.markdown.strip():
            return body.markdown.strip()
        if body.bullet_points:
            lines = [f"- {bullet}" for bullet in body.bullet_points if bullet.strip()]
            if lines:
                return "\n".join(lines)
    return ""


def summary_from_recording(recording: PocketRecording) -> str:
    if recording.summary and recording.summary.markdown:
        return recording.summary.markdown.strip()
    return summary_from_summarizations(recording.summarizations)


def request_from_recording(recording: PocketRecording, provider: str = "offline") -> EvaluateRequest:
    segments = _segments_from_transcript(recording.transcript)
    if recording.speakers:
        names = {key: value.name for key, value in recording.speakers.items() if value.name}
        patched: list[Segment] = []
        for segment in segments:
            name = segment.speaker_name or names.get(segment.speaker)
            patched.append(segment.model_copy(update={"speaker_name": name}))
        segments = patched
    return EvaluateRequest(
        transcript=segments,
        summary=summary_from_recording(recording),
        provider=provider,  # type: ignore[arg-type]
        recording_id=recording.id,
        title=recording.title,
    )


def request_from_webhook(payload: PocketWebhookPayload, provider: str = "offline") -> EvaluateRequest:
    segments = _segments_from_transcript(payload.transcript)
    if payload.speakers:
        names = {key: value.name for key, value in payload.speakers.items() if value.name}
        patched = []
        for segment in segments:
            name = segment.speaker_name or names.get(segment.speaker)
            patched.append(segment.model_copy(update={"speaker_name": name}))
        segments = patched
    recording = payload.recording
    return EvaluateRequest(
        transcript=segments,
        summary=summary_from_summarizations(payload.summarizations),
        provider=provider,  # type: ignore[arg-type]
        recording_id=recording.id if recording else None,
        title=recording.title if recording else None,
    )
