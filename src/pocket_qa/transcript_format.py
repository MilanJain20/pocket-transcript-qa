"""Line-oriented transcript format used by the CLI, MCP tools, and demo UI.

Each line is:

    Name | start_seconds | end_seconds | utterance

JSON is accepted as well: a list of segment objects, or an object with
`transcript` and `summary` keys.
"""

from __future__ import annotations

import json

from pocket_qa.models import EvaluateRequest, Segment


def parse_transcript(text: str, summary: str = "") -> EvaluateRequest:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Transcript text is empty.")
    if stripped[0] in "[{":
        payload = json.loads(stripped)
        if isinstance(payload, list):
            segments = [Segment.model_validate(item) for item in payload]
            return EvaluateRequest(transcript=segments, summary=summary)
        if isinstance(payload, dict):
            if "transcript" not in payload and "segments" in payload:
                payload = {**payload, "transcript": payload["segments"]}
            merged = dict(payload)
            if summary and not merged.get("summary"):
                merged["summary"] = summary
            return EvaluateRequest.model_validate(merged)
        raise ValueError("JSON transcript must be a list or an object.")

    segments: list[Segment] = []
    for line_number, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            raise ValueError(
                f"Line {line_number} must be 'Name | start | end | utterance', got {len(parts)} fields."
            )
        name, start_text, end_text, utterance = parts
        if not utterance:
            raise ValueError(f"Line {line_number} has empty utterance text.")
        try:
            start = float(start_text)
            end = float(end_text)
        except ValueError as exc:
            raise ValueError(f"Line {line_number} has a non-numeric timestamp.") from exc
        segments.append(
            Segment(
                speaker=name,
                speaker_name=name,
                text=utterance,
                start=start,
                end=end,
            )
        )
    if not segments:
        raise ValueError("No transcript lines were found.")
    return EvaluateRequest(transcript=segments, summary=summary)


def format_transcript(segments: list[Segment]) -> str:
    lines = []
    for segment in segments:
        name = segment.speaker_name or segment.speaker
        lines.append(f"{name} | {segment.start:.2f} | {segment.end:.2f} | {segment.text}")
    return "\n".join(lines)
