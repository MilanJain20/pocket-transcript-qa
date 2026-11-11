"""Load transcripts and summaries that stay on this machine.

The folder layout is documented in the README. Real exports belong under
`recordings/`, which is gitignored. This module does not upload anything
and does not call Pocket.
"""

from __future__ import annotations

import json
from pathlib import Path

from pocket_qa.models import EvaluateRequest, ProviderName
from pocket_qa.pocket.normalize import request_from_recording, request_from_webhook
from pocket_qa.pocket.schemas import PocketRecording, PocketWebhookPayload
from pocket_qa.transcript_format import parse_transcript

LOCAL_NOTICE = (
    "Recordings stay on this machine. This command does not upload audio, "
    "transcripts, or summaries, and it does not call Pocket. Do not commit "
    "the input files or a report that quotes them."
)

TRANSCRIPT_NAMES = ("transcript.txt", "transcript.md", "transcript.json")
SUMMARY_NAMES = ("summary.txt", "summary.md", "summary.json")
AUDIO_NAMES = ("audio.wav", "audio.wave")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 text.") from exc


def _read_summary(path: Path) -> str:
    raw = _read_text(path).strip()
    if not raw:
        return ""
    if path.suffix.lower() != ".json" and not raw.startswith(("{", "[")):
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        lines = [f"- {item}" for item in payload if isinstance(item, str) and item.strip()]
        return "\n".join(lines)
    if isinstance(payload, dict):
        for key in ("markdown", "summary", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                markdown = value.get("markdown") or value.get("text")
                if isinstance(markdown, str) and markdown.strip():
                    return markdown.strip()
    raise ValueError(f"{path} is JSON but has no summary text.")


def _looks_like_recording(payload: dict) -> bool:
    transcript = payload.get("transcript")
    if isinstance(transcript, dict) and "segments" in transcript:
        return True
    if "summarizations" in payload and "transcript" in payload:
        return True
    if "id" in payload and isinstance(transcript, list):
        return True
    return False


def request_from_export_text(text: str, summary: str = "", *, provider: ProviderName = "offline") -> EvaluateRequest:
    """Parse pipe text, an EvaluateRequest JSON object, or a Pocket-shaped export."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Transcript text is empty.")
    if stripped[0] not in "[{":
        request = parse_transcript(stripped, summary=summary)
        request.provider = provider
        return request
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Transcript JSON could not be parsed: {exc.msg}.") from exc
    if isinstance(payload, list):
        request = parse_transcript(stripped, summary=summary)
        request.provider = provider
        return request
    if not isinstance(payload, dict):
        raise ValueError("JSON transcript must be a list or an object.")
    if isinstance(payload.get("data"), dict) and _looks_like_recording(payload["data"]):
        payload = payload["data"]
    if payload.get("event") and "transcript" in payload:
        request = request_from_webhook(PocketWebhookPayload.model_validate(payload), provider=provider)
        if summary and not request.summary:
            request.summary = summary
        return request
    if _looks_like_recording(payload):
        request = request_from_recording(PocketRecording.model_validate(payload), provider=provider)
        if summary and not request.summary:
            request.summary = summary
        return request
    request = parse_transcript(stripped, summary=summary)
    request.provider = provider
    return request


def _first(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.is_file():
            return path
    return None


def load_pair(
    transcript_path: Path,
    summary_path: Path | None,
    audio_path: Path | None,
    *,
    provider: ProviderName = "offline",
) -> EvaluateRequest:
    summary = _read_summary(summary_path) if summary_path is not None else ""
    request = request_from_export_text(_read_text(transcript_path), summary=summary, provider=provider)
    if audio_path is not None:
        request.audio_path = str(audio_path)
    return request


def discover_recordings(
    folder: Path, *, provider: ProviderName = "offline"
) -> list[tuple[str, EvaluateRequest, str]]:
    """Return (id, request) for each recording in a local folder.

    A recording is either a subdirectory with `transcript.txt` (or `.md` / `.json`)
    and an optional `summary.txt`, or a top-level JSON export.
    """
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")
    found: list[tuple[str, EvaluateRequest, str]] = []
    for child in sorted(path for path in folder.iterdir() if path.is_dir() and not path.name.startswith(".")):
        transcript = _first(child, TRANSCRIPT_NAMES)
        if transcript is None:
            continue
        request = load_pair(
            transcript,
            _first(child, SUMMARY_NAMES),
            _first(child, AUDIO_NAMES),
            provider=provider,
        )
        recording_id = request.recording_id or child.name
        request.recording_id = recording_id
        if not request.title:
            request.title = child.name
        found.append((recording_id, request, str(child)))
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("."):
            continue
        request = request_from_export_text(_read_text(path), provider=provider)
        recording_id = request.recording_id or path.stem
        request.recording_id = recording_id
        found.append((recording_id, request, str(path)))
    if not found:
        raise ValueError(
            f"No recordings found in {folder}. Expected a subfolder with transcript.txt "
            "and summary.txt, or a JSON export. The layout is in the README section "
            "'Run on your own recordings'."
        )
    return found


def write_report(path: Path, report: dict, *, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"notice": LOCAL_NOTICE, "source": source, "report": report}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"notice": LOCAL_NOTICE, "recordings": rows}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
