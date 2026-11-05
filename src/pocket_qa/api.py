"""Local FastAPI app: evaluation, release gate, and a Pocket webhook receiver.

The server has no authentication of its own. It is meant to run on localhost.
Webhook HMAC verification is separate and uses POCKET_WEBHOOK_SECRET.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, field_validator

from pocket_qa import __version__
from pocket_qa.demo_pages import readme_page, title_card
from pocket_qa.evaluate import evaluate
from pocket_qa.models import EvaluationReport, GateThresholds, ProviderName, Segment
from pocket_qa.pocket.errors import PocketNotFoundError
from pocket_qa.pocket.mock import MockPocketClient
from pocket_qa.pocket.normalize import request_from_recording, request_from_webhook
from pocket_qa.pocket.schemas import PocketWebhookPayload
from pocket_qa.pocket.webhook import (
    WebhookVerificationError,
    body_key,
    clear_cache,
    recall,
    recent,
    remember,
    verify_signature,
)
from pocket_qa.transcript_format import format_transcript

_WEB_ROOT = Path(__file__).resolve().parent / "web"


class ApiEvaluateRequest(BaseModel):
    transcript: list[Segment]
    summary: str = ""
    audio_id: str | None = None
    provider: ProviderName = "offline"
    thresholds: GateThresholds | None = None
    recording_id: str | None = None
    title: str | None = None

    @field_validator("audio_id")
    @classmethod
    def _audio(cls, value: str | None) -> str | None:
        if value in (None, "", "silence", "speech"):
            return value or None
        raise ValueError("audio_id must be 'silence', 'speech', or omitted")


class ApiError(BaseModel):
    error: str


def _client() -> MockPocketClient:
    return MockPocketClient()


def create_app(
    *,
    webhook_secret: str | None = None,
    allow_unsigned: bool | None = None,
) -> FastAPI:
    secret = webhook_secret if webhook_secret is not None else os.environ.get("POCKET_WEBHOOK_SECRET")
    if allow_unsigned is None:
        allow_unsigned = os.environ.get("POCKET_QA_ALLOW_UNSIGNED_WEBHOOKS") == "1"

    app = FastAPI(
        title="Pocket Transcript QA",
        version=__version__,
        description=(
            "Independent evaluation harness for voice transcripts and summaries. "
            "Not affiliated with or endorsed by Pocket. "
            "The bundled Pocket client used by /v1/pocket is a mock."
        ),
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = _WEB_ROOT / "index.html"
        return page.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "provider_default": "offline",
            "pocket_client": "mock",
            "webhook_secret_configured": bool(secret),
            "unsigned_webhooks_allowed": bool(allow_unsigned),
        }

    @app.post("/v1/evaluate", response_model=EvaluationReport)
    def evaluate_route(body: ApiEvaluateRequest) -> EvaluationReport | JSONResponse:
        from pocket_qa.models import EvaluateRequest

        try:
            report = evaluate(
                EvaluateRequest(
                    transcript=body.transcript,
                    summary=body.summary,
                    audio_id=body.audio_id,  # type: ignore[arg-type]
                    provider=body.provider,
                    thresholds=body.thresholds,
                    recording_id=body.recording_id,
                    title=body.title,
                )
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return report

    @app.post("/v1/gate", response_model=EvaluationReport)
    def gate_route(body: ApiEvaluateRequest) -> EvaluationReport | JSONResponse:
        return evaluate_route(body)

    @app.get("/v1/samples")
    def samples() -> dict[str, object]:
        client = _client()
        items = []
        for recording_id, blurb, audio_id in (
            ("rec_demo_standup", "Grounded summary of a synthetic standup. Expect PASS.", None),
            (
                "rec_demo_faulty",
                "Unsupported acquisition claim plus a 'Thanks for watching' segment. Expect BLOCK.",
                None,
            ),
        ):
            recording = client.get_recording(recording_id)
            request = request_from_recording(recording)
            items.append(
                {
                    "id": recording_id,
                    "title": recording.title,
                    "description": blurb,
                    "summary": request.summary,
                    "transcript": [segment.model_dump() for segment in request.transcript],
                    "transcript_text": format_transcript(request.transcript),
                    "audio_id": audio_id,
                    "mock": True,
                }
            )
        items.append(_silence_sample())
        items.append(_speaker_sample())
        items.append(_combined_fault_sample())
        return {"samples": items, "mock": True}

    @app.get("/readme", response_class=HTMLResponse)
    def readme(play: int = 0) -> str:
        return readme_page(play=bool(play))

    @app.get("/demo/card", response_class=HTMLResponse)
    def demo_card(step: str = "open", play: int = 0) -> str:
        return title_card(step, play=bool(play))

    @app.get("/v1/pocket/recordings")
    def list_mock_recordings() -> dict[str, object]:
        client = _client()
        payload = client.list_recordings()
        return {
            "mock": True,
            "note": "Served by MockPocketClient. No request was sent to Pocket.",
            "recordings": [item.model_dump() for item in payload.data],
            "pagination": payload.pagination,
        }

    @app.post("/v1/pocket/recordings/{recording_id}/evaluate", response_model=EvaluationReport)
    def evaluate_mock_recording(recording_id: str) -> EvaluationReport | JSONResponse:
        client = _client()
        try:
            recording = client.get_recording(recording_id)
        except PocketNotFoundError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc), "mock": True})
        request = request_from_recording(recording)
        report = evaluate(request)
        return report

    @app.post("/v1/webhooks/pocket")
    async def pocket_webhook(request: Request) -> JSONResponse:
        raw = await request.body()
        timestamp = request.headers.get("x-heypocket-timestamp")
        signature = request.headers.get("x-heypocket-signature")
        try:
            verify_signature(
                secret,
                timestamp,
                raw,
                signature,
                allow_unsigned=bool(allow_unsigned),
            )
        except WebhookVerificationError as exc:
            return JSONResponse(status_code=401, content={"error": str(exc)})
        key = body_key(raw)
        cached = recall(key)
        if cached is not None:
            return JSONResponse(
                {
                    "duplicate": True,
                    "mock_receiver": True,
                    "report": cached.model_dump(),
                }
            )
        try:
            payload = PocketWebhookPayload.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=400, content={"error": f"Invalid webhook payload: {exc}"})
        if not payload.transcript:
            return JSONResponse(
                {
                    "accepted": True,
                    "evaluated": False,
                    "event": payload.event,
                    "reason": "This delivery has no transcript segments to check.",
                    "mock_receiver": True,
                }
            )
        try:
            report = evaluate(request_from_webhook(payload))
        except (ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        remember(key, report)
        return JSONResponse(
            {
                "accepted": True,
                "evaluated": True,
                "duplicate": False,
                "event": payload.event,
                "mock_receiver": True,
                "report": report.model_dump(),
            }
        )

    @app.get("/v1/webhooks/pocket/recent")
    def recent_webhooks(limit: int = 20) -> dict[str, object]:
        reports = recent(limit=max(1, min(limit, 50)))
        return {
            "mock_receiver": True,
            "reports": [item.model_dump() for item in reports],
        }

    @app.delete("/v1/webhooks/pocket/recent")
    def reset_webhooks() -> dict[str, bool]:
        clear_cache()
        return {"cleared": True}

    return app


def _silence_sample() -> dict[str, object]:
    segments = [
        Segment(
            speaker="SPEAKER_00",
            speaker_name="Alice",
            text="The quarterly forecast looks strong overall today.",
            start=1.0,
            end=5.0,
        )
    ]
    summary = "The quarterly forecast looks strong overall today."
    return {
        "id": "sample_silence",
        "title": "Text over near-silent audio",
        "description": "A normal-looking sentence on the synthetic silence fixture. Expect BLOCK from audio energy.",
        "summary": summary,
        "transcript": [segment.model_dump() for segment in segments],
        "transcript_text": format_transcript(segments),
        "audio_id": "silence",
        "mock": True,
    }


def _combined_fault_sample() -> dict[str, object]:
    """One synthetic note that trips claim, fabrication, and speaker checks together."""
    segments = [
        Segment(
            speaker="SPEAKER_00",
            speaker_name="Alice",
            text="We should ship the release on Friday.",
            start=0.0,
            end=3.4,
        ),
        Segment(
            speaker="SPEAKER_01",
            speaker_name="Bob",
            text="The budget for the launch is 12000 dollars.",
            start=3.8,
            end=7.6,
        ),
        Segment(
            speaker="SPEAKER_01",
            speaker_name="Bob",
            text="Thanks for watching.",
            start=8.0,
            end=9.4,
        ),
    ]
    summary = (
        "Alice said the budget for the launch is 12000 dollars. "
        "The board approved a 4000000 dollar acquisition of Northwind Robotics."
    )
    return {
        "id": "sample_faulty_all",
        "title": "Faulty note: claim, fabrication, speaker",
        "description": (
            "Mock recording. The summary invents an acquisition, quotes Bob's budget as Alice's, "
            "and the transcript contains 'Thanks for watching'. Expect BLOCK."
        ),
        "summary": summary,
        "transcript": [segment.model_dump() for segment in segments],
        "transcript_text": format_transcript(segments),
        "audio_id": None,
        "mock": True,
    }


def _speaker_sample() -> dict[str, object]:
    segments = [
        Segment(
            speaker="SPEAKER_00",
            speaker_name="Alice",
            text="I will present the forecast on Monday.",
            start=0.0,
            end=3.2,
        ),
        Segment(
            speaker="SPEAKER_01",
            speaker_name="Bob",
            text="The budget for the launch is 12000 dollars.",
            start=3.6,
            end=7.4,
        ),
    ]
    summary = "Bob said that he will present the forecast on Monday."
    return {
        "id": "sample_speaker",
        "title": "Summary credits the wrong speaker",
        "description": "Alice is the one presenting the forecast. The summary gives the line to Bob. Expect BLOCK.",
        "summary": summary,
        "transcript": [segment.model_dump() for segment in segments],
        "transcript_text": format_transcript(segments),
        "audio_id": None,
        "mock": True,
    }


app = create_app()
