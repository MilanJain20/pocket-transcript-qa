"""Pipeline, HTTP API, webhook, and Pocket client tests."""

import json

import httpx
from fastapi.testclient import TestClient

from pocket_qa.api import create_app
from pocket_qa.evaluate import evaluate
from pocket_qa.llm.anthropic import AnthropicClient
from pocket_qa.llm.azure import AzureOpenAIClient
from pocket_qa.llm.openai_compat import OpenAICompatibleClient
from pocket_qa.models import EvaluateRequest, Segment
from pocket_qa.pocket.http import DOCUMENTED_BASE_URL, HttpPocketClient
from pocket_qa.pocket.mock import MockPocketClient
from pocket_qa.pocket.normalize import request_from_recording, request_from_webhook
from pocket_qa.pocket.schemas import PocketWebhookPayload
from pocket_qa.pocket.webhook import clear_cache, sign_payload
from pocket_qa.summary_check import check_summary
from pocket_qa.transcript_format import parse_transcript


def _request() -> EvaluateRequest:
    return EvaluateRequest(
        transcript=[
            Segment(
                speaker="SPEAKER_00",
                speaker_name="Alice",
                text="We should ship the release on Friday.",
                start=0,
                end=3.4,
            ),
            Segment(
                speaker="SPEAKER_01",
                speaker_name="Bob",
                text="The budget for the launch is 12000 dollars.",
                start=3.8,
                end=7.6,
            ),
        ],
        summary="The team should ship the release on Friday. The budget for the launch is 12000 dollars.",
        recording_id="rec_test",
        title="Test",
    )


def test_pipeline_order_and_pass() -> None:
    report = evaluate(_request())
    assert [step.name for step in report.pipeline] == [
        "ingest",
        "transcript_checks",
        "summary_checks",
        "speaker_checks",
        "gate",
        "report",
    ]
    assert report.gate.decision == "PASS"
    assert report.provider == "offline"


def test_parse_pipe_format() -> None:
    request = parse_transcript(
        "Alice | 0 | 3.4 | We should ship the release on Friday.\n",
        summary="The team should ship the release on Friday.",
    )
    assert request.transcript[0].speaker_name == "Alice"
    assert request.summary.startswith("The team")


def test_api_evaluate_and_samples() -> None:
    client = TestClient(create_app(webhook_secret="test-secret"))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["pocket_client"] == "mock"
    assert health.json()["webhook_secret_configured"] is True
    page = client.get("/")
    assert page.status_code == 200
    assert "Not affiliated" in page.text
    samples = client.get("/v1/samples")
    assert samples.status_code == 200
    assert samples.json()["mock"] is True
    body = _request().model_dump()
    body.pop("audio_path")
    response = client.post("/v1/evaluate", json=body)
    assert response.status_code == 200
    assert response.json()["gate"]["decision"] == "PASS"
    empty = client.post("/v1/evaluate", json={"transcript": [], "summary": "Hello team today."})
    assert empty.status_code == 400


def test_webhook_signature_and_idempotency() -> None:
    clear_cache()
    app = create_app(webhook_secret="test-secret")
    client = TestClient(app)
    payload = {
        "event": "summary.completed",
        "timestamp": "2026-04-02T15:00:00.000Z",
        "user": {"id": "user_abc", "email": "alice@example.com"},
        "recording": {
            "id": "rec_webhook",
            "title": "Synthetic standup",
            "duration": 12,
            "recordingAt": "2026-04-02T14:55:00.000Z",
            "createdAt": "2026-04-02T15:00:00.000Z",
        },
        "summarizations": {
            "sum_1": {
                "summarizationId": "sum_1",
                "processingStatus": "completed",
                "updatedAt": "2026-04-02T15:05:00.000Z",
                "v2": {
                    "summary": {
                        "markdown": "The team should ship the release on Friday."
                    }
                },
            }
        },
        "transcript": [
            {
                "speaker": "SPEAKER_00",
                "speakerName": "Alice",
                "text": "We should ship the release on Friday.",
                "start": 0.0,
                "end": 3.4,
            }
        ],
        "speakers": {"SPEAKER_00": {"name": "Alice", "speakerId": "spk_alice"}},
    }
    raw = json.dumps(payload).encode()
    rejected = client.post("/v1/webhooks/pocket", content=raw, headers={"content-type": "application/json"})
    assert rejected.status_code == 401
    import time

    timestamp = str(int(time.time() * 1000))
    signature = sign_payload("test-secret", timestamp, raw)
    headers = {
        "content-type": "application/json",
        "X-HeyPocket-Timestamp": timestamp,
        "X-HeyPocket-Signature": signature,
        "User-Agent": "HeyPocket-Webhook/1.0",
    }
    first = client.post("/v1/webhooks/pocket", content=raw, headers=headers)
    assert first.status_code == 200
    assert first.json()["evaluated"] is True
    assert first.json()["report"]["gate"]["decision"] == "PASS"
    second = client.post("/v1/webhooks/pocket", content=raw, headers=headers)
    assert second.json()["duplicate"] is True
    recent = client.get("/v1/webhooks/pocket/recent")
    assert len(recent.json()["reports"]) == 1
    clear_cache()


def test_webhook_normalizer_matches_documented_shape() -> None:
    payload = PocketWebhookPayload.model_validate(
        {
            "event": "summary.completed",
            "recording": {"id": "rec_abc123", "title": "Team Standup"},
            "summarizations": {
                "sum_456": {
                    "v2": {"summary": {"markdown": "Ship the release on Friday soon."}}
                }
            },
            "transcript": [
                {"speaker": "SPEAKER_00", "speakerName": "Alice", "text": "Let's go.", "start": 0, "end": 1}
            ],
        }
    )
    request = request_from_webhook(payload)
    assert request.recording_id == "rec_abc123"
    assert request.transcript[0].speaker_name == "Alice"
    assert "Ship the release" in request.summary


def test_mock_client_and_http_client_parse_documented_recording() -> None:
    mock = MockPocketClient()
    assert mock.is_mock is True
    recording = mock.get_recording("rec_demo_standup")
    request = request_from_recording(recording)
    assert request.summary
    assert request.transcript[0].speaker_name == "Alice"
    listed = mock.list_recordings()
    assert listed.pagination["total"] == 2
    missing = mock.search("this phrase is not present")
    assert missing["data"] == []

    documented = {
        "success": True,
        "data": {
            "id": "rec_documented",
            "title": "Team Standup",
            "duration": 18,
            "created_at": "2026-04-02T15:00:00.000Z",
            "recording_at": "2026-04-02T14:55:00.000Z",
            "speakers": {"SPEAKER_00": {"name": "Alice", "speakerId": "spk_alice"}},
            "summarizations": None,
            "transcript": {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "speakerName": "Alice",
                        "text": "We should ship the release on Friday.",
                        "originalText": "We should ship the release on Friday.",
                        "start": 0,
                        "end": 3.4,
                    }
                ]
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer pk_test"
        assert request.url.path == "/api/v1/public/recordings/rec_documented"
        return httpx.Response(200, json=documented)

    client = HttpPocketClient(
        api_key="pk_test",
        base_url="https://public.heypocketai.com/api/v1",
        transport=httpx.MockTransport(handler),
    )
    assert client.is_mock is False
    assert client.base_url == DOCUMENTED_BASE_URL
    fetched = client.get_recording("rec_documented")
    assert fetched.transcript.segments[0].speaker_name == "Alice"


def test_http_client_requires_a_key() -> None:
    client = HttpPocketClient(api_key="", base_url=DOCUMENTED_BASE_URL)
    try:
        client.list_recordings()
    except Exception as exc:  # noqa: BLE001
        assert "POCKET_API_KEY" in str(exc)
    else:
        raise AssertionError("expected an auth error")


def test_openai_compatible_client_and_summary_provider(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_object"
        content = json.dumps(
            {"label": "contradicted", "evidence": "12000", "rationale": "amount differs", "score": 0.8}
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    model = OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr("pocket_qa.llm.factory.build_llm", lambda provider: model)
    segments = [
        Segment(
            speaker="Bob",
            speaker_name="Bob",
            text="The budget for the launch is 12000 dollars.",
            start=0,
            end=4,
        )
    ]
    report = check_summary(segments, "The budget for the launch is 40000 dollars.", provider="openai")
    assert report.provider == "openai"
    assert report.claims[0].label == "contradicted"


def test_azure_and_anthropic_request_shapes() -> None:
    def azure(request: httpx.Request) -> httpx.Response:
        assert "/openai/deployments/qa-deploy/chat/completions" in str(request.url)
        assert request.headers["api-key"] == "azure-key"
        content = json.dumps({"label": "entailed", "evidence": "q", "rationale": "ok", "score": 0.7})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    azure_client = AzureOpenAIClient(
        endpoint="https://example.openai.azure.com",
        api_key="azure-key",
        deployment="qa-deploy",
        transport=httpx.MockTransport(azure),
    )
    assert azure_client.complete_json("sys", "user")["label"] == "entailed"

    def anthropic(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "anthropic-key"
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"label":"unsupported","rationale":"missing","score":0.2}'}]},
        )

    anthropic_client = AnthropicClient(
        api_key="anthropic-key",
        base_url="https://api.anthropic.com",
        transport=httpx.MockTransport(anthropic),
    )
    assert anthropic_client.complete_json("sys", "user")["label"] == "unsupported"


def test_mock_pull_blocks_faulty_recording() -> None:
    client = TestClient(create_app())
    response = client.post("/v1/pocket/recordings/rec_demo_faulty/evaluate")
    assert response.status_code == 200
    assert response.json()["gate"]["decision"] == "BLOCK"
    missing = client.post("/v1/pocket/recordings/rec_missing/evaluate")
    assert missing.status_code == 404
