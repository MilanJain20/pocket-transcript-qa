"""Pydantic models for the public Pocket API shapes this client uses.

Fields follow the public docs fetched from docs.heypocketai.com on 2026-11-05.
Unknown fields are kept so a doc update does not crash the client.
The search response schema was only partially expanded on that docs page, so
search hits are parsed defensively.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PocketModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class PocketSpeaker(PocketModel):
    name: str | None = None
    speaker_id: str | None = Field(default=None, alias="speakerId")


class PocketTranscriptSegment(PocketModel):
    speaker: str | None = None
    speaker_name: str | None = Field(default=None, alias="speakerName")
    text: str | None = None
    original_text: str | None = Field(default=None, alias="originalText")
    start: float = 0.0
    end: float = 0.0


class PocketTranscriptObject(PocketModel):
    segments: list[PocketTranscriptSegment] = Field(default_factory=list)


class PocketSummaryBody(PocketModel):
    title: str | None = None
    markdown: str | None = None
    bullet_points: list[str] | None = Field(default=None, alias="bulletPoints")


class PocketSummarizationV2(PocketModel):
    summary: PocketSummaryBody | None = None


class PocketSummarization(PocketModel):
    id: str | None = None
    summarization_id: str | None = Field(default=None, alias="summarizationId")
    processing_status: str | None = Field(default=None, alias="processingStatus")
    v2: PocketSummarizationV2 | None = None
    updated_at: str | None = Field(default=None, alias="updatedAt")


class PocketRecording(PocketModel):
    """Recording resource from GET /public/recordings/{id} `data`."""

    id: str
    title: str | None = None
    duration: float | None = None
    language: str | None = None
    state: str | None = None
    created_at: str | None = None
    recording_at: str | None = None
    transcript: PocketTranscriptObject | list[PocketTranscriptSegment] | None = None
    summarizations: dict[str, PocketSummarization] | None = None
    summary: PocketSummaryBody | None = None
    speakers: dict[str, PocketSpeaker] | None = None


class PocketListResponse(PocketModel):
    success: bool | None = None
    data: list[PocketRecording] = Field(default_factory=list)
    pagination: dict[str, Any] | None = None


class PocketRecordingResponse(PocketModel):
    success: bool | None = None
    data: PocketRecording | None = None
    error: str | None = None
    error_code: str | None = Field(default=None, alias="error_code")


class PocketWebhookUser(PocketModel):
    id: str | None = None
    email: str | None = None


class PocketWebhookOrganization(PocketModel):
    id: str | None = None


class PocketWebhookRecording(PocketModel):
    id: str | None = None
    title: str | None = None
    description: str | None = None
    duration: float | None = None
    language: str | None = None
    recording_at: str | None = Field(default=None, alias="recordingAt")
    created_at: str | None = Field(default=None, alias="createdAt")


class PocketWebhookPayload(PocketModel):
    """Webhook body documented at docs.heypocketai.com/docs/api/webhooks."""

    event: str
    timestamp: str | None = None
    user: PocketWebhookUser | None = None
    organization: PocketWebhookOrganization | None = None
    recording: PocketWebhookRecording | None = None
    summarizations: dict[str, PocketSummarization] | None = None
    transcript: list[PocketTranscriptSegment] | None = None
    speakers: dict[str, PocketSpeaker] | None = None
