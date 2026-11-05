"""In-memory Pocket client.

MOCK. This is not the Pocket service, it does not send network requests, and
the recordings below are synthetic. It implements the same methods as
HttpPocketClient so a webhook or CLI demo can run with no API key.
"""

from __future__ import annotations

from typing import Any

from pocket_qa.pocket.errors import PocketNotFoundError
from pocket_qa.pocket.schemas import (
    PocketListResponse,
    PocketRecording,
    PocketRecordingResponse,
    PocketSpeaker,
    PocketSummarization,
    PocketSummarizationV2,
    PocketSummaryBody,
    PocketTranscriptObject,
    PocketTranscriptSegment,
)


def _demo_recording() -> PocketRecording:
    return PocketRecording(
        id="rec_demo_standup",
        title="Synthetic standup",
        duration=18.0,
        language="English",
        state="completed",
        created_at="2026-04-02T15:00:00.000Z",
        recording_at="2026-04-02T14:55:00.000Z",
        transcript=PocketTranscriptObject(
            segments=[
                PocketTranscriptSegment(
                    speaker="SPEAKER_00",
                    speakerName="Alice",
                    text="We should ship the release on Friday.",
                    start=0.0,
                    end=3.4,
                ),
                PocketTranscriptSegment(
                    speaker="SPEAKER_01",
                    speakerName="Bob",
                    text="The budget for the launch is 12000 dollars.",
                    start=3.8,
                    end=7.6,
                ),
                PocketTranscriptSegment(
                    speaker="SPEAKER_00",
                    speakerName="Alice",
                    text="I will send the test plan by Thursday.",
                    start=8.0,
                    end=11.4,
                ),
            ]
        ),
        summarizations={
            "sum_demo": PocketSummarization(
                id="row_demo",
                summarizationId="sum_demo",
                processingStatus="completed",
                updatedAt="2026-04-02T15:05:00.000Z",
                v2=PocketSummarizationV2(
                    summary=PocketSummaryBody(
                        title="Synthetic standup",
                        markdown=(
                            "The team should ship the release on Friday. "
                            "The budget for the launch is 12000 dollars."
                        ),
                        bulletPoints=[
                            "Ship the release on Friday",
                            "Launch budget is 12000 dollars",
                        ],
                    )
                ),
            )
        },
        speakers={
            "SPEAKER_00": PocketSpeaker(name="Alice", speakerId="spk_alice"),
            "SPEAKER_01": PocketSpeaker(name="Bob", speakerId="spk_bob"),
        },
    )


def _faulty_recording() -> PocketRecording:
    return PocketRecording(
        id="rec_demo_faulty",
        title="Synthetic standup with a bad summary",
        duration=12.0,
        language="English",
        state="completed",
        created_at="2026-04-03T15:00:00.000Z",
        recording_at="2026-04-03T14:55:00.000Z",
        transcript=PocketTranscriptObject(
            segments=[
                PocketTranscriptSegment(
                    speaker="SPEAKER_00",
                    speakerName="Alice",
                    text="We should ship the release on Friday.",
                    start=0.0,
                    end=3.4,
                ),
                PocketTranscriptSegment(
                    speaker="SPEAKER_01",
                    speakerName="Bob",
                    text="The budget for the launch is 12000 dollars.",
                    start=3.8,
                    end=7.6,
                ),
                PocketTranscriptSegment(
                    speaker="SPEAKER_01",
                    text="Thanks for watching.",
                    start=8.0,
                    end=9.4,
                ),
            ]
        ),
        summarizations={
            "sum_faulty": PocketSummarization(
                id="row_faulty",
                summarizationId="sum_faulty",
                processingStatus="completed",
                updatedAt="2026-04-03T15:05:00.000Z",
                v2=PocketSummarizationV2(
                    summary=PocketSummaryBody(
                        markdown=(
                            "The team should ship the release on Friday. "
                            "The board approved a 4000000 dollar acquisition of Northwind Robotics."
                        ),
                    )
                ),
            )
        },
        speakers={"SPEAKER_00": PocketSpeaker(name="Alice", speakerId="spk_alice")},
    )


class MockPocketClient:
    """Synthetic stand-in for the Pocket Public API. MOCK."""

    is_mock = True

    def __init__(self, recordings: list[PocketRecording] | None = None) -> None:
        source = recordings if recordings is not None else [_demo_recording(), _faulty_recording()]
        self._recordings = {recording.id: recording for recording in source}

    def list_recordings(
        self,
        *,
        page: int = 1,
        limit: int = 20,
        start_date: str | None = None,
        end_date: str | None = None,
        tag_ids: str | None = None,
    ) -> PocketListResponse:
        del start_date, end_date, tag_ids
        items = list(self._recordings.values())
        start = max(page - 1, 0) * limit
        window = items[start : start + limit]
        return PocketListResponse(
            success=True,
            data=window,
            pagination={
                "page": page,
                "limit": limit,
                "total": len(items),
                "total_pages": max(1, (len(items) + limit - 1) // limit),
                "has_more": start + limit < len(items),
            },
        )

    def get_recording(
        self,
        recording_id: str,
        *,
        include_transcript: bool = True,
        include_summarizations: bool = True,
        summarization_id: str | None = None,
    ) -> PocketRecording:
        del summarization_id
        recording = self._recordings.get(recording_id)
        if recording is None:
            raise PocketNotFoundError(f"Mock client has no recording {recording_id}")
        data = recording.model_copy(deep=True)
        if not include_transcript:
            data.transcript = None
        if not include_summarizations:
            data.summarizations = None
        return data

    def search(self, query: str, limit: int | None = None) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query is required")
        cap = limit or 8
        needle = query.lower()
        hits = []
        for recording in self._recordings.values():
            summary = ""
            if recording.summarizations:
                for item in recording.summarizations.values():
                    if item.v2 and item.v2.summary and item.v2.summary.markdown:
                        summary = item.v2.summary.markdown
            blob = f"{recording.title or ''}\n{summary}".lower()
            if needle in blob:
                hits.append(
                    {
                        "id": recording.id,
                        "title": recording.title,
                        "mock": True,
                    }
                )
            if len(hits) >= cap:
                break
        return {"success": True, "mock": True, "data": hits}

    def get_organization_recording(
        self,
        org_id: str,
        recording_id: str,
        *,
        include: str = "transcript,summary",
    ) -> PocketRecording:
        del org_id, include
        return self.get_recording(recording_id)

    def as_response(self, recording_id: str) -> PocketRecordingResponse:
        return PocketRecordingResponse(success=True, data=self.get_recording(recording_id))
