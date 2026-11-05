"""Shared request and report models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["offline", "openai", "azure", "anthropic", "local-nli"]
ClaimLabel = Literal["entailed", "unsupported", "contradicted"]
FabricationKind = Literal[
    "filler_hallucination",
    "repetition_loop",
    "word_rate",
    "invalid_timestamp",
    "silent_audio",
]
SpeakerKind = Literal[
    "mid_sentence_flip",
    "implausible_turns",
    "voice_profile_mismatch",
    "wrong_speaker_attribution",
]


class Segment(BaseModel):
    """One diarized transcript span. Times are seconds from the start of the recording."""

    id: str | None = None
    speaker: str
    speaker_name: str | None = None
    text: str
    start: float
    end: float
    embedding: list[float] | None = None


class GateThresholds(BaseModel):
    """Release policy. A recording is BLOCKED when any limit is missed."""

    min_grounding_score: float = 0.80
    max_unsupported_rate: float = 0.15
    max_contradicted_claims: int = 0
    max_fabricated_segments: int = 0
    max_speaker_issues: int = 0
    block_on_empty_summary: bool = True


class EvaluateRequest(BaseModel):
    transcript: list[Segment]
    summary: str = ""
    audio_path: str | None = None
    audio_id: Literal["silence", "speech"] | None = None
    provider: ProviderName = "offline"
    thresholds: GateThresholds | None = None
    recording_id: str | None = None
    title: str | None = None


class ClaimJudgment(BaseModel):
    claim: str
    label: ClaimLabel
    score: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    rationale: str


class SummaryReport(BaseModel):
    claims: list[ClaimJudgment]
    grounding_score: float
    unsupported_count: int
    contradicted_count: int
    unsupported_rate: float
    has_issue: bool
    provider: str


class FabricationFlag(BaseModel):
    segment_id: str
    kind: FabricationKind
    message: str
    evidence: str


class TranscriptReport(BaseModel):
    flags: list[FabricationFlag]
    segment_count: int
    fabricated_segment_ids: list[str]
    has_issue: bool


class SpeakerFlag(BaseModel):
    kind: SpeakerKind
    message: str
    segment_id: str | None = None
    evidence: str


class SpeakerReport(BaseModel):
    flags: list[SpeakerFlag]
    has_issue: bool


class GateDecision(BaseModel):
    decision: Literal["PASS", "BLOCK"]
    reasons: list[str]
    scores: dict[str, float]


class AgentStep(BaseModel):
    name: str
    role: str
    status: Literal["ok", "error"] = "ok"


class EvaluationReport(BaseModel):
    recording_id: str | None
    title: str | None
    summary: SummaryReport
    transcript: TranscriptReport
    speakers: SpeakerReport
    gate: GateDecision
    provider: str
    pipeline: list[AgentStep]
