"""LangGraph pipeline: ingest, transcript, summary, speakers, gate, report.

Each node is a single-purpose stage. Offline runs do not call a model.
When a provider is selected, only the summary stage calls it.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from pocket_qa.gate import decide
from pocket_qa.models import (
    AgentStep,
    EvaluateRequest,
    EvaluationReport,
    GateThresholds,
)
from pocket_qa.normalize import prepare_request
from pocket_qa.speaker_check import check_speakers
from pocket_qa.summary_check import check_summary
from pocket_qa.transcript_check import check_transcript

AGENT_ROLES: tuple[tuple[str, str], ...] = (
    ("ingest", "Normalize segments, ids, and optional audio."),
    ("transcript_checks", "Flag fabricated or silent-audio transcript spans."),
    ("summary_checks", "Split the summary into claims and ground each one."),
    ("speaker_checks", "Check diarization, voice clusters, and attribution."),
    ("gate", "Apply release thresholds and decide PASS or BLOCK."),
    ("report", "Assemble the evaluation report."),
)


class GraphState(TypedDict, total=False):
    request: dict
    segments: list[dict]
    audio_path: str | None
    summary_text: str
    provider: str
    thresholds: dict | None
    recording_id: str | None
    title: str | None
    transcript_report: dict
    summary_report: dict
    speaker_report: dict
    gate: dict
    report: dict
    trace: list[str]


def _ingest(state: GraphState) -> dict:
    request = EvaluateRequest.model_validate(state["request"])
    segments, audio_path = prepare_request(request)
    return {
        "segments": [segment.model_dump() for segment in segments],
        "audio_path": audio_path,
        "summary_text": request.summary,
        "provider": request.provider,
        "thresholds": request.thresholds.model_dump() if request.thresholds else None,
        "recording_id": request.recording_id,
        "title": request.title,
        "trace": ["ingest"],
    }


def _transcript(state: GraphState) -> dict:
    from pocket_qa.models import Segment

    segments = [Segment.model_validate(item) for item in state["segments"]]
    report = check_transcript(segments, audio_path=state.get("audio_path"))
    return {
        "transcript_report": report.model_dump(),
        "trace": [*state.get("trace", []), "transcript_checks"],
    }


def _summary(state: GraphState) -> dict:
    from pocket_qa.models import Segment

    segments = [Segment.model_validate(item) for item in state["segments"]]
    report = check_summary(segments, state.get("summary_text", ""), provider=state.get("provider", "offline"))
    return {
        "summary_report": report.model_dump(),
        "trace": [*state.get("trace", []), "summary_checks"],
    }


def _speakers(state: GraphState) -> dict:
    from pocket_qa.models import Segment

    segments = [Segment.model_validate(item) for item in state["segments"]]
    report = check_speakers(segments, state.get("summary_text", ""))
    return {
        "speaker_report": report.model_dump(),
        "trace": [*state.get("trace", []), "speaker_checks"],
    }


def _gate(state: GraphState) -> dict:
    from pocket_qa.models import SpeakerReport, SummaryReport, TranscriptReport

    thresholds = GateThresholds.model_validate(state["thresholds"]) if state.get("thresholds") else None
    decision = decide(
        SummaryReport.model_validate(state["summary_report"]),
        TranscriptReport.model_validate(state["transcript_report"]),
        SpeakerReport.model_validate(state["speaker_report"]),
        thresholds,
    )
    return {
        "gate": decision.model_dump(),
        "trace": [*state.get("trace", []), "gate"],
    }


def _report(state: GraphState) -> dict:
    trace = [*state.get("trace", []), "report"]
    roles = dict(AGENT_ROLES)
    report = EvaluationReport(
        recording_id=state.get("recording_id"),
        title=state.get("title"),
        summary=state["summary_report"],  # type: ignore[arg-type]
        transcript=state["transcript_report"],  # type: ignore[arg-type]
        speakers=state["speaker_report"],  # type: ignore[arg-type]
        gate=state["gate"],  # type: ignore[arg-type]
        provider=state.get("provider", "offline"),
        pipeline=[AgentStep(name=name, role=roles.get(name, name)) for name in trace],
    )
    return {"report": report.model_dump(), "trace": trace}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("ingest", _ingest)
    graph.add_node("transcript_checks", _transcript)
    graph.add_node("summary_checks", _summary)
    graph.add_node("speaker_checks", _speakers)
    graph.add_node("gate", _gate)
    graph.add_node("report", _report)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "transcript_checks")
    graph.add_edge("transcript_checks", "summary_checks")
    graph.add_edge("summary_checks", "speaker_checks")
    graph.add_edge("speaker_checks", "gate")
    graph.add_edge("gate", "report")
    graph.add_edge("report", END)
    return graph.compile()


_COMPILED = None


def get_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED
