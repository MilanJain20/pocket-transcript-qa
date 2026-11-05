"""MCP server for the eval checks.

This process is the Pocket Transcript QA harness. It is not Pocket's official
MCP server at https://public.heypocketai.com/mcp, which requires a Pocket login
or API key and is not called from here.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from pocket_qa.evaluate import evaluate
from pocket_qa.gate import decide
from pocket_qa.models import GateThresholds
from pocket_qa.normalize import prepare_request
from pocket_qa.speaker_check import check_speakers
from pocket_qa.summary_check import check_summary
from pocket_qa.transcript_check import check_transcript
from pocket_qa.transcript_format import parse_transcript

mcp = MCPServer("pocket-transcript-qa")


def _request(transcript: str, summary: str, audio_id: str | None = None):
    request = parse_transcript(transcript, summary=summary)
    if audio_id:
        request.audio_id = audio_id  # type: ignore[assignment]
    return request


@mcp.tool()
def evaluate_transcript(transcript: str, summary: str, audio_id: str | None = None) -> str:
    """Run the full offline eval pipeline and return PASS or BLOCK with reasons.

    transcript uses lines of: Name | start_seconds | end_seconds | utterance
    audio_id may be 'silence', 'speech', or empty. No API key is used.
    """
    request = _request(transcript, summary, audio_id)
    report = evaluate(request)
    return json.dumps(report.model_dump(), indent=2)


@mcp.tool()
def check_summary_claims(transcript: str, summary: str) -> str:
    """Split a summary into claims and mark each entailed, unsupported, or contradicted."""
    request = _request(transcript, summary)
    segments, _ = prepare_request(request)
    report = check_summary(segments, summary, provider="offline")
    return json.dumps(report.model_dump(), indent=2)


@mcp.tool()
def check_transcript_fabrication(transcript: str, audio_id: str | None = None) -> str:
    """Flag filler hallucinations, repetition loops, word-rate anomalies, and silent audio."""
    request = _request(transcript, "", audio_id)
    segments, audio_path = prepare_request(request)
    report = check_transcript(segments, audio_path=audio_path)
    return json.dumps(report.model_dump(), indent=2)


@mcp.tool()
def check_speaker_consistency(transcript: str, summary: str = "") -> str:
    """Flag mid-sentence speaker flips, implausible turns, and wrong summary attribution."""
    request = _request(transcript, summary)
    segments, _ = prepare_request(request)
    report = check_speakers(segments, summary)
    return json.dumps(report.model_dump(), indent=2)


@mcp.tool()
def release_gate(transcript: str, summary: str, audio_id: str | None = None) -> str:
    """Apply the default release thresholds and return PASS or BLOCK."""
    request = _request(transcript, summary, audio_id)
    segments, audio_path = prepare_request(request)
    summary_report = check_summary(segments, summary, provider="offline")
    transcript_report = check_transcript(segments, audio_path=audio_path)
    speaker_report = check_speakers(segments, summary)
    decision = decide(summary_report, transcript_report, speaker_report, GateThresholds())
    return json.dumps(decision.model_dump(), indent=2)


def tool_names() -> list[str]:
    tools = mcp._tool_manager.list_tools()
    return sorted(tool.name for tool in tools)
