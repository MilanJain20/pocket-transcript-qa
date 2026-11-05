"""Turn checker scores into a PASS or BLOCK release decision."""

from __future__ import annotations

from pocket_qa.models import (
    GateDecision,
    GateThresholds,
    SpeakerReport,
    SummaryReport,
    TranscriptReport,
)


def decide(
    summary: SummaryReport,
    transcript: TranscriptReport,
    speakers: SpeakerReport,
    thresholds: GateThresholds | None = None,
) -> GateDecision:
    policy = thresholds or GateThresholds()
    reasons: list[str] = []
    claim_count = len(summary.claims)

    if claim_count == 0 and policy.block_on_empty_summary:
        reasons.append("The summary produced no atomic claims.")
    if summary.contradicted_count > policy.max_contradicted_claims:
        reasons.append(
            f"{summary.contradicted_count} contradicted claim(s), "
            f"above the limit of {policy.max_contradicted_claims}."
        )
    if claim_count and summary.unsupported_rate > policy.max_unsupported_rate + 1e-9:
        reasons.append(
            f"Unsupported claim rate {summary.unsupported_rate:.2f} "
            f"exceeds {policy.max_unsupported_rate:.2f}."
        )
    if claim_count and summary.grounding_score + 1e-9 < policy.min_grounding_score:
        reasons.append(
            f"Grounding score {summary.grounding_score:.2f} "
            f"is below {policy.min_grounding_score:.2f}."
        )
    fabricated = len(transcript.fabricated_segment_ids)
    if fabricated > policy.max_fabricated_segments:
        reasons.append(
            f"{fabricated} fabricated transcript segment(s), "
            f"above the limit of {policy.max_fabricated_segments}."
        )
    speaker_issues = len(speakers.flags)
    if speaker_issues > policy.max_speaker_issues:
        reasons.append(
            f"{speaker_issues} speaker-consistency issue(s), "
            f"above the limit of {policy.max_speaker_issues}."
        )

    decision = "BLOCK" if reasons else "PASS"
    if decision == "PASS":
        reasons.append("All configured release thresholds passed.")
    scores = {
        "grounding_score": summary.grounding_score,
        "unsupported_rate": summary.unsupported_rate,
        "contradicted_claims": float(summary.contradicted_count),
        "fabricated_segments": float(fabricated),
        "speaker_issues": float(speaker_issues),
    }
    return GateDecision(decision=decision, reasons=reasons, scores=scores)
