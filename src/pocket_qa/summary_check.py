"""Ground each summary claim in the transcript.

Offline mode is a lexical entailment heuristic: content-word overlap, number
mismatch, a closed antonym list, and negation polarity. It is not a neural
NLI model. The same judgments can be delegated to an OpenAI-compatible,
Azure, or Anthropic model, or to an optional local cross-encoder.
"""

from __future__ import annotations

from pocket_qa.models import ClaimJudgment, Segment, SummaryReport
from pocket_qa.textutil import (
    antonym_conflict,
    content_tokens,
    has_negation,
    numbers,
)

ENTAIL_RECALL = 0.60
CONTRADICTION_RECALL = 0.45


def _speaker_name_tokens(segments: list[Segment]) -> set[str]:
    tokens: set[str] = set()
    for segment in segments:
        if segment.speaker_name:
            tokens.update(content_tokens(segment.speaker_name))
    return tokens


def _windows(segments: list[Segment]) -> list[tuple[str, set[str]]]:
    """Every contiguous span. The shortest high-recall span wins later."""
    texts = [segment.text.strip() for segment in segments if segment.text.strip()]
    windows: list[tuple[str, set[str]]] = []
    for start in range(len(texts)):
        accumulated = ""
        for end in range(start, len(texts)):
            accumulated = texts[end] if start == end else f"{accumulated} {texts[end]}"
            windows.append((accumulated, set(content_tokens(accumulated))))
    return windows


def best_evidence(claim: str, segments: list[Segment]) -> tuple[str, float]:
    """Return the tightest transcript span with the highest content-token recall."""
    claim_tokens = set(content_tokens(claim))
    names = _speaker_name_tokens(segments)
    if not claim_tokens:
        return "", 0.0
    best_text = ""
    best_key = (-1.0, -1.0, -1)
    for text, tokens in _windows(segments):
        matched = (claim_tokens & tokens) | (claim_tokens & names)
        recall = len(matched) / len(claim_tokens)
        precision = len(claim_tokens & tokens) / len(tokens) if tokens else 0.0
        key = (recall, precision, -len(text))
        if key > best_key:
            best_text = text
            best_key = key
    if best_key[0] < 0:
        return "", 0.0
    return best_text, best_key[0]


def heuristic_judge(claim: str, segments: list[Segment]) -> ClaimJudgment:
    evidence, recall = best_evidence(claim, segments)
    transcript_text = " ".join(segment.text for segment in segments)
    claim_numbers = numbers(claim)
    transcript_numbers = numbers(transcript_text)
    evidence_numbers = numbers(evidence)
    score = max(0.0, min(1.0, recall))

    if claim_numbers and not claim_numbers <= transcript_numbers:
        if evidence_numbers and recall >= CONTRADICTION_RECALL:
            return ClaimJudgment(
                claim=claim,
                label="contradicted",
                score=max(score, 0.7),
                evidence=evidence,
                rationale="The claim uses a number that is not in the transcript, "
                "while the closest span uses a different number.",
            )
        return ClaimJudgment(
            claim=claim,
            label="unsupported",
            score=score,
            evidence=evidence,
            rationale="The claim contains a number that does not appear in the transcript.",
        )

    conflict = antonym_conflict(claim, evidence) or antonym_conflict(claim, transcript_text)
    if conflict and recall >= CONTRADICTION_RECALL:
        return ClaimJudgment(
            claim=claim,
            label="contradicted",
            score=max(score, 0.7),
            evidence=evidence,
            rationale=f"A closed-list antonym pair ({conflict}) is split between the claim and the transcript.",
        )

    if evidence and has_negation(claim) != has_negation(evidence) and recall >= CONTRADICTION_RECALL:
        return ClaimJudgment(
            claim=claim,
            label="contradicted",
            score=max(score, 0.7),
            evidence=evidence,
            rationale="The claim and the closest transcript span disagree on negation.",
        )

    if recall >= ENTAIL_RECALL:
        return ClaimJudgment(
            claim=claim,
            label="entailed",
            score=score,
            evidence=evidence,
            rationale="Enough content words in the claim appear in a transcript span.",
        )

    return ClaimJudgment(
        claim=claim,
        label="unsupported",
        score=score,
        evidence=evidence,
        rationale="The claim's content words are not covered by the transcript.",
    )


def _judgment_from_payload(claim: str, payload: dict[str, object], evidence_fallback: str) -> ClaimJudgment:
    label = str(payload.get("label", "")).strip().lower()
    if label not in {"entailed", "unsupported", "contradicted"}:
        label = "unsupported"
    rationale = str(payload.get("rationale") or "The model did not return a rationale.")
    evidence = str(payload.get("evidence") or evidence_fallback)
    try:
        score = float(payload.get("score", 0.5))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))
    return ClaimJudgment(
        claim=claim,
        label=label,  # type: ignore[arg-type]
        score=score,
        evidence=evidence,
        rationale=rationale,
    )


def judge_claim_with_llm(claim: str, segments: list[Segment], client: object) -> ClaimJudgment:
    """Ask a pluggable chat model. Fail closed if the payload is not usable."""
    from pocket_qa.llm.base import LLMClient

    if not isinstance(client, LLMClient):
        raise TypeError("client must implement LLMClient")
    transcript = "\n".join(
        f"{segment.speaker_name or segment.speaker}: {segment.text}" for segment in segments
    )
    system = (
        "You check whether one summary claim is grounded in a transcript. "
        "Reply with JSON only, using this shape: "
        '{"label": "entailed"|"unsupported"|"contradicted", "evidence": "short quote or empty", '
        '"rationale": "one sentence", "score": 0.0}. '
        "entailed means the transcript supports the claim. "
        "contradicted means the transcript says the opposite. "
        "unsupported means the transcript does not give enough evidence. "
        "Do not use outside knowledge."
    )
    user = f"TRANSCRIPT:\n{transcript}\n\nCLAIM:\n{claim}"
    try:
        payload = client.complete_json(system, user)
    except Exception as exc:  # noqa: BLE001 - fail closed for the release gate
        return ClaimJudgment(
            claim=claim,
            label="unsupported",
            score=0.0,
            evidence="",
            rationale=f"The model call failed closed: {exc.__class__.__name__}.",
        )
    evidence, _ = best_evidence(claim, segments)
    return _judgment_from_payload(claim, payload, evidence)


def judge_claim_with_local_nli(claim: str, segments: list[Segment]) -> ClaimJudgment:
    from pocket_qa.llm.local_nli import score_nli

    evidence, recall = best_evidence(claim, segments)
    if not evidence:
        return ClaimJudgment(
            claim=claim,
            label="unsupported",
            score=0.0,
            evidence="",
            rationale="No transcript text was available for the local NLI model.",
        )
    label, score = score_nli(premise=evidence, hypothesis=claim)
    rationale = f"Local cross-encoder label on the nearest transcript span (lexical recall {recall:.2f})."
    return ClaimJudgment(
        claim=claim,
        label=label,
        score=score,
        evidence=evidence,
        rationale=rationale,
    )


def check_summary(segments: list[Segment], summary: str, provider: str = "offline") -> SummaryReport:
    from pocket_qa.claims import split_claims

    claims = split_claims(summary)
    judgments: list[ClaimJudgment] = []
    client = None
    if provider not in {"offline", "local-nli"}:
        from pocket_qa.llm.factory import build_llm

        client = build_llm(provider)
    for claim in claims:
        if provider == "offline":
            judgments.append(heuristic_judge(claim, segments))
        elif provider == "local-nli":
            judgments.append(judge_claim_with_local_nli(claim, segments))
        else:
            judgments.append(judge_claim_with_llm(claim, segments, client))

    unsupported = sum(1 for item in judgments if item.label == "unsupported")
    contradicted = sum(1 for item in judgments if item.label == "contradicted")
    entailed = sum(1 for item in judgments if item.label == "entailed")
    total = len(judgments)
    grounding = entailed / total if total else 0.0
    unsupported_rate = unsupported / total if total else 0.0
    return SummaryReport(
        claims=judgments,
        grounding_score=round(grounding, 6),
        unsupported_count=unsupported,
        contradicted_count=contradicted,
        unsupported_rate=round(unsupported_rate, 6),
        has_issue=unsupported + contradicted > 0,
        provider=provider,
    )
