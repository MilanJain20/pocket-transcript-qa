"""Ground each summary claim in the transcript.

Offline mode is a lexical entailment heuristic: content-word overlap, number
mismatch, a closed antonym list, and negation polarity. It is the default.
It does not download a model and it does not call an API.

`local-nli` is an opt-in semantic checker. It scores the claim against
transcript spans with a local cross-encoder. `hybrid` keeps a decisive
lexical judgment and calls that model only for the remaining claims.
OpenAI-compatible, Azure, and Anthropic models remain separate providers.
"""

from __future__ import annotations

from pocket_qa.models import ClaimJudgment, Segment, SummaryReport
from pocket_qa.textutil import (
    antonym_conflict,
    content_tokens,
    has_negation,
    numbers,
    raw_words,
    token_recall,
)

# Hybrid keeps a near-copy lexical entailment. The bound sits above the
# lexical entailment cutoff and was not grid-searched on the benchmark.
HYBRID_EXTRACTIVE_RECALL = 0.90
# A three-way probability is a majority only at one half. Below that, no
# class holds half the mass, so the claim stays unsupported. Not tuned.
NLI_MAJORITY = 0.5
# Past this length the candidate set is pruned before NLI. The synthetic
# benchmark max is 7 segments, so the prune does not touch those scores.
LONG_TRANSCRIPT_SEGMENTS = 24
LONG_TRANSCRIPT_KEEP = 12
_MAX_FULL_CHARS = 2000
_CONDITIONAL_WORDS = frozenset({"if", "unless", "until", "provided", "assuming", "except"})

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


def _shortest_span_with_number(segments: list[Segment]) -> str:
    best = ""
    best_len = 10**9
    for text, _tokens in _windows(segments):
        if numbers(text) and len(text) < best_len:
            best = text
            best_len = len(text)
    return best


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
        # Label stays unsupported. Quote a span that actually states a number
        # so the reader can see the transcript's figure beside the claim.
        number_span = _shortest_span_with_number(segments) or evidence
        return ClaimJudgment(
            claim=claim,
            label="unsupported",
            score=score,
            evidence=number_span,
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


def _utterance(segment: Segment) -> str:
    text = segment.text.strip()
    name = (segment.speaker_name or "").strip()
    if name and name.lower() not in text.lower():
        return f"{name}: {text}"
    return text


def _dedupe(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def candidate_premises(claim: str, segments: list[Segment]) -> list[str]:
    """Spans the local NLI model may use as the premise for one claim.

    Short transcripts contribute each utterance, each adjacent pair, each
    adjacent triple, and the full transcript when it is still short. Speaker
    names are prefixed so "Bob will …" can be checked against "I will …".
    Longer transcripts keep the utterances with the most lexical overlap,
    plus their neighbors, and only those singles and adjacent pairs.
    """
    utterances = [_utterance(segment) for segment in segments if segment.text and segment.text.strip()]
    if not utterances:
        return []
    if len(utterances) > LONG_TRANSCRIPT_SEGMENTS:
        claim_tokens = set(content_tokens(claim))
        ranked = sorted(
            range(len(utterances)),
            key=lambda index: token_recall(claim_tokens, set(content_tokens(utterances[index]))),
            reverse=True,
        )
        chosen = set(ranked[:LONG_TRANSCRIPT_KEEP])
        expanded: set[int] = set()
        for index in chosen:
            expanded.add(index)
            if index - 1 >= 0:
                expanded.add(index - 1)
            if index + 1 < len(utterances):
                expanded.add(index + 1)
        selected = [index for index in range(len(utterances)) if index in expanded]
        texts = [utterances[index] for index in selected]
        selected_set = set(selected)
        for index in selected:
            if index + 1 in selected_set:
                texts.append(f"{utterances[index]} {utterances[index + 1]}")
        return _dedupe(texts)

    texts = list(utterances)
    for index in range(len(utterances) - 1):
        texts.append(f"{utterances[index]} {utterances[index + 1]}")
    if len(utterances) <= 12:
        for index in range(len(utterances) - 2):
            texts.append(f"{utterances[index]} {utterances[index + 1]} {utterances[index + 2]}")
    full = " ".join(utterances)
    if len(utterances) > 1 and len(full) <= _MAX_FULL_CHARS:
        texts.append(full)
    return _dedupe(texts)


def _has_conditional(text: str) -> bool:
    if raw_words(text) & _CONDITIONAL_WORDS:
        return True
    lowered = text.lower()
    return "as long as" in lowered or "only if" in lowered or "only when" in lowered


def _dropped_condition(claim: str, evidence: str) -> bool:
    return bool(evidence) and _has_conditional(evidence) and not _has_conditional(claim)


def _nli_from_distributions(claim: str, premises: list[str], distributions: list[object]) -> ClaimJudgment:
    from pocket_qa.llm.local_nli import model_id

    entail_index = max(range(len(distributions)), key=lambda index: distributions[index].prob("entailed"))  # type: ignore[attr-defined]
    contradict_index = max(
        range(len(distributions)),
        key=lambda index: distributions[index].prob("contradicted"),  # type: ignore[attr-defined]
    )
    entail_p = float(distributions[entail_index].prob("entailed"))  # type: ignore[attr-defined]
    contradict_p = float(distributions[contradict_index].prob("contradicted"))  # type: ignore[attr-defined]
    if entail_p >= NLI_MAJORITY:
        label = "entailed"
        score = entail_p
        evidence = premises[entail_index]
    elif contradict_p >= NLI_MAJORITY:
        label = "contradicted"
        score = contradict_p
        evidence = premises[contradict_index]
    else:
        label = "unsupported"
        score = entail_p
        evidence = premises[entail_index]
    rationale = (
        f"Local NLI ({model_id()}) on {len(premises)} transcript spans. "
        f"Best entailment probability {entail_p:.3f}; "
        f"best contradiction probability {contradict_p:.3f}. "
        f"A label is kept only when its probability is at least {NLI_MAJORITY:.1f}."
    )
    return ClaimJudgment(
        claim=claim,
        label=label,  # type: ignore[arg-type]
        score=max(0.0, min(1.0, score)),
        evidence=evidence,
        rationale=rationale,
    )


def judge_claim_with_local_nli(claim: str, segments: list[Segment]) -> ClaimJudgment:
    """Semantic support: NLI over transcript spans, not the lexical nearest span alone."""
    from pocket_qa.llm.local_nli import score_pairs

    premises = candidate_premises(claim, segments)
    if not premises:
        return ClaimJudgment(
            claim=claim,
            label="unsupported",
            score=0.0,
            evidence="",
            rationale="No transcript text was available for the local NLI model.",
        )
    distributions = score_pairs([(premise, claim) for premise in premises])
    if len(distributions) != len(premises):
        raise RuntimeError("The local NLI model returned a different number of scores than premises.")
    return _nli_from_distributions(claim, premises, distributions)


def _number_outside_transcript(claim: str, segments: list[Segment]) -> bool:
    claim_numbers = numbers(claim)
    if not claim_numbers:
        return False
    transcript_numbers = numbers(" ".join(segment.text for segment in segments))
    return not claim_numbers <= transcript_numbers


def hybrid_keeps_lexical(claim: str, segments: list[Segment]) -> bool:
    """True when the lexical judgment is decisive and NLI is not called.

    Kept: a number that is not in the transcript, any lexical contradiction,
    or a near-copy entailment whose best span does not add a conditional the
    claim dropped. Everything else is borderline and goes to local NLI.
    """
    judgment = heuristic_judge(claim, segments)
    if _number_outside_transcript(claim, segments):
        return True
    if judgment.label == "contradicted":
        return True
    evidence, recall = best_evidence(claim, segments)
    if (
        judgment.label == "entailed"
        and recall >= HYBRID_EXTRACTIVE_RECALL
        and not _dropped_condition(claim, evidence)
    ):
        return True
    return False


def judge_claim_hybrid(claim: str, segments: list[Segment]) -> ClaimJudgment:
    """Lexical first. Local NLI only when the lexical judgment is not decisive."""
    judgment = heuristic_judge(claim, segments)
    if hybrid_keeps_lexical(claim, segments):
        return judgment.model_copy(
            update={"rationale": f"Lexical decision kept. {judgment.rationale}"}
        )
    revised = judge_claim_with_local_nli(claim, segments)
    return revised.model_copy(update={"rationale": f"Borderline lexical score sent to NLI. {revised.rationale}"})


def check_summary(segments: list[Segment], summary: str, provider: str = "offline") -> SummaryReport:
    from pocket_qa.claims import split_claims

    claims = split_claims(summary)
    judgments: list[ClaimJudgment] = []
    client = None
    if provider not in {"offline", "local-nli", "hybrid"}:
        from pocket_qa.llm.factory import build_llm

        client = build_llm(provider)
    for claim in claims:
        if provider == "offline":
            judgments.append(heuristic_judge(claim, segments))
        elif provider == "local-nli":
            judgments.append(judge_claim_with_local_nli(claim, segments))
        elif provider == "hybrid":
            judgments.append(judge_claim_hybrid(claim, segments))
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
