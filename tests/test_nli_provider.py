"""Local NLI and hybrid claim support. The model is mocked unless weights are cached."""

from __future__ import annotations

import pytest

from pocket_qa.llm.local_nli import NliDistribution, available, score_nli, score_pairs, weights_cached
from pocket_qa.models import Segment
from pocket_qa.summary_check import candidate_premises, check_summary, hybrid_keeps_lexical


def _seg(text: str, name: str = "Alice", start: float = 0.0, end: float = 2.0) -> Segment:
    return Segment(speaker=name, speaker_name=name, text=text, start=start, end=end)


def _dist(entailed: float, contradicted: float, unsupported: float) -> NliDistribution:
    return NliDistribution(
        {"entailed": entailed, "contradicted": contradicted, "unsupported": unsupported}
    )


def test_candidate_premises_include_speaker_and_neighbors() -> None:
    segments = [
        _seg("We should ship the release on Friday.", "Alice", 0, 2),
        _seg("The budget for the launch is 12000 dollars.", "Bob", 2, 4),
    ]
    premises = candidate_premises("The budget for the launch is 12000 dollars.", segments)
    assert "Alice: We should ship the release on Friday." in premises
    assert "Bob: The budget for the launch is 12000 dollars." in premises
    assert any("Alice:" in premise and "Bob:" in premise for premise in premises)


def test_long_transcript_is_pruned_before_nli() -> None:
    segments = [
        _seg(f"Unrelated sentence number {index} about nothing.", "Sam", index, index + 0.5)
        for index in range(30)
    ]
    segments[29] = _seg("The budget for the launch is 12000 dollars.", "Bob", 29, 30)
    premises = candidate_premises("The budget for the launch is 12000 dollars.", segments)
    assert len(premises) < 30
    assert any("12000" in premise for premise in premises)


def test_local_nli_uses_the_span_with_majority_entailment(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(pairs: list[tuple[str, str]]) -> list[NliDistribution]:
        scored = []
        for premise, _claim in pairs:
            if "12000" in premise and "Friday" not in premise:
                scored.append(_dist(0.81, 0.05, 0.14))
            else:
                scored.append(_dist(0.1, 0.1, 0.8))
        return scored

    monkeypatch.setattr("pocket_qa.llm.local_nli.score_pairs", fake)
    segments = [
        _seg("We should ship the release on Friday.", "Alice", 0, 2),
        _seg("The budget for the launch is 12000 dollars.", "Bob", 2, 4),
    ]
    report = check_summary(segments, "The budget for the launch is 12000 dollars.", provider="local-nli")
    assert report.provider == "local-nli"
    assert report.claims[0].label == "entailed"
    assert "12000" in report.claims[0].evidence
    assert report.claims[0].score == pytest.approx(0.81)


def test_local_nli_contradiction_and_no_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pocket_qa.llm.local_nli.score_pairs",
        lambda pairs: [_dist(0.1, 0.7, 0.2) for _pair in pairs],
    )
    contradicted = check_summary(
        [_seg("We will launch in March.")],
        "The team will not launch in March.",
        provider="local-nli",
    )
    assert contradicted.claims[0].label == "contradicted"

    monkeypatch.setattr(
        "pocket_qa.llm.local_nli.score_pairs",
        lambda pairs: [_dist(0.4, 0.3, 0.3) for _pair in pairs],
    )
    unsure = check_summary(
        [_seg("We will launch in March.")],
        "The team will launch in March.",
        provider="local-nli",
    )
    assert unsure.claims[0].label == "unsupported"
    assert unsure.has_issue is True


def test_local_nli_empty_transcript_does_not_call_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(pairs: list[tuple[str, str]]) -> list[NliDistribution]:
        raise AssertionError(pairs)

    monkeypatch.setattr("pocket_qa.llm.local_nli.score_pairs", fail)
    report = check_summary([], "The team should ship the release on Friday.", provider="local-nli")
    assert report.claims[0].label == "unsupported"
    assert report.claims[0].evidence == ""


def test_missing_extra_fails_closed_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pocket_qa.llm.local_nli._MODEL", None)
    monkeypatch.setattr("pocket_qa.llm.local_nli.available", lambda: False)
    with pytest.raises(RuntimeError, match="local-nli"):
        check_summary(
            [_seg("We should ship the release on Friday.")],
            "The team should ship the release on Friday.",
            provider="local-nli",
        )


def test_hybrid_keeps_a_number_mismatch_and_a_near_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(pairs: list[tuple[str, str]]) -> list[NliDistribution]:
        raise AssertionError(pairs)

    monkeypatch.setattr("pocket_qa.llm.local_nli.score_pairs", fail)
    number = check_summary(
        [_seg("The budget for the launch is 12000 dollars.", "Bob")],
        "The budget for the launch is 40000 dollars.",
        provider="hybrid",
    )
    assert number.provider == "hybrid"
    assert number.claims[0].label == "contradicted"
    assert number.claims[0].rationale.startswith("Lexical decision kept.")

    copied = check_summary(
        [_seg("We should ship the release on Friday.")],
        "We should ship the release on Friday.",
        provider="hybrid",
    )
    assert copied.claims[0].label == "entailed"
    kept = hybrid_keeps_lexical(
        "We should ship the release on Friday.",
        [_seg("We should ship the release on Friday.")],
    )
    assert kept is True


def test_hybrid_sends_a_paraphrase_and_a_dropped_condition_to_nli(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake(pairs: list[tuple[str, str]]) -> list[NliDistribution]:
        calls.append(len(pairs))
        return [_dist(0.77, 0.08, 0.15) for _pair in pairs]

    monkeypatch.setattr("pocket_qa.llm.local_nli.score_pairs", fake)
    paraphrase = check_summary(
        [_seg("We should buy more seats before the renewal.", "Alice")],
        "We should purchase additional licenses before the renewal.",
        provider="hybrid",
    )
    assert paraphrase.claims[0].label == "entailed"
    assert "Borderline" in paraphrase.claims[0].rationale

    dropped = check_summary(
        [_seg("I will send the test plan if QA signs off on Thursday.", "Alice")],
        "I will send the test plan on Thursday.",
        provider="hybrid",
    )
    assert dropped.claims[0].label == "entailed"
    assert len(calls) == 2


def test_score_pairs_rejects_a_bad_width(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False, activation_fn=None):
            return [[0.2, 0.8]]

    monkeypatch.setattr("pocket_qa.llm.local_nli._MODEL", FakeModel())
    monkeypatch.setattr("pocket_qa.llm.local_nli._LABELS", ("contradicted", "entailed", "unsupported"))
    with pytest.raises(RuntimeError, match="width"):
        score_pairs([("premise", "hypothesis")])


@pytest.mark.skipif(
    not (available() and weights_cached()),
    reason="Local NLI weights are not cached. This test does not download them.",
)
def test_cached_model_entails_a_near_copy() -> None:
    label, score = score_nli(
        "Alice: We should ship the release on Friday.",
        "We should ship the release on Friday.",
    )
    assert label == "entailed"
    assert score >= 0.5
