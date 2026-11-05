"""Unit tests for the offline checkers."""

from pocket_qa.audio import segment_rms
from pocket_qa.claims import split_claims
from pocket_qa.gate import decide
from pocket_qa.models import GateThresholds, Segment
from pocket_qa.paths import audio_dir
from pocket_qa.speaker_check import check_speakers
from pocket_qa.summary_check import check_summary
from pocket_qa.transcript_check import check_transcript


def _seg(speaker: str, text: str, start: float, end: float, name: str | None = None, embedding=None) -> Segment:
    return Segment(
        speaker=speaker,
        speaker_name=name or speaker,
        text=text,
        start=start,
        end=end,
        embedding=embedding,
    )


def test_split_claims_keeps_abbreviations_and_decimals() -> None:
    claims = split_claims("Dr. Shah approved a 3.5 percent raise.\n- The team will ship on Friday.")
    assert claims == [
        "Dr. Shah approved a 3.5 percent raise.",
        "The team will ship on Friday.",
    ]


def test_grounded_summary_passes() -> None:
    segments = [
        _seg("SPEAKER_00", "We should ship the release on Friday.", 0, 3, "Alice"),
        _seg("SPEAKER_01", "The budget for the launch is 12000 dollars.", 3.4, 7, "Bob"),
    ]
    summary = (
        "The team should ship the release on Friday. "
        "The budget for the launch is 12000 dollars."
    )
    report = check_summary(segments, summary)
    assert [item.label for item in report.claims] == ["entailed", "entailed"]
    assert report.has_issue is False


def test_unsupported_and_contradicted_claims() -> None:
    segments = [
        _seg("SPEAKER_00", "We will launch in March.", 0, 3, "Alice"),
        _seg("SPEAKER_01", "The budget for the launch is 12000 dollars.", 3.4, 7, "Bob"),
    ]
    summary = (
        "The team will not launch in March. "
        "The board approved a 4000000 dollar acquisition of Northwind Robotics."
    )
    report = check_summary(segments, summary)
    labels = {item.claim: item.label for item in report.claims}
    assert any(label == "contradicted" for label in labels.values())
    assert any(label == "unsupported" for label in labels.values())
    contradicted = next(item for item in report.claims if item.label == "contradicted")
    assert contradicted.evidence


def test_filler_repetition_and_word_rate() -> None:
    segments = [
        _seg("A", "Thanks for watching.", 0, 2, "Alice"),
        _seg("B", "We should revisit the pricing page.", 2, 5, "Bob"),
        _seg("B", "We should revisit the pricing page.", 5, 8, "Bob"),
        _seg("B", "We should revisit the pricing page.", 8, 11, "Bob"),
        _seg(
            "A",
            "We should ship the release on Friday and send the test plan by Thursday without delay today.",
            12,
            13,
            "Alice",
        ),
    ]
    report = check_transcript(segments)
    kinds = {flag.kind for flag in report.flags}
    assert "filler_hallucination" in kinds
    assert "repetition_loop" in kinds
    assert "word_rate" in kinds


def test_silence_and_speech_energy() -> None:
    text = "The quarterly forecast looks strong overall today."
    silent = [_seg("A", text, 1.0, 5.0, "Alice")]
    loud = [_seg("A", text, 1.0, 4.0, "Alice")]
    silence = check_transcript(silent, audio_path=str(audio_dir() / "silence.wav"))
    speech = check_transcript(loud, audio_path=str(audio_dir() / "speech.wav"))
    assert silence.has_issue is True
    assert silence.flags[0].kind == "silent_audio"
    assert speech.has_issue is False
    assert segment_rms(audio_dir() / "silence.wav", 1, 5) < segment_rms(audio_dir() / "speech.wav", 1, 4)


def test_speaker_flip_and_backchannel() -> None:
    flipped = [
        _seg("SPEAKER_00", "The launch date is", 0, 2, "Alice"),
        _seg("SPEAKER_01", "friday afternoon at the earliest.", 2.2, 5, "Bob"),
    ]
    assert check_speakers(flipped, "The launch date is friday afternoon at the earliest.").has_issue
    backchannel = [
        _seg("SPEAKER_00", "Can you review the design doc today?", 0, 3, "Alice"),
        _seg("SPEAKER_01", "Yeah.", 3.2, 3.8, "Bob"),
        _seg("SPEAKER_00", "Okay.", 4.0, 4.6, "Alice"),
        _seg("SPEAKER_01", "I will send notes after lunch.", 5, 8, "Bob"),
    ]
    assert check_speakers(backchannel, "Bob will send notes after lunch.").has_issue is False


def test_wrong_attribution_and_passive_miss() -> None:
    segments = [
        _seg("SPEAKER_00", "I will present the forecast on Monday.", 0, 3, "Alice"),
        _seg("SPEAKER_01", "The budget for the launch is 12000 dollars.", 3.4, 7, "Bob"),
    ]
    wrong = check_speakers(segments, "Alice said the budget for the launch is 12000 dollars.")
    assert any(flag.kind == "wrong_speaker_attribution" for flag in wrong.flags)
    passive = check_speakers(
        [
            _seg("SPEAKER_00", "I will present the forecast on Monday.", 0, 3, "Alice"),
            _seg("SPEAKER_01", "I requested the extra budget of 12000 dollars.", 3.4, 7, "Bob"),
        ],
        "The extra budget of 12000 dollars was requested by Alice.",
    )
    assert passive.has_issue is False


def test_embedding_mismatch() -> None:
    def emb(index: int, salt: int) -> list[float]:
        vector = [0.0] * 8
        vector[index] = 1.0
        vector[(index + 1) % 8] = salt / 1000
        return vector

    segments = [
        _seg("SPEAKER_00", "The kickoff meeting is Monday at 10.", 0, 3, "Alice", emb(0, 1)),
        _seg("SPEAKER_00", "I will send the staffing notes today.", 3.4, 6, "Alice", emb(1, 1)),
        _seg("SPEAKER_00", "The agenda should cover staffing and timeline.", 6.4, 10, "Alice", emb(0, 2)),
        _seg("SPEAKER_00", "I will book a follow-up on Wednesday.", 10.4, 14, "Alice", emb(0, 3)),
        _seg("SPEAKER_01", "I will book the room for the kickoff meeting.", 14.4, 18, "Bob", emb(1, 4)),
        _seg("SPEAKER_01", "The budget for the launch is 12000 dollars.", 18.4, 22, "Bob", emb(1, 5)),
    ]
    report = check_speakers(segments, "The kickoff meeting is Monday at 10.")
    assert any(flag.kind == "voice_profile_mismatch" for flag in report.flags)


def test_gate_blocks_on_contradiction() -> None:
    segments = [_seg("A", "We will launch in March.", 0, 3, "Alice")]
    summary = check_summary(segments, "The team will not launch in March.")
    transcript = check_transcript(segments)
    speakers = check_speakers(segments, "The team will not launch in March.")
    decision = decide(summary, transcript, speakers, GateThresholds())
    assert decision.decision == "BLOCK"
    assert any("contradicted" in reason for reason in decision.reasons)
