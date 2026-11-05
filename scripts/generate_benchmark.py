#!/usr/bin/env python3
"""Freeze the synthetic benchmark and its audio fixtures.

Every recording is invented for this repository. Nothing here is a Pocket
user, a public-domain speech corpus, or a scraped transcript. Gold labels are
assigned by the injection, not by the detector. The detector is not imported.
"""

from __future__ import annotations

import json
import math
import random
import struct
import wave
from pathlib import Path

from pocket_qa.claims import split_claims
from pocket_qa.paths import audio_dir, benchmark_dir

SAMPLE_RATE = 16000
DURATION_SECONDS = 12


def _embedding(speaker_index: int, salt: int) -> list[float]:
    rng = random.Random(salt)
    vector = [round(rng.uniform(-0.02, 0.02), 4) for _ in range(8)]
    vector[speaker_index] = round(vector[speaker_index] + 1.0, 4)
    return vector


def _finish_claim(text: str) -> str:
    cleaned = text.strip()
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _segments_from_turns(
    turns: list[tuple],
    *,
    embeddings: dict[str, int] | None = None,
    swapped: dict[int, int] | None = None,
) -> list[dict]:
    """turns items are (speaker, name, text) or (speaker, name, text, start, end)."""
    segments: list[dict] = []
    clock = 0.0
    for index, turn in enumerate(turns):
        speaker, name, text = turn[0], turn[1], turn[2]
        if len(turn) == 5:
            start, end = float(turn[3]), float(turn[4])
        else:
            words = max(1, len(text.split()))
            duration = max(1.0, round(words / 2.4, 2))
            start, end = round(clock, 2), round(clock + duration, 2)
            clock = end + 0.35
        segment = {
            "id": f"s{index}",
            "speaker": speaker,
            "speaker_name": name,
            "text": text,
            "start": start,
            "end": end,
        }
        if embeddings is not None:
            speaker_index = embeddings[speaker]
            if swapped and index in swapped:
                speaker_index = swapped[index]
            segment["embedding"] = _embedding(speaker_index, salt=1000 + index * 17 + speaker_index)
        segments.append(segment)
    return segments


def _case(
    case_id: str,
    title: str,
    turns: list[tuple],
    claims: list[tuple[str, str]],
    *,
    audio_id: str | None = None,
    fabricated: list[str] | None = None,
    speaker_fault: bool = False,
    notes: str,
    embeddings: dict[str, int] | None = None,
    swapped: dict[int, int] | None = None,
) -> dict:
    claim_text = [_finish_claim(text) for text, _label in claims]
    labels = [label for _text, label in claims]
    summary = " ".join(claim_text)
    produced = split_claims(summary)
    if produced != claim_text:
        raise SystemExit(
            f"{case_id}: splitter produced {produced!r} but claims are {claim_text!r}"
        )
    segments = _segments_from_turns(turns, embeddings=embeddings, swapped=swapped)
    fabricated_ids = fabricated or []
    for segment_id in fabricated_ids:
        if segment_id not in {segment["id"] for segment in segments}:
            raise SystemExit(f"{case_id}: unknown fabricated id {segment_id}")
    summary_fault = any(label != "entailed" for label in labels)
    transcript_fault = bool(fabricated_ids)
    return {
        "id": case_id,
        "title": title,
        "audio_id": audio_id,
        "summary": summary,
        "transcript": segments,
        "gold": {
            "summary_hallucination": summary_fault,
            "transcript_fabrication": transcript_fault,
            "speaker_inconsistency": speaker_fault,
            "block": summary_fault or transcript_fault or speaker_fault,
            "claims": [{"text": text, "label": label} for text, label in zip(claim_text, labels, strict=True)],
            "fabricated_segment_ids": fabricated_ids,
            "notes": notes,
        },
    }


def _clean_cases() -> list[dict]:
    specs = [
        (
            "clean-01",
            "Friday release check-in",
            [
                ("SPEAKER_00", "Alice", "We should ship the release on Friday."),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
                ("SPEAKER_00", "Alice", "I will send the test plan by Thursday."),
            ],
            [
                ("The team should ship the release on Friday.", "entailed"),
                ("The budget for the launch is 12000 dollars.", "entailed"),
            ],
            "Extractive summary. Names of participants are in the speaker labels.",
        ),
        (
            "clean-02",
            "Hiring debrief",
            [
                ("SPEAKER_00", "Priya", "The candidate has seven years of backend experience."),
                ("SPEAKER_01", "Luis", "We should schedule a systems interview next Tuesday."),
                ("SPEAKER_00", "Priya", "I will share the scorecard with the hiring panel."),
            ],
            [
                ("The candidate has seven years of backend experience.", "entailed"),
                ("The team should schedule a systems interview next Tuesday.", "entailed"),
            ],
            "Extractive hiring debrief with no injected fault.",
        ),
        (
            "clean-03",
            "Customer call",
            [
                ("SPEAKER_00", "Mina", "The customer wants the export finished by April."),
                ("SPEAKER_01", "Owen", "We can deliver the export once billing signs off."),
                ("SPEAKER_00", "Mina", "I will confirm the April date in the follow-up note."),
            ],
            [
                ("The customer wants the export finished by April.", "entailed"),
                ("Owen can deliver the export once billing signs off.", "entailed"),
            ],
            "Light name mention that is also a speaker label.",
        ),
        (
            "clean-04",
            "Incident review",
            [
                ("SPEAKER_00", "Noah", "The checkout errors started at 14:10 UTC."),
                ("SPEAKER_01", "Rina", "A bad config push caused the checkout errors."),
                ("SPEAKER_00", "Noah", "We rolled the config back and the errors stopped."),
            ],
            [
                ("The checkout errors started at 14:10 UTC.", "entailed"),
                ("A bad config push caused the checkout errors.", "entailed"),
            ],
            "Times and cause are copied from the transcript.",
        ),
        (
            "clean-05",
            "Design critique",
            [
                ("SPEAKER_00", "Helen", "The new navigation hides the search box on mobile."),
                ("SPEAKER_01", "Sam", "We should keep the search box visible on mobile."),
                ("SPEAKER_00", "Helen", "I will update the mobile mock tonight."),
            ],
            [
                ("The new navigation hides the search box on mobile.", "entailed"),
                ("The team should keep the search box visible on mobile.", "entailed"),
            ],
            "Design notes restated with the same content words.",
        ),
        (
            "clean-06",
            "Backchannel check",
            [
                ("SPEAKER_00", "Alice", "Can you review the design doc today?"),
                ("SPEAKER_01", "Bob", "Yeah."),
                ("SPEAKER_00", "Alice", "Okay."),
                ("SPEAKER_01", "Bob", "I will send notes after lunch."),
            ],
            [
                ("Bob will send notes after lunch.", "entailed"),
                ("Alice asked for a review of the design doc today.", "entailed"),
            ],
            "Short acknowledgements are real backchannels and must not be turn-pattern faults. "
            "The second claim uses an attribution verb on purpose and is correct.",
        ),
        (
            "clean-07",
            "Second backchannel",
            [
                ("SPEAKER_00", "Priya", "Does the draft include the risk section?"),
                ("SPEAKER_01", "Luis", "Yes."),
                ("SPEAKER_00", "Priya", "Thanks."),
                ("SPEAKER_01", "Luis", "The risk section is on page four."),
            ],
            [
                ("The risk section is on page four.", "entailed"),
                ("The draft does include the risk section.", "entailed"),
            ],
            "Another backchannel pair that should stay PASS.",
        ),
        (
            "clean-08",
            "Speech-energy control",
            [
                ("SPEAKER_00", "Alice", "The roadmap review is on Wednesday.", 1.0, 4.0),
                ("SPEAKER_01", "Bob", "I will bring the roadmap draft.", 4.4, 7.2),
            ],
            [
                ("The roadmap review is on Wednesday.", "entailed"),
                ("Bob will bring the roadmap draft.", "entailed"),
            ],
            "Same words placed on the loud region of the synthetic speech fixture.",
        ),
        (
            "clean-09",
            "Second speech-energy control",
            [
                ("SPEAKER_00", "Mina", "The support queue is down to 40 tickets.", 1.2, 4.5),
                ("SPEAKER_01", "Owen", "We should keep the support queue under 40 tickets.", 4.9, 8.2),
            ],
            [
                ("The support queue is down to 40 tickets.", "entailed"),
                ("The team should keep the support queue under 40 tickets.", "entailed"),
            ],
            "Numbers match, and the audio window is inside the loud region.",
        ),
        (
            "clean-10",
            "Three speaker planning",
            [
                ("SPEAKER_00", "Alice", "The kickoff meeting is Monday at 10."),
                ("SPEAKER_01", "Bob", "I will book the room for the kickoff meeting."),
                ("SPEAKER_02", "Chen", "The agenda should cover staffing and timeline."),
            ],
            [
                ("The kickoff meeting is Monday at 10.", "entailed"),
                ("The agenda should cover staffing and timeline.", "entailed"),
            ],
            "Three consistent speakers and an extractive summary.",
        ),
    ]
    cases = []
    for case_id, title, turns, claims, notes in specs:
        audio_id = "speech" if case_id in {"clean-08", "clean-09"} else None
        cases.append(_case(case_id, title, turns, claims, audio_id=audio_id, notes=notes))
    return cases


def _summary_fault_cases() -> list[dict]:
    unsupported = [
        (
            "sum-unsup-01",
            "Invented acquisition",
            [
                ("SPEAKER_00", "Alice", "We should ship the release on Friday."),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
            ],
            [
                ("The team should ship the release on Friday.", "entailed"),
                ("The board approved a 4000000 dollar acquisition of Northwind Robotics.", "unsupported"),
            ],
            "The acquisition, the board, and the amount are not in the transcript.",
        ),
        (
            "sum-unsup-02",
            "Invented headcount",
            [
                ("SPEAKER_00", "Priya", "The candidate has seven years of backend experience."),
                ("SPEAKER_01", "Luis", "We should schedule a systems interview next Tuesday."),
            ],
            [
                ("The candidate has seven years of backend experience.", "entailed"),
                ("The company will open a Berlin office with 80 new hires.", "unsupported"),
            ],
            "Berlin, the office, and the hire count are injected.",
        ),
        (
            "sum-unsup-03",
            "Invented legal hold",
            [
                ("SPEAKER_00", "Mina", "The customer wants the export finished by April."),
                ("SPEAKER_01", "Owen", "We can deliver the export once billing signs off."),
            ],
            [
                ("The customer wants the export finished by April.", "entailed"),
                ("Counsel placed a litigation hold on the export servers.", "unsupported"),
            ],
            "No legal hold is discussed.",
        ),
        (
            "sum-unsup-04",
            "Invented outage cause",
            [
                ("SPEAKER_00", "Noah", "The checkout errors started at 14:10 UTC."),
                ("SPEAKER_01", "Rina", "We rolled the config back and the errors stopped."),
            ],
            [
                ("The checkout errors started at 14:10 UTC.", "entailed"),
                ("A firewall vendor named BlueHarbor caused a nationwide outage.", "unsupported"),
            ],
            "The vendor and the nationwide outage are not in the transcript.",
        ),
        (
            "sum-unsup-05",
            "Invented revenue",
            [
                ("SPEAKER_00", "Helen", "The new navigation hides the search box on mobile."),
                ("SPEAKER_01", "Sam", "We should keep the search box visible on mobile."),
            ],
            [
                ("The new navigation hides the search box on mobile.", "entailed"),
                ("Mobile search changes lifted quarterly revenue by 900000 dollars.", "unsupported"),
            ],
            "A revenue outcome was added.",
        ),
        (
            "sum-unsup-06",
            "Invented attendee",
            [
                ("SPEAKER_00", "Alice", "The kickoff meeting is Monday at 10."),
                ("SPEAKER_01", "Bob", "I will book the room for the kickoff meeting."),
            ],
            [
                ("The kickoff meeting is Monday at 10.", "entailed"),
                ("The chief executive will fly in from Singapore for the kickoff.", "unsupported"),
            ],
            "A travel plan for an executive is not in the transcript.",
        ),
        (
            "sum-unsup-07",
            "Invented dependency",
            [
                ("SPEAKER_00", "Chen", "The agenda should cover staffing and timeline."),
                ("SPEAKER_01", "Alice", "I will send the staffing notes today."),
            ],
            [
                ("The agenda should cover staffing and timeline.", "entailed"),
                ("The timeline depends on a signed contract with Orbit Labs.", "unsupported"),
            ],
            "Orbit Labs is not mentioned.",
        ),
        (
            "sum-unsup-08",
            "Invented metric",
            [
                ("SPEAKER_00", "Mina", "The support queue is down to 40 tickets."),
                ("SPEAKER_01", "Owen", "We should keep the support queue under 40 tickets."),
            ],
            [
                ("The support queue is down to 40 tickets.", "entailed"),
                ("Customer satisfaction reached 99 percent after the queue dropped.", "unsupported"),
            ],
            "A satisfaction figure was added.",
        ),
    ]
    contradicted = [
        (
            "sum-neg-01",
            "Negated launch",
            [
                ("SPEAKER_00", "Alice", "We will launch in March."),
                ("SPEAKER_01", "Bob", "The launch checklist is already in the doc."),
            ],
            [
                ("The launch checklist is already in the doc.", "entailed"),
                ("The team will not launch in March.", "contradicted"),
            ],
            "Polarity of the launch is flipped.",
        ),
        (
            "sum-neg-02",
            "Negated rollback",
            [
                ("SPEAKER_00", "Noah", "We did roll the config back and the errors stopped."),
                ("SPEAKER_01", "Rina", "The checkout errors started at 14:10 UTC."),
            ],
            [
                ("The checkout errors started at 14:10 UTC.", "entailed"),
                ("We did not roll the config back and the errors stopped.", "contradicted"),
            ],
            "The summary denies a rollback the transcript states.",
        ),
        (
            "sum-num-01",
            "Wrong budget",
            [
                ("SPEAKER_00", "Bob", "The budget for the launch is 12000 dollars."),
                ("SPEAKER_01", "Alice", "We should ship the release on Friday."),
            ],
            [
                ("The team should ship the release on Friday.", "entailed"),
                ("The budget for the launch is 40000 dollars.", "contradicted"),
            ],
            "The budget figure is replaced.",
        ),
        (
            "sum-num-02",
            "Wrong experience",
            [
                ("SPEAKER_00", "Priya", "The candidate has 7 years of backend experience."),
                ("SPEAKER_01", "Luis", "We should schedule a systems interview next Tuesday."),
            ],
            [
                ("The team should schedule a systems interview next Tuesday.", "entailed"),
                ("The candidate has 15 years of backend experience.", "contradicted"),
            ],
            "Years of experience are changed.",
        ),
        (
            "sum-num-03",
            "Wrong queue size",
            [
                ("SPEAKER_00", "Mina", "The support queue is down to 40 tickets."),
                ("SPEAKER_01", "Owen", "We should keep the support queue under 40 tickets."),
            ],
            [
                ("The team should keep the support queue under 40 tickets.", "entailed"),
                ("The support queue is down to 400 tickets.", "contradicted"),
            ],
            "The ticket count is off by an order of magnitude.",
        ),
        (
            "sum-num-04",
            "Wrong start time",
            [
                ("SPEAKER_00", "Noah", "The checkout errors started at 14:10 UTC."),
                ("SPEAKER_01", "Rina", "A bad config push caused the checkout errors."),
            ],
            [
                ("A bad config push caused the checkout errors.", "entailed"),
                ("The checkout errors started at 09:45 UTC.", "contradicted"),
            ],
            "The incident clock time is changed.",
        ),
        (
            "sum-ant-01",
            "Accepted versus rejected",
            [
                ("SPEAKER_00", "Helen", "The client accepted the proposal for the renewal."),
                ("SPEAKER_01", "Sam", "I will send the renewal paperwork today."),
            ],
            [
                ("Sam will send the renewal paperwork today.", "entailed"),
                ("The client rejected the proposal for the renewal.", "contradicted"),
            ],
            "Uses the closed antonym pair accepted/rejected.",
        ),
        (
            "sum-ant-02",
            "Increase versus decrease",
            [
                ("SPEAKER_00", "Alice", "We will increase the budget for the launch."),
                ("SPEAKER_01", "Bob", "The launch checklist is already in the doc."),
            ],
            [
                ("The launch checklist is already in the doc.", "entailed"),
                ("The team will decrease the budget for the launch.", "contradicted"),
            ],
            "Uses the closed antonym pair increase/decrease.",
        ),
        (
            "sum-cond-01",
            "Dropped condition",
            [
                ("SPEAKER_00", "Alice", "I will send the test plan if QA signs off on Thursday."),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
            ],
            [
                ("The budget for the launch is 12000 dollars.", "entailed"),
                ("I will send the test plan on Thursday.", "unsupported"),
            ],
            "Dropped QA condition. Overlap stays high, so the offline heuristic is expected to miss it.",
        ),
        (
            "sum-cond-02",
            "Dropped security condition",
            [
                ("SPEAKER_00", "Mina", "We can launch in March if the security review is finished."),
                ("SPEAKER_01", "Owen", "The launch checklist is already in the doc."),
            ],
            [
                ("The launch checklist is already in the doc.", "entailed"),
                ("We can launch in March.", "unsupported"),
            ],
            "A conditional launch is restated as unconditional. Expected offline miss.",
        ),
        (
            "sum-para-01",
            "Synonym paraphrase",
            [
                ("SPEAKER_00", "Alice", "We should buy more seats before the renewal."),
                ("SPEAKER_01", "Bob", "I will ask finance about the renewal."),
            ],
            [
                ("Bob will ask finance about the renewal.", "entailed"),
                ("We should purchase additional licenses before the renewal.", "entailed"),
            ],
            "True paraphrase with different content words. Expected offline false positive.",
        ),
        (
            "sum-para-02",
            "Second synonym paraphrase",
            [
                ("SPEAKER_00", "Noah", "The customer asked us to fix the outage before noon."),
                ("SPEAKER_01", "Rina", "I will page the on-call engineer now."),
            ],
            [
                ("Rina will page the on-call engineer now.", "entailed"),
                ("The client requested a repair of the incident prior to midday.", "entailed"),
            ],
            "A second paraphrase the lexical checker is not designed to accept.",
        ),
    ]
    cases = []
    for case_id, title, turns, claims, notes in unsupported + contradicted:
        cases.append(_case(case_id, title, turns, claims, notes=notes))
    return cases


def _transcript_cases() -> list[dict]:
    cases = []
    fillers = [
        ("tr-filler-01", "Thanks for watching.", "Classic Whisper-style filler inserted as its own segment."),
        ("tr-filler-02", "Please subscribe to the channel.", "A second published filler phrase."),
        ("tr-filler-03", "See you in the next video.", "Video-outro hallucination phrase."),
        (
            "tr-filler-04",
            "Subtitles by the amara.org community.",
            "Amara.org subtitle credit, a known non-speech insertion.",
        ),
    ]
    for case_id, filler, notes in fillers:
        turns = [
            ("SPEAKER_00", "Alice", "We should ship the release on Friday."),
            ("SPEAKER_01", "Bob", filler),
            ("SPEAKER_00", "Alice", "I will send the test plan by Thursday."),
        ]
        claims = [("The team should ship the release on Friday.", "entailed")]
        cases.append(
            _case(
                case_id,
                "Filler insertion",
                turns,
                claims,
                fabricated=["s1"],
                notes=notes,
            )
        )

    repeated = "We should revisit the pricing page."
    cases.append(
        _case(
            "tr-repeat-01",
            "Four identical sentences",
            [
                ("SPEAKER_00", "Alice", repeated),
                ("SPEAKER_00", "Alice", repeated),
                ("SPEAKER_00", "Alice", repeated),
                ("SPEAKER_00", "Alice", repeated),
            ],
            [("The team should revisit the pricing page.", "entailed")],
            fabricated=["s0", "s1", "s2", "s3"],
            notes="The same sentence is repeated four times in a row.",
        )
    )
    other = "The support queue needs a weekend owner."
    cases.append(
        _case(
            "tr-repeat-02",
            "Loop after a real turn",
            [
                ("SPEAKER_00", "Mina", "The customer wants the export finished by April."),
                ("SPEAKER_01", "Owen", other),
                ("SPEAKER_01", "Owen", other),
                ("SPEAKER_01", "Owen", other),
            ],
            [("The customer wants the export finished by April.", "entailed")],
            fabricated=["s1", "s2", "s3"],
            notes="A three-peat follows one genuine sentence. Only the loop is labeled fabricated.",
        )
    )
    loop = "I don't know. " * 4
    cases.append(
        _case(
            "tr-repeat-03",
            "Phrase loop inside one segment",
            [
                ("SPEAKER_00", "Noah", "The checkout errors started at 14:10 UTC."),
                ("SPEAKER_01", "Rina", loop.strip()),
            ],
            [("The checkout errors started at 14:10 UTC.", "entailed")],
            fabricated=["s1"],
            notes="One segment is an internal repetition loop.",
        )
    )
    cases.append(
        _case(
            "tr-repeat-04",
            "Second full-sentence loop",
            [
                ("SPEAKER_00", "Helen", "The new navigation hides the search box on mobile."),
                ("SPEAKER_01", "Sam", "The new navigation hides the search box on mobile."),
                ("SPEAKER_00", "Helen", "The new navigation hides the search box on mobile."),
            ],
            [("The new navigation hides the search box on mobile.", "entailed")],
            fabricated=["s0", "s1", "s2"],
            notes="Three identical sentences across two speaker labels.",
        )
    )

    fast = (
        "We should ship the release on Friday and send the test plan by Thursday "
        "without slipping the budget review."
    )
    cases.append(
        _case(
            "tr-rate-01",
            "Impossible word rate",
            [
                ("SPEAKER_00", "Alice", fast, 0.0, 1.4),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars.", 2.0, 6.0),
            ],
            [("The budget for the launch is 12000 dollars.", "entailed")],
            fabricated=["s0"],
            notes="A long sentence is stamped into 1.4 seconds.",
        )
    )
    cases.append(
        _case(
            "tr-rate-02",
            "Second impossible word rate",
            [
                (
                    "SPEAKER_00",
                    "Priya",
                    (
                        "The candidate has seven years of backend experience "
                        "and we should schedule the systems interview Tuesday morning."
                    ),
                    0.2,
                    1.5,
                ),
                ("SPEAKER_01", "Luis", "I will share the scorecard with the hiring panel.", 2.0, 6.0),
            ],
            [("Luis will share the scorecard with the hiring panel.", "entailed")],
            fabricated=["s0"],
            notes="Another contentful span with an impossible speaking rate.",
        )
    )
    cases.append(
        _case(
            "tr-time-01",
            "End before start",
            [
                ("SPEAKER_00", "Noah", "The checkout errors started at 14:10 UTC.", 5.0, 4.0),
                ("SPEAKER_01", "Rina", "A bad config push caused the checkout errors.", 6.0, 10.0),
            ],
            [("A bad config push caused the checkout errors.", "entailed")],
            fabricated=["s0"],
            notes="The first segment has a timestamp that runs backwards.",
        )
    )

    quiet = "The quarterly forecast looks strong overall today."
    cases.append(
        _case(
            "tr-silence-01",
            "Text on the silence fixture",
            [("SPEAKER_00", "Alice", quiet, 1.0, 5.0)],
            [(quiet, "entailed")],
            audio_id="silence",
            fabricated=["s0"],
            notes="Content words sit on the synthetic near-silence waveform.",
        )
    )
    cases.append(
        _case(
            "tr-silence-02",
            "Second silence window",
            [
                ("SPEAKER_00", "Bob", "The budget for the launch is 12000 dollars.", 0.8, 4.2),
            ],
            [("The budget for the launch is 12000 dollars.", "entailed")],
            audio_id="silence",
            fabricated=["s0"],
            notes="A faithful summary of text that the audio does not support.",
        )
    )
    cases.append(
        _case(
            "tr-silence-03",
            "Quiet tail of the speech fixture",
            [
                ("SPEAKER_00", "Mina", "The customer wants the export finished by April.", 9.3, 11.6),
            ],
            [("The customer wants the export finished by April.", "entailed")],
            audio_id="speech",
            fabricated=["s0"],
            notes="The speech fixture is loud only until 8.5 seconds. This span is in the quiet tail.",
        )
    )

    subtle = [
        (
            "tr-subtle-01",
            "Fluent unspoken aside",
            "The facilities team already ordered a second espresso machine for the studio.",
            "Injected fluent sentence with normal timing and no audio. The offline checker has no cue.",
        ),
        (
            "tr-subtle-02",
            "Fluent unspoken metric",
            "Last spring the Dublin pod cut build times by half.",
            "Another fluent insertion with ordinary word rate. Expected miss.",
        ),
        (
            "tr-subtle-03",
            "Fluent unspoken promise",
            "Marketing promised a printed booklet for every attendee.",
            "Surface form is ordinary meeting speech. Expected miss.",
        ),
    ]
    for case_id, title, invented, notes in subtle:
        turns = [
            ("SPEAKER_00", "Alice", "We should ship the release on Friday."),
            ("SPEAKER_01", "Bob", invented),
        ]
        cases.append(
            _case(
                case_id,
                title,
                turns,
                [("The team should ship the release on Friday.", "entailed")],
                fabricated=["s1"],
                notes=notes,
            )
        )
    return cases


def _speaker_cases() -> list[dict]:
    cases = []
    flips = [
        (
            "spk-flip-01",
            [
                ("SPEAKER_00", "Alice", "The launch date is"),
                ("SPEAKER_01", "Bob", "friday afternoon at the earliest."),
            ],
            "The launch date is friday afternoon at the earliest.",
        ),
        (
            "spk-flip-02",
            [
                ("SPEAKER_00", "Priya", "The candidate will"),
                ("SPEAKER_01", "Luis", "need a systems interview next Tuesday."),
            ],
            "The candidate will need a systems interview next Tuesday.",
        ),
        (
            "spk-flip-03",
            [
                ("SPEAKER_00", "Mina", "We can deliver the export"),
                ("SPEAKER_01", "Owen", "once billing signs off."),
            ],
            "We can deliver the export once billing signs off.",
        ),
        (
            "spk-flip-04",
            [
                ("SPEAKER_00", "Helen", "The new navigation hides"),
                ("SPEAKER_01", "Sam", "the search box on mobile."),
            ],
            "The new navigation hides the search box on mobile.",
        ),
    ]
    for case_id, turns, claim in flips:
        cases.append(
            _case(
                case_id,
                "Mid-sentence speaker flip",
                turns,
                [(claim, "entailed")],
                speaker_fault=True,
                notes="The sentence continues in lowercase under a different speaker.",
            )
        )

    ping = [
        ("SPEAKER_00", "Alice", "Budget"),
        ("SPEAKER_01", "Bob", "Slipped"),
        ("SPEAKER_00", "Alice", "Again"),
        ("SPEAKER_01", "Bob", "Badly"),
        ("SPEAKER_00", "Alice", "Fix"),
        ("SPEAKER_01", "Bob", "Tomorrow"),
    ]
    cases.append(
        _case(
            "spk-ping-01",
            "Ping-pong content words",
            ping,
            [("The notes say budget slipped again badly fix tomorrow.", "entailed")],
            speaker_fault=True,
            notes="Six alternating micro-turns and none of them are backchannels.",
        )
    )
    cases.append(
        _case(
            "spk-ping-02",
            "Second ping-pong",
            [
                ("SPEAKER_00", "Mina", "Export"),
                ("SPEAKER_01", "Owen", "Blocked"),
                ("SPEAKER_00", "Mina", "Billing"),
                ("SPEAKER_01", "Owen", "Waiting"),
                ("SPEAKER_00", "Mina", "Escalate"),
                ("SPEAKER_01", "Owen", "Today"),
            ],
            [("The notes say export blocked billing waiting escalate today.", "entailed")],
            speaker_fault=True,
            notes="Another non-backchannel ping-pong.",
        )
    )
    cases.append(
        _case(
            "spk-ping-03",
            "Third ping-pong",
            [
                ("SPEAKER_00", "Helen", "Search"),
                ("SPEAKER_01", "Sam", "Hidden"),
                ("SPEAKER_00", "Helen", "Mobile"),
                ("SPEAKER_01", "Sam", "Broken"),
                ("SPEAKER_00", "Helen", "Restore"),
                ("SPEAKER_01", "Sam", "Tonight"),
            ],
            [("The notes say search hidden mobile broken restore tonight.", "entailed")],
            speaker_fault=True,
            notes="Third implausible turn pattern.",
        )
    )

    attributions = [
        (
            "spk-attr-01",
            [
                ("SPEAKER_00", "Alice", "I will present the forecast on Monday."),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
            ],
            "Alice said the budget for the launch is 12000 dollars.",
            "Bob stated the budget. The summary gives it to Alice.",
        ),
        (
            "spk-attr-02",
            [
                ("SPEAKER_00", "Priya", "I will share the scorecard with the hiring panel."),
                ("SPEAKER_01", "Luis", "The candidate has seven years of backend experience."),
            ],
            "Priya said the candidate has seven years of backend experience.",
            "Luis stated the experience. The summary gives it to Priya.",
        ),
        (
            "spk-attr-03",
            [
                ("SPEAKER_00", "Mina", "I will confirm the April date in the follow-up note."),
                ("SPEAKER_01", "Owen", "The customer wants the export finished by April."),
            ],
            "According to Mina, the customer wants the export finished by April.",
            "Owen stated the customer request.",
        ),
        (
            "spk-attr-04",
            [
                ("SPEAKER_00", "Noah", "I will write the incident timeline tonight."),
                ("SPEAKER_01", "Rina", "A bad config push caused the checkout errors."),
            ],
            "Noah noted that a bad config push caused the checkout errors.",
            "Rina stated the cause.",
        ),
        (
            "spk-attr-05",
            [
                ("SPEAKER_00", "Helen", "I will update the mobile mock tonight."),
                ("SPEAKER_01", "Sam", "The new navigation hides the search box on mobile."),
            ],
            "Helen mentioned that the new navigation hides the search box on mobile.",
            "Sam stated the navigation problem.",
        ),
    ]
    for case_id, turns, claim, notes in attributions:
        cases.append(
            _case(
                case_id,
                "Wrong speaker in the summary",
                turns,
                [(claim, "entailed")],
                speaker_fault=True,
                notes=notes,
            )
        )

    cases.append(
        _case(
            "spk-passive-01",
            "Passive attribution",
            [
                ("SPEAKER_00", "Alice", "I will present the forecast on Monday."),
                ("SPEAKER_01", "Bob", "I requested the extra budget of 12000 dollars."),
            ],
            [("The extra budget of 12000 dollars was requested by Alice.", "entailed")],
            speaker_fault=True,
            notes="Passive 'requested by Alice' is a real speaker error. The offline pattern list does not parse it.",
        )
    )
    cases.append(
        _case(
            "spk-passive-02",
            "Second passive attribution",
            [
                ("SPEAKER_00", "Priya", "I will share the scorecard with the hiring panel."),
                ("SPEAKER_01", "Luis", "I scheduled the systems interview next Tuesday."),
            ],
            [("The systems interview next Tuesday was scheduled by Priya.", "entailed")],
            speaker_fault=True,
            notes="Second passive attribution the verb pattern is expected to miss.",
        )
    )

    alice_bob = {"SPEAKER_00": 0, "SPEAKER_01": 1}
    base_turns = [
        ("SPEAKER_00", "Alice", "The kickoff meeting is Monday at 10."),
        ("SPEAKER_00", "Alice", "I will send the staffing notes today."),
        ("SPEAKER_00", "Alice", "The agenda should cover staffing and timeline."),
        ("SPEAKER_00", "Alice", "I will book a follow-up on Wednesday."),
        ("SPEAKER_01", "Bob", "I will book the room for the kickoff meeting."),
        ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
        ("SPEAKER_01", "Bob", "I will bring the printed agenda."),
    ]
    for case_id, swap_at in (("spk-embed-01", 2), ("spk-embed-02", 3), ("spk-embed-03", 1)):
        cases.append(
            _case(
                case_id,
                "Voice cluster mismatch",
                base_turns,
                [
                    ("The kickoff meeting is Monday at 10.", "entailed"),
                    ("The budget for the launch is 12000 dollars.", "entailed"),
                ],
                speaker_fault=True,
                notes=f"Segment s{swap_at} is labeled Alice but uses Bob's synthetic embedding.",
                embeddings=alice_bob,
                swapped={swap_at: 1},
            )
        )
    return cases


def _mixed_cases() -> list[dict]:
    return [
        _case(
            "mix-01",
            "Unsupported claim and wrong speaker",
            [
                ("SPEAKER_00", "Alice", "I will present the forecast on Monday."),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
            ],
            [
                ("Alice said the budget for the launch is 12000 dollars.", "entailed"),
                ("The board approved a 4000000 dollar acquisition of Northwind Robotics.", "unsupported"),
            ],
            speaker_fault=True,
            notes="Both a wrong attribution and an unsupported claim are injected.",
        ),
        _case(
            "mix-02",
            "Filler plus a contradicted budget",
            [
                ("SPEAKER_00", "Alice", "We should ship the release on Friday."),
                ("SPEAKER_01", "Bob", "The budget for the launch is 12000 dollars."),
                ("SPEAKER_01", "Bob", "Thanks for watching."),
            ],
            [
                ("The team should ship the release on Friday.", "entailed"),
                ("The budget for the launch is 40000 dollars.", "contradicted"),
            ],
            fabricated=["s2"],
            notes="A filler segment and a wrong budget figure.",
        ),
        _case(
            "mix-03",
            "Repetition plus wrong speaker",
            [
                ("SPEAKER_00", "Mina", "I will confirm the April date in the follow-up note."),
                ("SPEAKER_01", "Owen", "The customer wants the export finished by April."),
                ("SPEAKER_01", "Owen", "The customer wants the export finished by April."),
                ("SPEAKER_01", "Owen", "The customer wants the export finished by April."),
            ],
            [("Mina said the customer wants the export finished by April.", "entailed")],
            fabricated=["s1", "s2", "s3"],
            speaker_fault=True,
            notes="The export line is both repeated and given to the wrong speaker. "
            "Three copies are enough for the loop rule, and the first copy is part of that loop.",
        ),
        _case(
            "mix-04",
            "Silence plus a negated launch",
            [
                ("SPEAKER_00", "Alice", "We will launch in March.", 1.0, 4.0),
                ("SPEAKER_01", "Bob", "The launch checklist is already in the doc.", 4.4, 8.0),
            ],
            [
                ("The launch checklist is already in the doc.", "entailed"),
                ("The team will not launch in March.", "contradicted"),
            ],
            audio_id="silence",
            fabricated=["s0", "s1"],
            notes="Both spans sit on silence, and the summary flips the launch.",
        ),
    ]


def generate_cases() -> list[dict]:
    cases = _clean_cases() + _summary_fault_cases() + _transcript_cases() + _speaker_cases() + _mixed_cases()
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate case ids")
    return cases


def _write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        frames = b"".join(struct.pack("<h", max(-32767, min(32767, sample))) for sample in samples)
        handle.writeframes(frames)


def write_audio() -> None:
    """Deterministic 12 second fixtures. Not a recording of a person."""
    silence_rng = random.Random(7)
    speech_rng = random.Random(11)
    silence: list[int] = []
    speech: list[int] = []
    total = SAMPLE_RATE * DURATION_SECONDS
    for index in range(total):
        silence.append(silence_rng.randint(-40, 40))
        seconds = index / SAMPLE_RATE
        if 0.4 <= seconds <= 8.5:
            envelope = 0.55 + 0.45 * math.sin(2 * math.pi * 3 * seconds)
            speech.append(int(speech_rng.randint(-6000, 6000) * envelope))
        else:
            speech.append(speech_rng.randint(-40, 40))
    target = audio_dir()
    _write_wav(target / "silence.wav", silence)
    _write_wav(target / "speech.wav", speech)


def write_cases(cases: list[dict]) -> Path:
    path = benchmark_dir() / "cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=True, sort_keys=True) for case in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    cases = generate_cases()
    write_audio()
    path = write_cases(cases)
    print(f"Wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
