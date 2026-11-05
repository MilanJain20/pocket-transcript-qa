"""Score the synthetic benchmark. Numbers in the README come from this module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pocket_qa.evaluate import evaluate
from pocket_qa.models import EvaluateRequest, Segment
from pocket_qa.paths import benchmark_dir, repo_root

RESULTS_START = "<!-- BENCHMARK_RESULTS_START -->"
RESULTS_END = "<!-- BENCHMARK_RESULTS_END -->"


def cases_path() -> Path:
    return benchmark_dir() / "cases.jsonl"


def load_cases() -> list[dict]:
    path = cases_path()
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def _bump(bucket: dict[str, int], gold: bool, pred: bool) -> None:
    if gold and pred:
        bucket["tp"] += 1
    elif pred and not gold:
        bucket["fp"] += 1
    elif gold and not pred:
        bucket["fn"] += 1
    else:
        bucket["tn"] += 1


def score_cases(cases: list[dict] | None = None) -> dict:
    rows = cases if cases is not None else load_cases()
    recording = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for name in ("summary_hallucination", "transcript_fabrication", "speaker_inconsistency", "release_gate")
    }
    claims = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    claim_label_hits = 0
    claim_label_total = 0
    segments = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_case = []

    for case in rows:
        request = EvaluateRequest(
            transcript=[Segment.model_validate(item) for item in case["transcript"]],
            summary=case["summary"],
            audio_id=case.get("audio_id"),
            provider="offline",
            recording_id=case["id"],
            title=case.get("title"),
        )
        report = evaluate(request)
        gold = case["gold"]
        pred_summary = report.summary.has_issue
        pred_transcript = report.transcript.has_issue
        pred_speaker = report.speakers.has_issue
        pred_block = report.gate.decision == "BLOCK"
        _bump(recording["summary_hallucination"], bool(gold["summary_hallucination"]), pred_summary)
        _bump(recording["transcript_fabrication"], bool(gold["transcript_fabrication"]), pred_transcript)
        _bump(recording["speaker_inconsistency"], bool(gold["speaker_inconsistency"]), pred_speaker)
        _bump(recording["release_gate"], bool(gold["block"]), pred_block)

        predicted_claims = {item.claim: item.label for item in report.summary.claims}
        for gold_claim in gold["claims"]:
            label = predicted_claims.get(gold_claim["text"])
            gold_positive = gold_claim["label"] in {"unsupported", "contradicted"}
            pred_positive = label in {"unsupported", "contradicted"}
            if label is None:
                pred_positive = False
            _bump(claims, gold_positive, pred_positive)
            claim_label_total += 1
            if label == gold_claim["label"]:
                claim_label_hits += 1

        predicted_segments = set(report.transcript.fabricated_segment_ids)
        gold_segments = set(gold["fabricated_segment_ids"])
        for segment in case["transcript"]:
            segment_id = segment["id"]
            _bump(segments, segment_id in gold_segments, segment_id in predicted_segments)

        per_case.append(
            {
                "id": case["id"],
                "gold": {
                    "summary_hallucination": gold["summary_hallucination"],
                    "transcript_fabrication": gold["transcript_fabrication"],
                    "speaker_inconsistency": gold["speaker_inconsistency"],
                    "block": gold["block"],
                },
                "pred": {
                    "summary_hallucination": pred_summary,
                    "transcript_fabrication": pred_transcript,
                    "speaker_inconsistency": pred_speaker,
                    "block": pred_block,
                },
                "notes": gold.get("notes", ""),
            }
        )

    metrics = {name: _prf(bucket["tp"], bucket["fp"], bucket["fn"]) for name, bucket in recording.items()}
    metrics["summary_claims"] = _prf(claims["tp"], claims["fp"], claims["fn"])
    metrics["fabricated_segments"] = _prf(segments["tp"], segments["fp"], segments["fn"])
    label_accuracy = claim_label_hits / claim_label_total if claim_label_total else 0.0
    misses = [
        row
        for row in per_case
        if row["gold"] != row["pred"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case_count": len(rows),
        "provider": "offline",
        "synthetic": True,
        "note": (
            "Synthetic benchmark generated by scripts/generate_benchmark.py. "
            "These figures are not an estimate of production precision."
        ),
        "metrics": metrics,
        "claim_label_accuracy": label_accuracy,
        "claim_label_total": claim_label_total,
        "misses": misses,
        "per_case": per_case,
    }


def render_markdown(report: dict) -> str:
    labels = [
        ("summary_hallucination", "Summary hallucination (recording)"),
        ("transcript_fabrication", "Transcript fabrication (recording)"),
        ("speaker_inconsistency", "Speaker inconsistency (recording)"),
        ("release_gate", "Release gate BLOCK (recording)"),
        ("summary_claims", "Hallucinated claims (claim)"),
        ("fabricated_segments", "Fabricated segments (segment)"),
    ]
    lines = [
        f"Generated by `python scripts/run_benchmark.py` at {report['generated_at']} UTC.",
        f"Offline provider. {report['case_count']} synthetic recordings. "
        "Do not read these figures as production precision.",
        "",
        "| Check | Precision | Recall | F1 | TP | FP | FN | Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in labels:
        metric = report["metrics"][key]
        lines.append(
            "| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {tp} | {fp} | {fn} | {support} |".format(
                label=label,
                precision=metric["precision"],
                recall=metric["recall"],
                f1=metric["f1"],
                tp=metric["tp"],
                fp=metric["fp"],
                fn=metric["fn"],
                support=metric["support"],
            )
        )
    accuracy = report["claim_label_accuracy"]
    total = report["claim_label_total"]
    lines.append("")
    lines.append(
        f"Three-way claim label accuracy (entailed / unsupported / contradicted): "
        f"{accuracy:.3f} on {total} claims."
    )
    lines.append("")
    lines.append(
        "A recording is positive for a check when the generator injected that fault. "
        "A prediction is positive when the offline checker emitted at least one finding of that type. "
        "The gate's positive class is the generator's `block` label, which is true when any fault was injected. "
        "Claim and segment rows are micro-averaged over every claim and every segment."
    )
    return "\n".join(lines)


def results_block(report: dict) -> str:
    return f"{RESULTS_START}\n{render_markdown(report)}\n{RESULTS_END}"


def write_results(report: dict) -> Path:
    path = repo_root() / "results" / "benchmark_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def update_readme(report: dict) -> None:
    readme = repo_root() / "README.md"
    text = readme.read_text(encoding="utf-8")
    block = results_block(report)
    if RESULTS_START not in text or RESULTS_END not in text:
        raise RuntimeError("README is missing benchmark result markers.")
    start = text.index(RESULTS_START)
    end = text.index(RESULTS_END) + len(RESULTS_END)
    readme.write_text(text[:start] + block + text[end:], encoding="utf-8")


def readme_matches(report: dict) -> bool:
    """True when the README table contains this run's figures.

    The timestamp line is allowed to differ. CI checks the numbers, which are
    the part that must not be typed by hand.
    """
    text = (repo_root() / "README.md").read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return False
    block = text.split(RESULTS_START, 1)[1].split(RESULTS_END, 1)[0]
    return render_markdown(report).split("\n", 2)[-1] in block or _rows_match(report, block)


def _rows_match(report: dict, block: str) -> bool:
    labels = [
        ("summary_hallucination", "Summary hallucination (recording)"),
        ("transcript_fabrication", "Transcript fabrication (recording)"),
        ("speaker_inconsistency", "Speaker inconsistency (recording)"),
        ("release_gate", "Release gate BLOCK (recording)"),
        ("summary_claims", "Hallucinated claims (claim)"),
        ("fabricated_segments", "Fabricated segments (segment)"),
    ]
    for key, label in labels:
        metric = report["metrics"][key]
        row = (
            "| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {tp} | {fp} | {fn} | {support} |"
        ).format(
            label=label,
            precision=metric["precision"],
            recall=metric["recall"],
            f1=metric["f1"],
            tp=metric["tp"],
            fp=metric["fp"],
            fn=metric["fn"],
            support=metric["support"],
        )
        if row not in block:
            return False
    accuracy = (
        f"{report['claim_label_accuracy']:.3f} on {report['claim_label_total']} claims."
    )
    return accuracy in block


def run_benchmark(*, write_readme: bool = False, check_readme: bool = False) -> dict:
    report = score_cases()
    markdown = render_markdown(report)
    report["markdown"] = markdown
    write_results(report)
    if write_readme:
        update_readme(report)
    if check_readme and not readme_matches(report):
        raise SystemExit("README benchmark table does not match a fresh offline run.")
    return report
