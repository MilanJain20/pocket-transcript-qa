#!/usr/bin/env python3
"""Compare offline, local-nli, and hybrid claim support on the synthetic benchmark.

This script downloads the NLI checkpoint on first use and runs it on CPU.
It does not call a hosted model API and it does not read the recordings/ folder.
CI does not run it. If the extra or the weights cannot be loaded, it exits
without writing numbers.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pocket_qa.bench import load_cases, render_comparison, score_cases
from pocket_qa.paths import repo_root


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _shared_counts(report: dict) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    transcript = report["metrics"]["transcript_fabrication"]
    speaker = report["metrics"]["speaker_inconsistency"]
    return (
        (transcript["tp"], transcript["fp"], transcript["fn"]),
        (speaker["tp"], speaker["fp"], speaker["fn"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure heuristic vs local NLI vs hybrid on the synthetic set.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["offline", "local-nli", "hybrid"],
        choices=["offline", "local-nli", "hybrid"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "results" / "provider_comparison.json",
    )
    args = parser.parse_args()
    needs_model = any(name != "offline" for name in args.providers)
    model_name = ""
    load_seconds = 0.0
    if needs_model:
        from pocket_qa.llm.local_nli import available, load_model, model_id, score_nli

        if not available():
            sys.exit(
                "local-nli is not installed. Install with: pip install -e \".[local-nli]\"\n"
                "The default offline benchmark does not need it, and CI does not run this script."
            )
        model_name = model_id()
        started = time.perf_counter()
        try:
            load_model()
            score_nli("The budget for the launch is 12000 dollars.", "The budget for the launch is 12000 dollars.")
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                f"The local NLI model could not be loaded ({exc.__class__.__name__}: {exc}). "
                "No numbers were written."
            )
        load_seconds = time.perf_counter() - started

    cases = load_cases()
    providers: dict[str, dict] = {}
    for name in args.providers:
        print(f"Scoring provider={name} on {len(cases)} synthetic recordings...", flush=True)
        report = score_cases(cases, provider=name, time_cases=True, include_claim_errors=True)
        report.pop("per_case", None)
        report.pop("markdown", None)
        providers[name] = report
        summary = report["metrics"]["summary_hallucination"]
        print(
            f"  summary F1={summary['f1']:.6f} tp={summary['tp']} fp={summary['fp']} fn={summary['fn']} "
            f"claim_acc={report['claim_label_accuracy']:.6f} "
            f"mean_s={report['latency_seconds']['mean_seconds']:.6f}",
            flush=True,
        )

    offline_shared = _shared_counts(providers["offline"]) if "offline" in providers else None
    shared_match = True
    if offline_shared is not None:
        shared_match = all(_shared_counts(report) == offline_shared for report in providers.values())

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_name,
        "device": "cpu",
        "model_load_seconds": load_seconds,
        "results_path": str(args.output),
        "synthetic": True,
        "case_count": len(cases),
        "shared_checks_match_offline": shared_match,
        "decision_rule": {
            "nli_majority": 0.5,
            "hybrid_extractive_recall": 0.9,
            "note": (
                "NLI accepts entailment or contradiction only when that class probability "
                "is at least 0.5 on its best span. Entailment wins if both clear the bar. "
                "Hybrid keeps lexical judgments for missing numbers, lexical contradictions, "
                "and near-copy entailments without a dropped conditional. "
                "These cutoffs were not grid-searched on this benchmark."
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "sentence_transformers": _version("sentence-transformers"),
            "torch": _version("torch"),
        },
        "note": (
            "Synthetic benchmark only. Not an estimate of production precision. "
            "Latency excludes model load and the warmup pair."
        ),
        "providers": providers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload["results_path"] = str(args.output.resolve().relative_to(repo_root()))
    except ValueError:
        payload["results_path"] = str(args.output)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = render_comparison(payload)
    print("\n" + markdown)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
