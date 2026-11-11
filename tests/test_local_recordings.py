"""Local recording folder and CLI. Inputs are synthetic and written under tmp."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pocket_qa.bench import latency_summary, render_comparison
from pocket_qa.cli import app
from pocket_qa.local_recordings import LOCAL_NOTICE, discover_recordings
from pocket_qa.paths import repo_root

runner = CliRunner()


def test_gitignore_covers_local_recordings() -> None:
    text = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "recordings/" in text
    assert "reports/local/" in text


def test_folder_of_text_and_pocket_json(tmp_path: Path) -> None:
    folder = tmp_path / "recordings"
    note = folder / "standup"
    note.mkdir(parents=True)
    (note / "transcript.txt").write_text(
        (repo_root() / "examples" / "standup.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (note / "summary.md").write_text(
        (repo_root() / "examples" / "standup.summary.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    export = {
        "id": "rec_export",
        "title": "Exported standup",
        "transcript": {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "speakerName": "Alice",
                    "text": "We should ship the release on Friday.",
                    "start": 0,
                    "end": 3.4,
                }
            ]
        },
        "summarizations": {
            "sum_1": {"v2": {"summary": {"markdown": "The team should ship the release on Friday."}}}
        },
    }
    (folder / "export.json").write_text(json.dumps(export), encoding="utf-8")
    found = discover_recordings(folder, provider="offline")
    ids = {item[0] for item in found}
    assert ids == {"standup", "rec_export"}
    assert all(request.provider == "offline" for _id, request, _source in found)


def test_recordings_cli_writes_a_local_report(tmp_path: Path) -> None:
    out = tmp_path / "standup-report.json"
    result = runner.invoke(
        app,
        [
            "recordings",
            "--transcript",
            "examples/standup.txt",
            "--summary",
            "examples/faulty.summary.txt",
            "--output",
            str(out),
            "--provider",
            "offline",
        ],
    )
    assert result.exit_code == 1
    assert "stay on this machine" in result.output
    assert "Do not commit" in result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["notice"] == LOCAL_NOTICE
    assert payload["report"]["gate"]["decision"] == "BLOCK"
    assert payload["report"]["provider"] == "offline"


def test_recordings_cli_folder_pass(tmp_path: Path) -> None:
    folder = tmp_path / "recordings"
    note = folder / "standup"
    note.mkdir(parents=True)
    (note / "transcript.txt").write_text(
        (repo_root() / "examples" / "standup.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (note / "summary.txt").write_text(
        (repo_root() / "examples" / "standup.summary.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out = tmp_path / "reports"
    result = runner.invoke(
        app,
        ["recordings", "--input", str(folder), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["recordings"][0]["decision"] == "PASS"
    assert "Do not commit" in index["notice"]


def test_recordings_cli_rejects_an_empty_folder(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    result = runner.invoke(
        app,
        ["recordings", "--input", str(folder), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 2
    assert "No recordings found" in result.output


def test_unknown_provider_is_rejected() -> None:
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--transcript",
            "examples/standup.txt",
            "--summary",
            "examples/standup.summary.txt",
            "--provider",
            "nope",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown provider" in result.output


def test_published_comparison_matches_the_readme() -> None:
    """The README comparison quotes results/provider_comparison.json. Do not retype it."""
    payload = json.loads((repo_root() / "results" / "provider_comparison.json").read_text(encoding="utf-8"))
    readme = (repo_root() / "README.md").read_text(encoding="utf-8")
    offline = payload["providers"]["offline"]["metrics"]["summary_hallucination"]
    nli = payload["providers"]["local-nli"]["metrics"]["summary_hallucination"]
    hybrid = payload["providers"]["hybrid"]["metrics"]["summary_hallucination"]
    assert offline["tp"] == 19 and offline["fp"] == 2 and offline["fn"] == 2
    assert nli["tp"] == 19 and nli["fp"] == 7 and nli["fn"] == 2
    assert hybrid["tp"] == 19 and hybrid["fp"] == 5 and hybrid["fn"] == 2
    assert f"{nli['f1']}" in readme
    assert f"{hybrid['f1']}" in readme
    assert payload["generated_at"] in readme
    assert payload["model_load_seconds"] and f"{payload['model_load_seconds']}" in readme


def test_comparison_renderer_uses_the_given_counts() -> None:
    metric = {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "support": 1,
    }
    report = {
        "metrics": {
            "summary_hallucination": metric,
            "transcript_fabrication": {"tp": 0, "fp": 0, "fn": 0},
            "speaker_inconsistency": {"tp": 0, "fp": 0, "fn": 0},
        },
        "claim_label_accuracy": 0.5,
        "claim_label_total": 2,
        "latency_seconds": latency_summary([0.125, 0.375]),
        "claim_errors": [{"id": "sum-para-01", "gold": "entailed", "pred": "unsupported", "claim": "A paraphrase."}],
    }
    text = render_comparison(
        {
            "generated_at": "2026-11-11T00:00:00Z",
            "model_id": "cross-encoder/nli-deberta-v3-xsmall",
            "model_load_seconds": 1.25,
            "results_path": "results/provider_comparison.json",
            "shared_checks_match_offline": True,
            "versions": {"python": "3.12.3", "sentence_transformers": "3.0.0", "torch": "2.0.0"},
            "providers": {"offline": report},
        }
    )
    assert "| offline | 0.500 | 1.000 | 0.667 | 1 | 1 | 0 | 0.500 on 2 | 0.250000 | 0.250000 | 0.375000 |" in text
    assert "sum-para-01" in text
    assert "match the offline run" in text
