"""Benchmark integrity, CLI, and MCP registration."""

import importlib.util
import json

from typer.testing import CliRunner

from pocket_qa.bench import load_cases, score_cases
from pocket_qa.cli import app
from pocket_qa.paths import benchmark_dir, repo_root


def _generator():
    path = repo_root() / "scripts" / "generate_benchmark.py"
    spec = importlib.util.spec_from_file_location("generate_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_benchmark_matches_generator() -> None:
    module = _generator()
    generated = json.loads(json.dumps(module.generate_cases()))
    on_disk = [json.loads(line) for line in (benchmark_dir() / "cases.jsonl").read_text().splitlines() if line]
    assert generated == on_disk
    assert (benchmark_dir() / "audio" / "silence.wav").is_file()
    assert (benchmark_dir() / "audio" / "speech.wav").is_file()


def test_gold_block_matches_injected_faults() -> None:
    for case in load_cases():
        gold = case["gold"]
        assert gold["block"] == (
            gold["summary_hallucination"] or gold["transcript_fabrication"] or gold["speaker_inconsistency"]
        )
        ids = {segment["id"] for segment in case["transcript"]}
        assert set(gold["fabricated_segment_ids"]) <= ids


def test_recording_level_misses_are_the_documented_limitations() -> None:
    """Locks the disagreements named in the README to this run, not a hand-typed score."""
    report = score_cases()
    assert {row["id"] for row in report["misses"]} == {
        "sum-cond-01",
        "sum-cond-02",
        "sum-para-01",
        "sum-para-02",
        "tr-subtle-01",
        "tr-subtle-02",
        "tr-subtle-03",
        "spk-passive-01",
        "spk-passive-02",
    }


def test_metrics_are_computed_not_hardcoded() -> None:
    report = score_cases()
    for metric in report["metrics"].values():
        assert metric["support"] == metric["tp"] + metric["fn"]
        assert 0.0 <= metric["precision"] <= 1.0
        assert 0.0 <= metric["recall"] <= 1.0
        assert 0.0 <= metric["f1"] <= 1.0
    assert report["synthetic"] is True
    assert report["provider"] == "offline"
    assert report["case_count"] == len(load_cases())


def test_cli_examples() -> None:
    runner = CliRunner()
    passed = runner.invoke(
        app,
        [
            "evaluate",
            "--transcript",
            "examples/standup.txt",
            "--summary",
            "examples/standup.summary.txt",
        ],
    )
    assert passed.exit_code == 0
    assert '"decision": "PASS"' in passed.stdout
    blocked = runner.invoke(
        app,
        [
            "evaluate",
            "--transcript",
            "examples/standup.txt",
            "--summary",
            "examples/faulty.summary.txt",
        ],
    )
    assert blocked.exit_code == 1
    assert '"decision": "BLOCK"' in blocked.stdout
    pulled = runner.invoke(app, ["pull", "rec_demo_standup"])
    assert pulled.exit_code == 0
    assert "MockPocketClient" in pulled.output
    assert '"pocket_client_mock": true' in pulled.output


def test_mcp_tools_are_registered() -> None:
    from pocket_qa.mcp_server import evaluate_transcript, tool_names

    names = set(tool_names())
    assert {
        "evaluate_transcript",
        "check_summary_claims",
        "check_transcript_fabrication",
        "check_speaker_consistency",
        "release_gate",
    } <= names
    report = json.loads(
        evaluate_transcript(
            "Alice | 0 | 3.4 | We should ship the release on Friday.",
            "The team should ship the release on Friday.",
        )
    )
    assert report["gate"]["decision"] == "PASS"
