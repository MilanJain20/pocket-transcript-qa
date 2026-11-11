"""Command line for the eval harness."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from pocket_qa import __version__
from pocket_qa.evaluate import evaluate
from pocket_qa.models import PROVIDER_NAMES, ProviderName
from pocket_qa.paths import repo_root
from pocket_qa.pocket.mock import MockPocketClient
from pocket_qa.pocket.normalize import request_from_recording
from pocket_qa.transcript_format import parse_transcript

app = typer.Typer(
    add_completion=False,
    help="Pocket Transcript QA: check a transcript and summary before it ships.",
    no_args_is_help=True,
)


def _dump(report: object) -> None:
    typer.echo(json.dumps(report.model_dump(), indent=2))  # type: ignore[attr-defined]


def _resolve_provider(provider: str | None) -> ProviderName:
    selected = provider if provider else os.environ.get("POCKET_QA_PROVIDER", "offline")
    if selected not in PROVIDER_NAMES:
        names = ", ".join(PROVIDER_NAMES)
        raise typer.BadParameter(f"Unknown provider {selected!r}. Expected one of: {names}.")
    return selected  # type: ignore[return-value]


def _safe_stem(recording_id: str, used: set[str]) -> str:
    stem = "".join(char if char.isalnum() or char in "-._" else "_" for char in recording_id).strip("._")
    stem = stem or "recording"
    name = stem
    suffix = 2
    while name in used:
        name = f"{stem}-{suffix}"
        suffix += 1
    used.add(name)
    return name


def _warn_if_repo_output(path: Path) -> None:
    resolved = path.resolve()
    root = repo_root().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return
    parts = relative.parts
    allowed = (parts[:1] == ("recordings",)) or (parts[:2] == ("reports", "local"))
    if not allowed:
        typer.echo(
            "Warning: the report path is inside this repository and can quote transcript text. "
            "Do not commit it. Prefer recordings/ or reports/local/, which are gitignored.",
            err=True,
        )


@app.callback()
def main_callback() -> None:
    """Independent eval harness. Not affiliated with Pocket."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("evaluate")
def evaluate_cmd(
    transcript: Path = typer.Option(..., "--transcript", "-t", exists=True, readable=True),
    summary: Path | None = typer.Option(None, "--summary", "-s", exists=True, readable=True),
    summary_text: str = typer.Option("", "--summary-text"),
    audio: Path | None = typer.Option(None, "--audio", exists=True, readable=True),
    audio_id: str | None = typer.Option(None, "--audio-id"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Claim-support provider. Default is POCKET_QA_PROVIDER, or offline when that is unset.",
    ),
    recording_id: str | None = typer.Option(None, "--recording-id"),
) -> None:
    """Evaluate a transcript file and a summary. Prints the JSON report."""
    summary_body = summary_text
    if summary is not None:
        summary_body = summary.read_text(encoding="utf-8")
    request = parse_transcript(transcript.read_text(encoding="utf-8"), summary=summary_body)
    request.provider = _resolve_provider(provider)
    request.recording_id = recording_id
    if audio is not None:
        request.audio_path = str(audio)
    if audio_id is not None:
        request.audio_id = audio_id  # type: ignore[assignment]
    try:
        report = evaluate(request)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    _dump(report)
    if report.gate.decision == "BLOCK":
        raise typer.Exit(code=1)


@app.command()
def bench(
    write_readme: bool = typer.Option(False, "--write-readme"),
    check_readme: bool = typer.Option(False, "--check-readme"),
) -> None:
    """Run the synthetic benchmark and print precision, recall, and F1."""
    from pocket_qa.bench import run_benchmark

    report = run_benchmark(write_readme=write_readme, check_readme=check_readme)
    typer.echo(report["markdown"])


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8741, "--port"),
) -> None:
    """Start the local API and demo page."""
    import uvicorn

    typer.echo(f"Pocket Transcript QA demo at http://{host}:{port}")
    uvicorn.run("pocket_qa.api:app", host=host, port=port, reload=False)


@app.command()
def mcp() -> None:
    """Start the eval MCP server on stdio. This is not Pocket's official MCP server."""
    from pocket_qa.mcp_server import mcp as mcp_app

    mcp_app.run()


@app.command()
def pull(recording_id: str, live: bool = typer.Option(False, "--live")) -> None:
    """Fetch a recording and evaluate it.

    The default client is MockPocketClient. --live calls the public Pocket API
    and requires POCKET_API_KEY. This repository does not contain a key.
    """
    if live:
        from pocket_qa.pocket.http import HttpPocketClient

        client = HttpPocketClient()
        typer.echo("Using the live Pocket Public API client.", err=True)
    else:
        client = MockPocketClient()
        typer.echo(
            "Using MockPocketClient. No request was sent to Pocket. "
            "Pass --live with POCKET_API_KEY to call the public API.",
            err=True,
        )
    try:
        recording = client.get_recording(recording_id)
        request = request_from_recording(recording)
        report = evaluate(request)
    except Exception as exc:  # noqa: BLE001
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    payload = report.model_dump()
    payload["pocket_client_mock"] = not live
    typer.echo(json.dumps(payload, indent=2))
    if report.gate.decision == "BLOCK":
        raise typer.Exit(code=1)


@app.command("recordings")
def recordings_cmd(
    transcript: Path | None = typer.Option(None, "--transcript", "-t", exists=True, readable=True),
    summary: Path | None = typer.Option(None, "--summary", "-s", exists=True, readable=True),
    audio: Path | None = typer.Option(None, "--audio", exists=True, readable=True),
    input_dir: Path | None = typer.Option(None, "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Claim-support provider. Default is POCKET_QA_PROVIDER, or offline when that is unset.",
    ),
) -> None:
    """Evaluate local transcript and summary files and write a JSON report.

    Files stay on this machine. Nothing is uploaded and nothing is sent to Pocket.
    Put real exports in recordings/ and write reports to reports/local/. Both paths
    are gitignored. Do not commit them.

    One recording: --transcript and --summary (summary may be omitted when the
    transcript file is a Pocket JSON export that already contains one).

    A folder: --input. Each subdirectory may contain transcript.txt and summary.txt,
    plus optional audio.wav. JSON files in the folder are read as exports.
    """
    from pocket_qa.local_recordings import (
        LOCAL_NOTICE,
        discover_recordings,
        load_pair,
        write_index,
        write_report,
    )

    selected = _resolve_provider(provider)
    typer.echo(LOCAL_NOTICE, err=True)
    _warn_if_repo_output(output)
    try:
        if input_dir is not None and transcript is not None:
            raise ValueError("Pass either --input or --transcript, not both.")
        if input_dir is None and transcript is None:
            raise ValueError("Pass --transcript and --summary, or --input with a folder of recordings.")
        from pocket_qa.models import EvaluateRequest

        jobs: list[tuple[str, EvaluateRequest, str]] = []
        if input_dir is not None:
            jobs.extend(discover_recordings(input_dir, provider=selected))
        else:
            assert transcript is not None
            request = load_pair(transcript, summary, audio, provider=selected)
            recording_id = request.recording_id or transcript.stem
            request.recording_id = recording_id
            jobs.append((recording_id, request, str(transcript)))
        written: list[dict[str, str]] = []
        blocked = False
        many = len(jobs) > 1 or input_dir is not None or output.suffix.lower() != ".json"
        if many:
            output.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        for recording_id, request, source in jobs:
            report = evaluate(request)
            if many:
                destination = output / f"{_safe_stem(recording_id, used_names)}.json"
            else:
                destination = output
            write_report(destination, report.model_dump(), source=source)
            written.append(
                {
                    "id": recording_id,
                    "decision": report.gate.decision,
                    "path": str(destination),
                }
            )
            blocked = blocked or report.gate.decision == "BLOCK"
            typer.echo(f"{recording_id}: {report.gate.decision} -> {destination}")
        if many:
            write_index(output / "index.json", written)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if blocked:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
