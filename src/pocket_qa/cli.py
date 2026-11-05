"""Command line for the eval harness."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pocket_qa import __version__
from pocket_qa.evaluate import evaluate
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
    provider: str = typer.Option("offline", "--provider"),
    recording_id: str | None = typer.Option(None, "--recording-id"),
) -> None:
    """Evaluate a transcript file and a summary. Prints the JSON report."""
    summary_body = summary_text
    if summary is not None:
        summary_body = summary.read_text(encoding="utf-8")
    request = parse_transcript(transcript.read_text(encoding="utf-8"), summary=summary_body)
    request.provider = provider  # type: ignore[assignment]
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
