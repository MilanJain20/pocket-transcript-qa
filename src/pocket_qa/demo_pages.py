"""Pages used by the local demo: title cards and a README rendering.

The results table is copied from README.md at request time. This module does
not recompute or rewrite benchmark figures.
"""

from __future__ import annotations

import html
from pathlib import Path

from pocket_qa.paths import repo_root

_STYLE = """
:root { --paper:#e6e0d2; --panel:#f6f3ea; --ink:#1c1914; --muted:#5e584c; --line:#cbbfa8; --block:#9d2c1c; }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font-family: "Newsreader", Georgia, serif; }
header, main { width: min(1080px, calc(100% - 32px)); margin: 0 auto; }
.kicker { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
  letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font-size: clamp(36px, 5vw, 56px); line-height: .95; margin: 8px 0 12px; letter-spacing: -.03em; }
.banner, .note { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; color: var(--muted); }
.banner { border: 1px solid var(--line); background: var(--panel); padding: 10px 12px; }
.flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 20px; }
.flow span { border: 1px solid var(--ink); background: var(--panel); padding: 8px 10px;
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; }
.flow i { font-style: normal; color: var(--muted); }
table { width: 100%; border-collapse: collapse; background: var(--panel); font-size: 16px; }
th, td { border-top: 1px solid var(--line); padding: 8px 10px; text-align: left; }
th { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; letter-spacing: .04em;
  text-transform: uppercase; color: var(--muted); }
td:nth-child(n+2), th:nth-child(n+2) { text-align: right; font-variant-numeric: tabular-nums; }
.card { min-height: 100vh; display: flex; align-items: center; }
.card main { width: min(900px, calc(100% - 48px)); }
.card h1 { font-size: clamp(42px, 6vw, 68px); }
"""

_FONT = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=Newsreader:opsz,wght@6..72,480;6..72,600&display=swap" rel="stylesheet">'
)


def title_card(step: str, play: bool = False) -> str:
    cards = {
        "open": (
            "Pocket Transcript QA",
            "An independent eval harness for voice transcripts and summaries. "
            "Not affiliated with Pocket. The Pocket client in this demo is a mock.",
        ),
        "gate": (
            "Release gate",
            "A clean synthetic note should PASS. A faulty note should BLOCK, "
            "with the unsupported claim, its evidence span, a fabricated segment, and a speaker error.",
        ),
        "readme": (
            "Pipeline and benchmark",
            "LangGraph runs ingest, transcript checks, summary checks, speaker checks, the gate, and the report. "
            "The table below is the synthetic benchmark output already written into the README.",
        ),
        "end": (
            "Synthetic benchmark. Offline. No API key.",
            "Independent project. Not affiliated with or endorsed by Pocket. "
            "The Pocket client in this demo is a mock.",
        ),
    }
    title, body = cards.get(step, cards["open"])
    script = ""
    if play and step == "open":
        script = "<script>setTimeout(() => { location.href='/?play=1'; }, 7000);</script>"
    return _page(
        f"<main><p class='kicker'>Milan Jain · independent project</p>"
        f"<h1>{html.escape(title)}</h1><p class='banner'>{html.escape(body)}</p></main>{script}",
        extra_class="card",
    )


def readme_page(play: bool = False) -> str:
    text = (repo_root() / "README.md").read_text(encoding="utf-8")
    results = _results_block(text)
    return _page(
        "<header><p class='kicker'>README · Pocket Transcript QA</p>"
        "<h1>Architecture and results</h1>"
        "<p class='banner'>Independent project. Not affiliated with or endorsed by Pocket. "
        "The Pocket client in this demo is a mock. Figures are read from README.md, not recomputed.</p>"
        "</header><main>"
        "<h2>LangGraph pipeline</h2>"
        "<div class='flow'>"
        "<span>ingest</span><i>→</i><span>transcript checks</span><i>→</i>"
        "<span>summary checks</span><i>→</i><span>speaker checks</span><i>→</i>"
        "<span>gate</span><i>→</i><span>report</span>"
        "</div>"
        "<p class='note'>Same six nodes as the diagram in README.md. "
        "Offline runs do not call a model. A selected provider is used only while grounding claims.</p>"
        "<h2 id='benchmark'>Synthetic benchmark</h2>"
        f"{results}"
        "</main>"
        + (
            "<script>setTimeout(() => { document.getElementById('benchmark').scrollIntoView({block:'start'}); }, 6000);"
            "setTimeout(() => { location.href='/demo/card?step=end'; }, 18000);</script>"
            if play
            else ""
        )
    )


def _results_block(text: str) -> str:
    start = "<!-- BENCHMARK_RESULTS_START -->"
    end = "<!-- BENCHMARK_RESULTS_END -->"
    if start not in text or end not in text:
        return "<p>README is missing the benchmark block.</p>"
    block = text.split(start, 1)[1].split(end, 1)[0].strip()
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    prose = [html.escape(line) for line in lines if not line.startswith("|")]
    table_lines = [line for line in lines if line.startswith("|")]
    table = _table(table_lines)
    paragraphs = "".join(f"<p class='note'>{line}</p>" for line in prose)
    return table + paragraphs


def _table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and set(cells[0]) <= {"-", ":", " "}:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in rows[0])
    body = []
    for row in rows[1:]:
        body.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _page(body: str, extra_class: str = "") -> str:
    klass = f" class='{extra_class}'" if extra_class else ""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Pocket Transcript QA</title>{_FONT}<style>{_STYLE}</style></head>"
        f"<body{klass}>{body}</body></html>"
    )


def readme_path() -> Path:
    return repo_root() / "README.md"
