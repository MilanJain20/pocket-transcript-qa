"""Locate repository data when running from a checkout."""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the checkout root, or the package root if data is not beside it."""
    override = os.environ.get("POCKET_QA_ROOT")
    if override:
        return Path(override).resolve()
    for candidate in (_PACKAGE_ROOT, *_PACKAGE_ROOT.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    return _PACKAGE_ROOT


def benchmark_dir() -> Path:
    return repo_root() / "data" / "benchmark"


def audio_dir() -> Path:
    return benchmark_dir() / "audio"


def resolve_audio_id(audio_id: str | None) -> Path | None:
    if audio_id is None:
        return None
    if audio_id not in {"silence", "speech"}:
        raise ValueError("audio_id must be 'silence', 'speech', or omitted")
    path = audio_dir() / f"{audio_id}.wav"
    if not path.is_file():
        raise FileNotFoundError(f"Bundled audio fixture is missing: {path}")
    return path
