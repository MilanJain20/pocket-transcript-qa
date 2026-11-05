"""Waveform energy for the silent-audio fabrication check.

Uses the stdlib `wave` module so the default install does not need librosa
or numpy. Fixtures are 16 kHz mono 16-bit PCM.
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path

# RMS below this, on a full-scale float in [-1, 1], is treated as near-silence.
# The synthetic silence fixture sits near 0.001; the speech-like fixture sits
# well above 0.05. See data/benchmark/GENERATION.md.
SILENCE_RMS = 0.01


def segment_rms(path: str | Path, start: float, end: float) -> float | None:
    """Root-mean-square energy of [start, end] seconds. None if the range is empty."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    with wave.open(str(file_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_rate = handle.getframerate()
        frame_count = handle.getnframes()
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM, got sample width {sample_width}")
        if channels < 1 or frame_rate <= 0:
            raise ValueError("Unreadable wav header")
        start_frame = max(0, int(start * frame_rate))
        end_frame = min(frame_count, int(end * frame_rate))
        if end_frame <= start_frame:
            return None
        handle.setpos(start_frame)
        raw = handle.readframes(end_frame - start_frame)
    samples = array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array("h", samples[::channels])
    if not samples:
        return None
    square = 0.0
    for sample in samples:
        value = sample / 32768.0
        square += value * value
    return (square / len(samples)) ** 0.5
