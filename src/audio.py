"""Audio-only signals: WAV loading, speech activity (Silero VAD) and silence metrics.

Silence is reported neutrally: in a chess class, quiet stretches are often children solving puzzles.
Pitch, loudness and speaking rate are deliberately not used as quality signals.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .schemas import Metric

LONG_SILENCE_S = 30.0  # gap with no detected speech


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


def speech_regions(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Speech intervals in seconds, from the Silero VAD bundled with faster-whisper."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=300, speech_pad_ms=100)
    return [(t["start"] / sr, t["end"] / sr) for t in get_speech_timestamps(audio, opts, sampling_rate=sr)]


def audio_metrics(regions: list[tuple[float, float]], duration: float) -> list[Metric]:
    if not regions:
        return [Metric(name="speech_ratio", value=0.0, source="audio", note="no speech detected")]
    speech = sum(b - a for a, b in regions)
    # Silences inside the class only (ignore waiting before the first and after the last speech).
    gaps = [b0 - a1 for (_, a1), (b0, _) in zip(regions, regions[1:])]
    long_gaps = [g for g in gaps if g >= LONG_SILENCE_S]
    return [
        Metric(name="speech_ratio", value=round(speech / max(duration, 1), 2), source="audio",
               note="share of the recording with detected speech"),
        Metric(name="long_silences", value=len(long_gaps), unit="count", source="audio",
               note=f"gaps >= {LONG_SILENCE_S:.0f}s; may be productive (puzzle solving), not a fault by itself"),
        Metric(name="longest_silence_s", value=round(max(gaps) if gaps else 0.0), unit="s", source="audio"),
        Metric(name="first_speech_s", value=round(regions[0][0]), unit="s", source="audio",
               note="time before anyone speaks (waiting for children to join)"),
    ]
