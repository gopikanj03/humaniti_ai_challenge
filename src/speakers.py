"""Lightweight teacher/student separation from voice embeddings (runs locally, no API).

1. Silero VAD (bundled with faster-whisper) finds speech regions.
2. Each region is cut into short overlapping windows; a WeSpeaker ResNet34 ONNX model turns each
   window into a voice embedding.
3. Agglomerative clustering on cosine distance groups windows by voice. The cluster with the most
   speech time is taken to be the teacher (true in 10/10 labelled classes, using the Zoom VTTs).
4. Transcript segments are labelled teacher / student / mixed by how much of them overlaps
   teacher windows.

This is an estimate, not ground truth: children's voices are outside the embedding model's
training data, and Zoom mixes everyone into one track. The returned confidence is used to
soften participation claims downstream.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .audio import as_float, read_wav, speech_regions  # noqa: F401 (re-exported)
from .config import get_config
from .models import ensure_model
from .schemas import Segment

log = logging.getLogger(__name__)


class SpeakerTurn(BaseModel):
    start: float
    end: float
    role: str               # "teacher" | "student"


class SpeakerResult(BaseModel):
    turns: list[SpeakerTurn] = Field(default_factory=list)
    speech_regions: list[tuple[float, float]] = Field(default_factory=list)
    n_voice_clusters: int = 0
    teacher_share: float | None = None   # of detected speech time
    confidence: float = 0.0              # 0..1, how cleanly the teacher voice separates
    warnings: list[str] = Field(default_factory=list)


def _windows(regions: list[tuple[float, float]], win: float, step: float, min_len: float) -> list[tuple[float, float]]:
    out = []
    for a, b in regions:
        if b - a < min_len:
            continue
        if b - a <= win:
            out.append((a, b))
            continue
        t = a
        while t + win <= b:
            out.append((t, t + win))
            t += step
        if b - (t - step + win) > step / 2:  # cover the tail
            out.append((b - win, b))
    return out


def _embed(audio: np.ndarray, sr: int, windows: list[tuple[float, float]]) -> np.ndarray:
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ensure_model("speaker_embedding")), num_threads=4)
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    embs = np.zeros((len(windows), extractor.dim), dtype=np.float32)
    for i, (a, b) in enumerate(windows):
        stream = extractor.create_stream()
        stream.accept_waveform(sr, as_float(audio[int(a * sr):int(b * sr)]))
        stream.input_finished()
        embs[i] = extractor.compute(stream)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def cluster(embs: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    if len(embs) < 2:
        return np.zeros(len(embs), dtype=int)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                   linkage="average").fit_predict(embs)


def separate(wav_path: str | Path) -> SpeakerResult:
    scfg = get_config()["speakers"]
    audio, sr = read_wav(wav_path)
    regions = speech_regions(audio, sr)
    windows = _windows(regions, scfg["window_s"], scfg["step_s"], scfg["min_region_s"])
    if len(windows) < 10:
        return SpeakerResult(speech_regions=regions, warnings=["Too little speech to separate speakers."])

    embs = _embed(audio, sr, windows)
    thr = scfg["cluster_threshold"]
    result = build_result(windows, cluster(embs, thr), embs, regions)

    # Confidence = stability: re-cluster at nearby thresholds and see how far the teacher share moves.
    # On the 12 dataset recordings, stability correlated with accuracy against the Zoom VTTs (r = -0.48
    # for spread vs balanced accuracy) while cluster-size/tightness heuristics did not (r = -0.25, wrong
    # sign). Capped at CONFIDENCE_CAP because even stable runs averaged only ~0.7 balanced accuracy.
    shares = [result.teacher_share] + [build_result(windows, cluster(embs, t), embs, regions).teacher_share
                                       for t in (thr - 0.05, thr + 0.05)]
    spread = max(shares) - min(shares)
    result.confidence = round(CONFIDENCE_CAP * float(np.clip(1 - spread / 0.4, 0, 1)), 2)
    if spread > 0.2:
        result.warnings.append("Teacher/student voice separation is unstable for this recording; participation "
                               "figures are rough estimates.")
    log.info("speakers: %d clusters, teacher share %.2f, stability spread %.2f, confidence %.2f",
             result.n_voice_clusters, result.teacher_share, spread, result.confidence)
    return result


CONFIDENCE_CAP = 0.75


def build_result(windows: list[tuple[float, float]], labels: np.ndarray, embs: np.ndarray,
                 regions: list[tuple[float, float]]) -> SpeakerResult:
    """Teacher = the cluster with the most speech time; everyone else = students. Confidence is set by
    `separate` (it needs re-clustering), so it is 0 here."""
    durations: dict[int, float] = {}
    for (a, b), lab in zip(windows, labels):
        durations[int(lab)] = durations.get(int(lab), 0.0) + (b - a)
    ranked = sorted(durations, key=durations.get, reverse=True)
    teacher = ranked[0]
    total = sum(durations.values())
    teacher_share = durations[teacher] / total
    second = durations[ranked[1]] / total if len(ranked) > 1 else 0.0

    turns = _merge_turns([(a, b, "teacher" if lab == teacher else "student") for (a, b), lab in zip(windows, labels)])
    warnings = []
    if teacher_share - second < 0.1:
        warnings.append("The two most active voices have similar talk time; teacher identification is uncertain.")
    return SpeakerResult(turns=turns, speech_regions=regions, n_voice_clusters=len(ranked),
                         teacher_share=round(teacher_share, 3), confidence=0.0, warnings=warnings)


def _merge_turns(items: list[tuple[float, float, str]]) -> list[SpeakerTurn]:
    """Collapse overlapping windows into turns; where windows of different roles overlap, split at the midpoint."""
    turns: list[SpeakerTurn] = []
    for a, b, role in sorted(items):
        if turns and a <= turns[-1].end + 0.3:
            if turns[-1].role == role:
                turns[-1].end = max(turns[-1].end, b)
                continue
            mid = (a + turns[-1].end) / 2 if a < turns[-1].end else a
            turns[-1].end = min(turns[-1].end, mid)
            a = mid
        turns.append(SpeakerTurn(start=a, end=b, role=role))
    return turns


def teacher_fraction(turns: list[SpeakerTurn], start: float, end: float) -> float | None:
    """Fraction of [start, end] covered by teacher turns, out of the part covered by any turn."""
    teacher = other = 0.0
    for t in turns:
        if t.end <= start:
            continue
        if t.start >= end:
            break
        overlap = min(end, t.end) - max(start, t.start)
        if t.role == "teacher":
            teacher += overlap
        else:
            other += overlap
    return None if teacher + other == 0 else teacher / (teacher + other)


def label_segments(segments: list[Segment], result: SpeakerResult) -> None:
    """Set segment.speaker in place: teacher / student / mixed, or None when no voice was detected."""
    scfg = get_config()["speakers"]
    for s in segments:
        frac = teacher_fraction(result.turns, s.start, s.end)
        s.teacher_fraction = None if frac is None else round(frac, 2)
        if frac is None:
            s.speaker = None
        elif frac >= scfg["teacher_min_fraction"]:
            s.speaker = "teacher"
        elif frac <= scfg["student_max_fraction"]:
            s.speaker = "student"
        else:
            s.speaker = "mixed"
