"""End-to-end orchestration.

validate -> extract audio -> { transcribe (Groq) || separate speakers (local CPU) } -> label segments
-> condense chunks into events -> metrics -> judge -> Python scoring.

All intermediate media lives in a per-run temp directory that is deleted in a `finally` block, so
nothing is left behind on success or failure. Every stage is cached by file hash under data/cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from . import cache, llm, media, scoring, speakers, visual
from .cache import file_sha256
from .condense import condense
from .config import ConfigError, get_config, setup_logging
from .download import download
from .evaluator import evaluate
from .llm import LLMOutputError
from .metrics import compute_all
from .net import ProviderError
from .schemas import Assessment, ChunkDigest, DimensionScore, MediaInfo, Metric, ScoreResult, Transcript
from .speakers import SpeakerResult
from .visual import VisualResult
from .transcribe import LANGUAGE_NAMES, transcribe

log = logging.getLogger(__name__)
# on_stage(stage_name, fraction_complete_of_that_stage, plain-language detail for the UI)
StageFn = Callable[[str, float, str], None]

MUTED_MAX_DB = -50.0  # if even the loudest sample is below this, treat as muted
TEMP_PREFIX = "classeval_"
STALE_TEMP_S = 2 * 3600  # far longer than any real run, so an analysis in progress is never touched


def sweep_stale_temp() -> int:
    """Delete per-run temp dirs left behind by runs that were killed or crashed (a hard kill skips
    `finally`, which would otherwise leave a class's audio on disk). Returns how many were removed."""
    import time

    removed = 0
    for d in Path(tempfile.gettempdir()).glob(f"{TEMP_PREFIX}*"):
        try:
            if d.is_dir() and time.time() - d.stat().st_mtime > STALE_TEMP_S:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("removed %d stale temp dir(s) from interrupted runs", removed)
    return removed


class Signal(BaseModel):
    name: str
    status: str            # "used" | "partial" | "missing"
    detail: str = ""


class PipelineResult(BaseModel):
    file_hash: str
    media: MediaInfo
    audio_levels: dict[str, float]
    transcript: Transcript
    speakers: SpeakerResult | None = None
    visual: VisualResult | None = None
    digests: list[ChunkDigest] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    assessment: Assessment | None = None
    result: ScoreResult | None = None
    signals: list[Signal] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _speakers_cached(wav: Path, file_hash: str) -> SpeakerResult:
    params = hashlib.sha256(json.dumps(get_config()["speakers"], sort_keys=True).encode()).hexdigest()[:8]
    key = f"speakers_{params}"
    hit = cache.load(file_hash, key)
    if hit is not None:
        return SpeakerResult.model_validate(hit)
    res = speakers.separate(wav)
    cache.save(file_hash, key, res.model_dump())
    return res


def _visual_cached(video: Path, duration: float, file_hash: str) -> VisualResult:
    params = hashlib.sha256(json.dumps(get_config()["visual"], sort_keys=True).encode()).hexdigest()[:8]
    key = f"visual_{params}"
    hit = cache.load(file_hash, key)
    if hit is not None:
        return VisualResult.model_validate(hit)
    res = visual.analyse(video, duration)
    cache.save(file_hash, key, res.model_dump())
    return res


def _quality_factors(t: Transcript, spk: SpeakerResult | None, digests: list[ChunkDigest],
                     metrics: list[Metric]) -> list[tuple[float, str]]:
    """Reasons to trust the result less. Each (factor, reason) multiplies the confidence."""
    out = []
    if spk is None or spk.teacher_share is None:
        out.append((0.75, "Teacher and student voices could not be separated."))
    elif spk.confidence < 0.7:
        out.append((0.75 + 0.25 * spk.confidence,
                    f"Teacher/student voice separation is uncertain (confidence {spk.confidence:.2f})."))
    if t.language and t.language != "en":
        name = LANGUAGE_NAMES.get(t.language, t.language)
        out.append((0.9, f"Class language is {name}; transcription and event extraction are less accurate "
                         "than for English."))
    clarity = next((m.value for m in metrics if m.name == "transcription_confidence"), None)
    if isinstance(clarity, float) and clarity < 0.6:
        out.append((0.85, f"Audio/transcription clarity is low ({clarity:.2f})."))
    kept = sum(len(d.events) for d in digests)
    dropped = sum(d.dropped_events for d in digests)
    if kept + dropped and dropped / (kept + dropped) > 0.3:
        out.append((0.85, f"{dropped} of {kept + dropped} extracted events could not be verified in the transcript."))
    return out


def _signals(r: PipelineResult) -> list[Signal]:
    t, spk = r.transcript, r.speakers
    usable = len(t.usable_segments())
    lang = LANGUAGE_NAMES.get(t.language, t.language) or "unknown"
    sig = [
        Signal(name="Transcript", status="used" if usable else "missing",
               detail=f"{usable} usable segments, {lang}"),
        Signal(name="Speaker separation (teacher vs students)",
               status="missing" if not spk or spk.teacher_share is None else
               "used" if spk.confidence >= 0.7 else "partial",
               detail="" if not spk or spk.teacher_share is None else
               f"estimated from voices, confidence {spk.confidence:.2f}"),
        Signal(name="Event digests", status="used" if r.digests else "missing",
               detail=f"{sum(len(d.events) for d in r.digests)} verified/unverified events, "
                      f"{sum(d.dropped_events for d in r.digests)} dropped as unverifiable"),
        _visual_signal(r.visual),
        Signal(name="Reference anchors / reviewer style",
               status="used" if (Path(__file__).parent.parent / "config" / "anchors.json").exists() else "missing",
               detail="built from the labelled dataset by scripts/build_anchors.py"),
    ]
    return sig


def _visual_signal(vis: VisualResult | None) -> Signal:
    name = "Video frames (camera, 8x8 background)"
    if vis is None or not vis.frames:
        return Signal(name=name, status="missing", detail="no video track, or frames could not be read")
    cam = sum(f.camera_view for f in vis.frames)
    return Signal(name=name, status="used" if cam >= 3 else "partial",
                  detail=f"{len(vis.frames)} frames sampled every ~{vis.interval_s:.0f} s, analysed locally; "
                         f"tutor camera large in {cam}. Lighting quality not measured (fairness)")


def run(src_path: str | Path | None = None, on_stage: StageFn | None = None, *,
        url: str | None = None) -> PipelineResult:
    """Run on a local file (`src_path`) or a link (`url`). Exactly one must be given."""
    if (src_path is None) == (url is None):
        raise ValueError("pass exactly one of src_path or url")
    def notify(stage: str, frac: float, detail: str = "") -> None:
        if on_stage:
            on_stage(stage, frac, detail)
    sweep_stale_temp()
    llm.fallbacks_used.clear()
    work = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    try:
        if url is not None:
            notify("Downloading", 0.0, "Fetching the recording from the link")
            src = download(url, work, on_progress=lambda f: notify("Downloading", f, f"{f:.0%} downloaded"))
        else:
            src = Path(src_path)
        notify("Validating file", 0.0, "Checking the file can be read and is a supported length")
        info = media.validate(src)
        file_hash = file_sha256(src)
        log.info("validated input: %.0fs, %.1f MB, video=%s", info.duration_s, info.size_mb, info.has_video)

        # Video frames are analysed locally in the background while the audio path runs.
        pool = ThreadPoolExecutor(max_workers=2)
        vis_future = (pool.submit(_visual_cached, src, info.duration_s, file_hash)
                      if info.has_video and get_config()["visual"]["enabled"] else None)

        notify("Extracting audio", 0.0, "Pulling the audio track out of the video")
        wav = media.extract_audio(src, work)
        levels = media.audio_levels(wav)
        warnings: list[str] = []
        spk: SpeakerResult | None = None

        if levels["max_volume"] < MUTED_MAX_DB:
            # Whisper invents text on silence, so don't transcribe a muted track at all.
            warnings.append(f"Audio appears muted (peak {levels['max_volume']:.0f} dBFS); no transcription attempted.")
            transcript = Transcript(duration_s=info.duration_s, provider=get_config()["provider"], model="none",
                                    warnings=["No usable speech was transcribed."])
        else:
            chunks = media.chunk_audio(wav, info.duration_s, work)
            notify("Transcribing + separating speakers", 0.0, f"Transcribing part 1 of {len(chunks)}")
            # Network-bound transcription and CPU-bound speaker separation run side by side.
            spk_future = pool.submit(_speakers_cached, wav, file_hash)
            transcript = transcribe(chunks, info.duration_s, file_hash, on_progress=lambda i, n: notify(
                "Transcribing + separating speakers", 0.9 * i / max(n, 1),
                f"Transcribing part {min(i + 1, n)} of {n}" if i < n else "Telling teacher and student voices apart"))
            try:
                spk = spk_future.result()
            except Exception as e:  # separation is optional: degrade, don't fail the run
                log.warning("speaker separation failed: %s", type(e).__name__)
                warnings.append("Speaker separation failed; participation metrics are unavailable.")
            if spk:
                speakers.label_segments(transcript.segments, spk)
                warnings += spk.warnings

        digests: list[ChunkDigest] = []
        if transcript.usable_segments():
            notify("Summarising the class", 0.0, "Reading the class section by section")
            digests, _names = condense(transcript, file_hash,
                                       on_progress=lambda i, n: notify("Summarising the class", i / max(n, 1),
                                                                       f"{i} of {n} sections summarised"))

        vis: VisualResult | None = None
        if vis_future is not None:
            try:
                vis = vis_future.result()
                warnings += vis.warnings
            except Exception as e:  # visual checks are optional: degrade, don't fail the run
                log.warning("visual analysis failed: %s", type(e).__name__)
                warnings.append("Video frames could not be analysed; camera and background were not assessed.")

        notify("Computing metrics", 0.0, "Counting questions, answers, praise and talk time")
        metrics = compute_all(transcript, spk, digests) + (vis.metrics if vis else [])

        meta = {"duration": f"{info.duration_s / 60:.0f} min",
                "language": LANGUAGE_NAMES.get(transcript.language, transcript.language) or "unknown",
                "subject": "chess, for children, online (Zoom)"}
        if digests:
            notify("Assessing against the rubric", 0.0, "Scoring 6 rubric areas and checking every piece of evidence")
            assessment, _ = evaluate(digests, metrics, meta, file_hash,
                                     visual=vis.observations if vis and vis.frames else None)
        else:
            assessment = Assessment(
                summary="The recording has no usable speech, so teaching could not be assessed.",
                dimensions=[DimensionScore(dimension=d["id"]) for d in scoring.load_rubric()["dimensions"]],
                limitations=["No usable speech: muted, silent or unintelligible audio."], confidence=0.0)

        result = scoring.score(assessment, quality=_quality_factors(transcript, spk, digests, metrics))
        notify("Done", 1.0)
        out = PipelineResult(file_hash=file_hash, media=info, audio_levels=levels, transcript=transcript,
                             speakers=spk, visual=vis, digests=digests, metrics=metrics, assessment=assessment,
                             result=result, warnings=warnings + transcript.warnings
                             + (["Part of this analysis used a backup model, so wording may differ slightly."]
                                if llm.fallbacks_used else []))
        out.signals = _signals(out)
        return out
    finally:
        if "pool" in locals():
            pool.shutdown(wait=True, cancel_futures=True)  # frame grabs read the source file, so finish before deleting it
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            log.warning("temp dir could not be fully removed: %s", work)


def summarize(result: PipelineResult) -> dict:
    """Text-free summary (no transcript, quotes or names), safe to print or log."""
    t = result.transcript
    usable = t.usable_segments()
    speech_s = sum(s.end - s.start for s in usable)
    r = result.result
    return {
        "file_hash": result.file_hash[:12],
        "duration_min": round(result.media.duration_s / 60, 1),
        "language": LANGUAGE_NAMES.get(t.language, t.language),
        "model": t.model,
        "usable_segments": len(usable),
        "speech_coverage": round(speech_s / max(result.media.duration_s, 1), 3),
        "teacher_share": result.speakers.teacher_share if result.speakers else None,
        "speaker_confidence": result.speakers.confidence if result.speakers else None,
        "events": sum(len(d.events) for d in result.digests),
        "events_dropped": sum(d.dropped_events for d in result.digests),
        "label": r.label if r else None,
        "score": r.score if r else None,
        "confidence": r.confidence if r else None,
        "low_confidence": r.low_confidence if r else None,
        "dimension_scores": {d.dimension: d.score for d in result.assessment.dimensions} if result.assessment else {},
        "warnings": result.warnings,
    }


if __name__ == "__main__":
    # python -m src.pipeline path/to/recording.mp4   |   python -m src.pipeline https://...
    setup_logging()
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.pipeline <recording file or link>")
    arg = sys.argv[1]
    progress = lambda s, f, d="": print(f"[{s}] {f:.0%} {d}", file=sys.stderr)  # noqa: E731
    is_link = arg.startswith(("http://", "https://"))
    try:
        res = run(url=arg, on_stage=progress) if is_link else run(arg, on_stage=progress)
    except (media.MediaError, ConfigError, ProviderError, LLMOutputError) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(summarize(res), indent=2))
