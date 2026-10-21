"""End-to-end orchestration. Stage 1: validate -> extract audio -> chunk -> transcribe.

All intermediate media lives in a per-run temp directory that is deleted in a
`finally` block, so nothing is left behind on success or failure.
"""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from . import media
from .cache import file_sha256
from .config import ConfigError, get_config, setup_logging
from .net import ProviderError
from .download import download
from .schemas import MediaInfo, Transcript
from .transcribe import LANGUAGE_NAMES, transcribe

log = logging.getLogger(__name__)
# on_stage(stage_name, fraction_complete_of_that_stage)
StageFn = Callable[[str, float], None]

MUTED_MAX_DB = -50.0  # if even the loudest sample is below this, treat as muted


class PipelineResult(BaseModel):
    file_hash: str
    media: MediaInfo
    audio_levels: dict[str, float]
    transcript: Transcript
    warnings: list[str] = Field(default_factory=list)


def run(src_path: str | Path | None = None, on_stage: StageFn | None = None, *,
        url: str | None = None) -> PipelineResult:
    """Run on a local file (`src_path`) or a link (`url`). Exactly one must be given."""
    if (src_path is None) == (url is None):
        raise ValueError("pass exactly one of src_path or url")
    notify = on_stage or (lambda stage, frac: None)
    work = Path(tempfile.mkdtemp(prefix="classeval_"))
    try:
        if url is not None:
            notify("Downloading", 0.0)
            src = download(url, work, on_progress=lambda f: notify("Downloading", f))
        else:
            src = Path(src_path)
        notify("Validating file", 0.0)
        info = media.validate(src)
        file_hash = file_sha256(src)
        log.info("validated input: %.0fs, %.1f MB, video=%s", info.duration_s, info.size_mb, info.has_video)

        notify("Extracting audio", 0.0)
        wav = media.extract_audio(src, work)
        levels = media.audio_levels(wav)
        warnings: list[str] = []

        if levels["max_volume"] < MUTED_MAX_DB:
            # Whisper invents text on silence, so don't transcribe a muted track at all.
            warnings.append(f"Audio appears muted (peak {levels['max_volume']:.0f} dBFS); no transcription attempted.")
            transcript = Transcript(duration_s=info.duration_s, provider=get_config()["provider"], model="none",
                                    warnings=["No usable speech was transcribed."])
        else:
            chunks = media.chunk_audio(wav, info.duration_s, work)
            notify("Transcribing", 0.0)
            transcript = transcribe(chunks, info.duration_s, file_hash,
                                    on_progress=lambda i, n: notify("Transcribing", i / max(n, 1)))

        notify("Done", 1.0)
        return PipelineResult(file_hash=file_hash, media=info, audio_levels=levels,
                              transcript=transcript, warnings=warnings + transcript.warnings)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            log.warning("temp dir could not be fully removed: %s", work)


def summarize(result: PipelineResult) -> dict:
    """Text-free summary, safe to print or log."""
    t = result.transcript
    usable = t.usable_segments()
    speech_s = sum(s.end - s.start for s in usable)
    return {
        "file_hash": result.file_hash[:12],
        "duration_min": round(result.media.duration_s / 60, 1),
        "has_video": result.media.has_video,
        "peak_dbfs": result.audio_levels.get("max_volume"),
        "provider": t.provider,
        "model": t.model,
        "language": LANGUAGE_NAMES.get(t.language, t.language),
        "segments": len(t.segments),
        "usable_segments": len(usable),
        "speech_coverage": round(speech_s / max(result.media.duration_s, 1), 3),
        "warnings": result.warnings,
    }


if __name__ == "__main__":
    # python -m src.pipeline path/to/recording.mp4   |   python -m src.pipeline https://...
    import json

    setup_logging()
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.pipeline <recording file or link>")
    arg = sys.argv[1]
    progress = lambda s, f: print(f"[{s}] {f:.0%}", file=sys.stderr)  # noqa: E731
    is_link = arg.startswith(("http://", "https://"))
    try:
        res = run(url=arg, on_stage=progress) if is_link else run(arg, on_stage=progress)
    except (media.MediaError, ConfigError, ProviderError) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(summarize(res), indent=2))
