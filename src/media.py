"""Upload validation and audio extraction/chunking with ffmpeg."""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from pathlib import Path

from .config import get_config
from .schemas import AudioChunk, MediaInfo

log = logging.getLogger(__name__)


class MediaError(ValueError):
    """User-facing validation / processing error. Message is safe to show in the UI."""


def _bin(name: str) -> str:
    exe = get_config()[f"{name}_bin"]
    if shutil.which(exe) is None:
        raise MediaError(f"{name} not found on PATH. Install ffmpeg (see README) or set {name.upper()}_BIN.")
    return exe


def _run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    # stderr is captured, never logged verbatim: it can contain file metadata.
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def validate(path: str | Path) -> MediaInfo:
    """Check extension, size, that ffprobe can read it, that audio decodes, and duration limits."""
    cfg = get_config()["media"]
    path = Path(path)

    ext = path.suffix.lower()
    if ext not in cfg["allowed_extensions"]:
        raise MediaError(f"Unsupported file type '{ext}'. Allowed: {', '.join(cfg['allowed_extensions'])}.")

    if not path.is_file():
        raise MediaError("The uploaded file could not be found.")
    size_mb = path.stat().st_size / 1e6
    if size_mb == 0:
        raise MediaError("The file is empty.")
    if size_mb > cfg["max_upload_mb"]:
        raise MediaError(f"File is {size_mb:.0f} MB; the limit is {cfg['max_upload_mb']} MB.")

    proc = _run([_bin("ffprobe"), "-v", "error", "-show_entries",
                 "format=duration,format_name:stream=codec_type,width,height", "-of", "json", str(path)], timeout=120)
    if proc.returncode != 0:
        raise MediaError("The file could not be read as audio/video (corrupt or not a media file).")
    try:
        meta = json.loads(proc.stdout)
        streams = {s.get("codec_type") for s in meta.get("streams", [])}
        duration = float(meta["format"]["duration"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        raise MediaError("Could not determine the recording's duration; the file may be corrupt.")

    if "audio" not in streams:
        raise MediaError("The recording has no audio track, so it cannot be assessed.")
    if duration < cfg["min_duration_s"]:
        raise MediaError(f"Recording is {duration:.0f}s long; at least {cfg['min_duration_s']}s is needed.")
    if duration > cfg["max_duration_s"]:
        raise MediaError(f"Recording is {duration / 60:.0f} min long; the limit is {cfg['max_duration_s'] / 60:.0f} min.")

    # ffprobe only reads headers; actually decode a short stretch to catch truncated/corrupt files.
    proc = _run([_bin("ffmpeg"), "-v", "error", "-xerror", "-i", str(path), "-t", "15",
                 "-map", "0:a:0", "-f", "null", "-"], timeout=120)
    if proc.returncode != 0:
        raise MediaError("The audio track could not be decoded; the file may be corrupt or truncated.")

    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    return MediaInfo(duration_s=duration, has_video="video" in streams, has_audio=True,
                     size_mb=round(size_mb, 1), container=meta["format"].get("format_name", ""),
                     width=video.get("width"), height=video.get("height"))


def extract_audio(src: str | Path, out_dir: str | Path) -> Path:
    """Extract 16 kHz mono 16-bit PCM WAV (used for chunking and, later, local audio metrics)."""
    sr = get_config()["media"]["sample_rate"]
    out = Path(out_dir) / "audio.wav"
    proc = _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vn", "-map", "0:a:0",
                 "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out)], timeout=1800)
    if proc.returncode != 0 or not out.exists():
        raise MediaError("Audio extraction failed.")
    return out


def audio_levels(wav: str | Path) -> dict[str, float]:
    """Mean/max volume in dBFS via ffmpeg volumedetect. Used to detect muted recordings."""
    proc = _run([_bin("ffmpeg"), "-v", "info", "-i", str(wav), "-af", "volumedetect", "-f", "null", "-"])
    levels = {}
    for key in ("mean_volume", "max_volume"):
        m = re.search(rf"{key}:\s*(-?[\d.]+|-inf) dB", proc.stderr)
        levels[key] = -120.0 if not m or m.group(1) == "-inf" else float(m.group(1))
    return levels


def plan_chunks(duration: float, target: float, min_len: float) -> list[tuple[float, float]]:
    """Split [0, duration] into ~equal chunks of at most ~target seconds.

    Equal-sized chunks (rather than target, target, ..., remainder) avoid a tiny
    trailing chunk; a chunk under min_len only happens for very short input,
    which validation already rejects.
    """
    if duration <= 0:
        return []
    n = max(1, math.ceil(duration / target))
    if n > 1 and duration / n < min_len:
        n = max(1, int(duration // min_len))
    step = duration / n
    return [(round(i * step, 3), round(duration if i == n - 1 else (i + 1) * step, 3)) for i in range(n)]


def chunk_audio(wav: str | Path, duration: float, out_dir: str | Path) -> list[AudioChunk]:
    """Cut the WAV into chunks, FLAC by default, falling back to low-bitrate MP3 if over the size cap."""
    cfg = get_config()["media"]
    out_dir = Path(out_dir)
    chunks = []
    for i, (start, end) in enumerate(plan_chunks(duration, cfg["chunk_target_s"], cfg["chunk_min_s"])):
        base = [_bin("ffmpeg"), "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(wav)]
        out = out_dir / f"chunk_{i:03d}.flac"
        proc = _run(base + ["-c:a", "flac", str(out)])
        if proc.returncode == 0 and out.stat().st_size / 1e6 > cfg["chunk_max_mb"]:
            out.unlink()
            out = out_dir / f"chunk_{i:03d}.mp3"
            proc = _run(base + ["-c:a", "libmp3lame", "-b:a", "32k", str(out)])
        if proc.returncode != 0 or not out.exists():
            raise MediaError(f"Failed to prepare audio chunk {i + 1}.")
        size_mb = out.stat().st_size / 1e6
        if size_mb > cfg["chunk_max_mb"]:
            raise MediaError(f"Audio chunk {i + 1} is still {size_mb:.1f} MB after compression.")
        chunks.append(AudioChunk(index=i, start_s=start, end_s=end, path=str(out), size_mb=round(size_mb, 2)))
    log.info("prepared %d audio chunks", len(chunks))
    return chunks
