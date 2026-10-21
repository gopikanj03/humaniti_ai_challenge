"""Speech-to-text via Groq Whisper (hosted) or faster-whisper (local), merged to global timestamps."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable

import requests

from . import cache
from .config import ConfigError, get_config, groq_api_key
from .net import ProviderError, request_with_retry
from .schemas import AudioChunk, Segment, Transcript

log = logging.getLogger(__name__)
ProgressFn = Callable[[int, int], None]


# ---------------------------------------------------------------- providers

def _groq_transcribe(chunk: AudioChunk, language: str | None = None) -> list[dict]:
    cfg = get_config()
    tcfg = cfg["transcription"]
    url = f"{cfg['groq']['base_url']}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {groq_api_key()}"}
    data = {
        "model": tcfg["groq"]["model"],
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "temperature": "0",
    }
    if language or tcfg.get("language"):
        data["language"] = language or tcfg["language"]

    def send() -> requests.Response:
        with open(chunk.path, "rb") as fh:
            return requests.post(url, headers=headers, data=data, files={"file": (Path(chunk.path).name, fh)},
                                 timeout=cfg["groq"]["timeout_s"])

    body = request_with_retry(send).json()
    return [
        {"start": s["start"], "end": s["end"], "text": s.get("text", ""),
         "avg_logprob": s.get("avg_logprob"), "no_speech_prob": s.get("no_speech_prob"),
         "compression_ratio": s.get("compression_ratio"), "language": to_code(body.get("language"))}
        for s in body.get("segments", [])
    ]


@lru_cache(maxsize=1)
def _local_model():
    from faster_whisper import WhisperModel  # heavy import; only when needed

    lcfg = get_config()["transcription"]["local"]
    log.info("loading faster-whisper model %s (%s, %s)", lcfg["model"], lcfg["device"], lcfg["compute_type"])
    return WhisperModel(lcfg["model"], device=lcfg["device"], compute_type=lcfg["compute_type"])


def _local_transcribe(chunk: AudioChunk, language: str | None = None) -> list[dict]:
    tcfg = get_config()["transcription"]
    segments, info = _local_model().transcribe(
        chunk.path, language=language or tcfg.get("language"), vad_filter=tcfg["local"]["vad_filter"],
        beam_size=5, temperature=0.0, condition_on_previous_text=False,  # reduces runaway repetition
    )
    return [
        {"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob,
         "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio,
         "language": info.language}
        for s in segments
    ]


def model_name(provider: str) -> str:
    t = get_config()["transcription"]
    return t["groq"]["model"] if provider == "groq" else f"faster-whisper-{t['local']['model']}"


# Groq reports language names ("Malayalam"), faster-whisper reports codes ("ml"); normalise to codes
# so languages compare equal across providers and can be passed back to force a language.
LANGUAGE_CODES = {
    "english": "en", "malayalam": "ml", "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "marathi": "mr", "bengali": "bn", "gujarati": "gu", "punjabi": "pa", "urdu": "ur", "nepali": "ne",
    "sinhala": "si", "french": "fr", "spanish": "es", "german": "de", "italian": "it", "portuguese": "pt",
    "dutch": "nl", "russian": "ru", "arabic": "ar", "turkish": "tr", "persian": "fa", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "indonesian": "id", "malay": "ms", "thai": "th", "vietnamese": "vi",
    "polish": "pl", "ukrainian": "uk", "swedish": "sv", "greek": "el", "hebrew": "he", "swahili": "sw",
}
LANGUAGE_NAMES = {code: name.title() for name, code in LANGUAGE_CODES.items()}


def to_code(language: str | None) -> str | None:
    if not language:
        return None
    lang = language.strip().lower()
    return LANGUAGE_CODES.get(lang, lang)  # unknown names pass through; they just won't be force-able


def _median_logprob(segs: list[dict]) -> float:
    values = sorted(s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None)
    return values[len(values) // 2] if values else float("-inf")


# ---------------------------------------------------------------- merge / flag

def _flag(seg: dict) -> bool:
    """Whisper confidence heuristics for hallucinated or unintelligible segments."""
    t = get_config()["transcription"]
    nsp, lp, cr = seg.get("no_speech_prob"), seg.get("avg_logprob"), seg.get("compression_ratio")
    if not seg.get("text", "").strip():
        return True
    if nsp is not None and lp is not None and nsp > t["flag_no_speech_prob"] and lp < t["flag_avg_logprob"]:
        return True
    return cr is not None and cr > t["flag_compression_ratio"]


def merge_chunks(chunks: list[AudioChunk], per_chunk: list[list[dict]]) -> list[Segment]:
    """Shift chunk-local timestamps by each chunk's global start and return sorted segments."""
    merged: list[Segment] = []
    for chunk, segs in zip(chunks, per_chunk):
        for s in segs:
            start = chunk.start_s + max(0.0, float(s["start"]))
            end = min(chunk.end_s, chunk.start_s + float(s["end"]))  # never spill past the chunk
            merged.append(Segment(
                start=round(start, 2), end=round(max(start, end), 2), text=s.get("text", "").strip(),
                avg_logprob=s.get("avg_logprob"), no_speech_prob=s.get("no_speech_prob"),
                compression_ratio=s.get("compression_ratio"), language=to_code(s.get("language")),
                chunk_index=chunk.index, flagged=_flag(s),
            ))
    merged.sort(key=lambda x: (x.start, x.end))
    return merged


# ---------------------------------------------------------------- entry point

def transcribe(chunks: list[AudioChunk], duration: float, file_hash: str,
               on_progress: ProgressFn | None = None) -> Transcript:
    cfg = get_config()
    provider = cfg["provider"]
    warnings: list[str] = []
    per_chunk: list[list[dict]] = []
    used_by_chunk: list[str] = []

    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(i, len(chunks))
        try:
            segs, used = _cached_run(provider, chunk, file_hash), provider
        except ProviderError as e:
            if not cfg["transcription"]["fallback_to_local"]:
                raise
            log.warning("hosted transcription failed for chunk %d (%s); falling back to local", i, e)
            warnings.append(f"Chunk {i + 1}: hosted transcription failed, transcribed locally instead.")
            segs, used = _cached_run("local", chunk, file_hash), "local"
        per_chunk.append(segs)
        used_by_chunk.append(used)

    if on_progress:
        on_progress(len(chunks), len(chunks))

    if cfg["transcription"].get("language") is None:
        _lock_language(chunks, per_chunk, used_by_chunk, file_hash)

    models_used = {model_name(u) for u in used_by_chunk}
    segments = merge_chunks(chunks, per_chunk)
    n_flagged = sum(s.flagged for s in segments)
    if segments and n_flagged:
        warnings.append(f"{n_flagged} of {len(segments)} segments had low transcription confidence "
                        "and are excluded from evidence.")
    if not [s for s in segments if not s.flagged]:
        warnings.append("No usable speech was transcribed.")
    log.info("transcribed %d segments (%d flagged) with %s", len(segments), n_flagged, sorted(models_used))
    return Transcript(segments=segments, duration_s=duration, provider=provider,
                      model="+".join(sorted(models_used)), language=dominant_language(segments),
                      warnings=warnings)


def _cached_run(provider: str, chunk: AudioChunk, file_hash: str, language: str | None = None) -> list[dict]:
    key = f"transcript_{model_name(provider)}_{chunk.start_s:.0f}-{chunk.end_s:.0f}"
    if language:
        key += f"_forced-{language}"
    segs = cache.load(file_hash, key)
    if segs is None:
        run = _groq_transcribe if provider == "groq" else _local_transcribe
        segs = run(chunk, language=language)
        cache.save(file_hash, key, segs)
    return segs


def _lock_language(chunks: list[AudioChunk], per_chunk: list[list[dict]], used_by_chunk: list[str],
                   file_hash: str) -> None:
    """Re-transcribe chunks whose auto-detected language disagrees with the recording's main language.

    Whisper guesses the language per chunk, and guesses badly on chunks with little speech
    (e.g. a quiet start of a Malayalam class came back as "English"). The forced version is
    kept only if Whisper is at least as confident in it, so genuinely different-language
    stretches survive. Updates per_chunk in place.
    """
    main = dominant_language(merge_chunks(chunks, per_chunk))
    if main not in LANGUAGE_NAMES:  # unknown/unsupported code: can't force it
        return
    for i, (chunk, segs) in enumerate(zip(chunks, per_chunk)):
        if not segs or {to_code(s.get("language")) for s in segs} == {main}:
            continue
        try:
            forced = _cached_run(used_by_chunk[i], chunk, file_hash, language=main)
        except ProviderError as e:
            log.warning("language re-check failed for chunk %d (%s); keeping auto-detected version", i, e)
            continue
        if forced and _median_logprob(forced) >= _median_logprob(segs):
            per_chunk[i] = forced
            log.info("chunk %d re-transcribed as %s (was %s)", i, main, {s.get('language') for s in segs})


def dominant_language(segments: list[Segment]) -> str | None:
    """Language with the most speech time (classes may mix languages, e.g. Malayalam + English)."""
    by_lang: dict[str, float] = {}
    for s in segments:
        if s.language and not s.flagged:
            by_lang[s.language] = by_lang.get(s.language, 0.0) + (s.end - s.start)
    return max(by_lang, key=by_lang.get) if by_lang else None


__all__ = ["transcribe", "merge_chunks", "ConfigError", "ProviderError"]
