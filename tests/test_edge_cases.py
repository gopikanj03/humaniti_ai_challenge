"""Edge cases from the brief: muted audio, corrupt/unsupported/very short files, overlapping speech,
API failure fallback, repeat-run consistency and private-data checks.

Media tests generate synthetic clips with ffmpeg in pytest's temp dir (skipped if ffmpeg is missing).
No test calls the network: the provider is replaced by fakes.
"""
import json
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from src import cache, llm, media, pipeline, scoring, transcribe
from src.config import ROOT, get_config
from src.net import ProviderError
from src.schemas import Assessment, AudioChunk, DimensionScore, Segment

FFMPEG = get_config()["ffmpeg_bin"]
needs_ffmpeg = pytest.mark.skipif(shutil.which(FFMPEG) is None, reason="ffmpeg not installed")


def _clip(path: Path, seconds: float, audio: str = "anullsrc=r=16000:cl=mono") -> Path:
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=5",
                    "-f", "lavfi", "-i", audio, "-t", str(seconds), "-c:v", "libx264", "-c:a", "aac", str(path)],
                   check=True)
    return path


# ---------------------------------------------------------------- files

def test_unsupported_extension(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(media.MediaError, match="Unsupported file type"):
        media.validate(f)


@needs_ffmpeg
def test_corrupt_file(tmp_path):
    good = _clip(tmp_path / "ok.mp4", 40, "sine=frequency=440:sample_rate=16000")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(good.read_bytes()[:3000])  # truncated: headers gone
    with pytest.raises(media.MediaError):
        media.validate(bad)


@needs_ffmpeg
def test_very_short_clip_is_rejected_with_a_clear_message(tmp_path):
    with pytest.raises(media.MediaError, match="at least"):
        media.validate(_clip(tmp_path / "short.mp4", 8))


@needs_ffmpeg
def test_muted_recording_gives_one_label_with_zero_confidence_and_no_api_calls(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("muted audio must not reach the provider")
    monkeypatch.setattr(transcribe, "_groq_transcribe", no_network)
    monkeypatch.setattr(llm, "_chat", no_network)
    monkeypatch.setitem(get_config()["visual"], "interval_s", 10)

    clip = _clip(tmp_path / "muted.mp4", 45)
    before = set(Path(tempfile_dir()).glob("classeval_*"))
    res = pipeline.run(clip)
    assert res.result.label in scoring.LABELS
    assert res.result.confidence == 0.0 and res.result.low_confidence
    assert any("muted" in w for w in res.warnings)
    assert set(Path(tempfile_dir()).glob("classeval_*")) <= before, "this run's temp dir must be removed"


def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()


# ---------------------------------------------------------------- overlapping speech

def test_overlapping_speech_is_labelled_mixed_not_assigned_to_one_speaker():
    import numpy as np
    from src import speakers
    windows = [(0, 1.5), (0.75, 2.25), (1.5, 3.0), (2.25, 3.75)]
    res = speakers.build_result(windows, np.array([0, 1, 0, 1]), np.zeros((4, 2)), [(0, 4)])
    seg = Segment(start=0.5, end=3.5, text="both talking")
    speakers.label_segments([seg], res)
    assert seg.speaker == "mixed"


# ---------------------------------------------------------------- API failure

def _chunk(tmp_path):
    return AudioChunk(index=0, start_s=0, end_s=60, path=str(tmp_path / "c.flac"), size_mb=1)


def test_hosted_failure_falls_back_to_local_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setitem(get_config()["transcription"], "fallback_to_local", True)
    monkeypatch.setattr(cache, "load", lambda *a: None)
    monkeypatch.setattr(cache, "save", lambda *a: None)
    monkeypatch.setattr(transcribe, "_groq_transcribe",
                        lambda c, language=None: (_ for _ in ()).throw(ProviderError("HTTP 503")))
    monkeypatch.setattr(transcribe, "_local_transcribe", lambda c, language=None: [
        {"start": 1, "end": 3, "text": "hello class", "avg_logprob": -0.2, "no_speech_prob": 0.1, "language": "en"}])
    t = transcribe.transcribe([_chunk(tmp_path)], 60, "h")
    assert t.segments[0].text == "hello class" and "faster-whisper" in t.model
    assert any("transcribed locally" in w for w in t.warnings)


def test_hosted_failure_surfaces_as_error_when_fallback_disabled(tmp_path, monkeypatch):
    monkeypatch.setitem(get_config()["transcription"], "fallback_to_local", False)
    monkeypatch.setattr(cache, "load", lambda *a: None)
    monkeypatch.setattr(transcribe, "_groq_transcribe",
                        lambda c, language=None: (_ for _ in ()).throw(ProviderError("HTTP 503")))
    with pytest.raises(ProviderError):
        transcribe.transcribe([_chunk(tmp_path)], 60, "h")


def test_invalid_judge_json_is_retried_once_then_reported(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_chat", lambda *a: calls.append(1) or "not json at all")
    with pytest.raises(llm.LLMOutputError):
        llm.chat_json("m", "sys", "user", Assessment, max_tokens=100)
    assert len(calls) == 2


def test_invalid_json_then_valid_json_succeeds(monkeypatch):
    replies = iter(['{"summary": "x"}', json.dumps({"summary": "ok", "dimensions": [], "confidence": 0.5})])
    monkeypatch.setattr(llm, "_chat", lambda *a: next(replies))
    assert llm.chat_json("m", "sys", "user", Assessment, max_tokens=100).summary == "ok"


# ---------------------------------------------------------------- repeat-run consistency

def test_scoring_is_deterministic_for_the_same_judge_output():
    a = Assessment(summary="", confidence=0.7, dimensions=[DimensionScore(dimension=d["id"], score=3.2)
                                                             for d in scoring.load_rubric()["dimensions"]])
    results = {scoring.score(a).model_dump_json() for _ in range(5)}
    assert len(results) == 1


def test_repeat_run_reuses_existing_results(monkeypatch, tmp_path):
    """Existing cache entries are reused, so re-analysing a recording gives identical results."""
    monkeypatch.setitem(get_config()["cache"], "dir", str(tmp_path))
    monkeypatch.setitem(get_config()["cache"], "write", True)
    calls = []
    monkeypatch.setattr(llm, "_chat", lambda *a: calls.append(1) or json.dumps(
        {"summary": "s", "dimensions": [], "confidence": 0.4}))
    first = llm.chat_json("m", "sys", "u", Assessment, max_tokens=10, file_hash="abc", cache_tag="t")
    second = llm.chat_json("m", "sys", "u", Assessment, max_tokens=10, file_hash="abc", cache_tag="t")
    assert first == second and len(calls) == 1


# ---------------------------------------------------------------- private data

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus",
             ".jpg", ".jpeg", ".vtt", ".srt", ".pdf"}
ALLOWED = {"assets/logo.png", "assets/8x8_logo_template.png", "assets/8x8_logo_template_2.png",
           "assets/8x8_logo_template_3.png"}  # logo-only crops, checked for faces when cut


def _tracked_files() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def test_no_media_frames_transcripts_or_feedback_would_be_committed():
    files = _tracked_files()
    bad = [f for f in files if f.startswith("data/") or Path(f).suffix.lower() in MEDIA_EXT
           or (f.endswith(".png") and f not in ALLOWED) or Path(f).name in (".env", "anchors.json", "style.json")]
    assert not bad, f"private files would be committed: {bad}"


def test_gitignore_covers_private_paths():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/", ".env", "config/anchors*.json", "config/style*.json", "*.mp4", "*.wav"):
        assert pattern in gi


def test_transcript_text_never_reaches_the_logs(tmp_path, monkeypatch, caplog):
    secret = "Ananya please move your knight"
    monkeypatch.setattr(cache, "load", lambda *a: None)
    monkeypatch.setattr(cache, "save", lambda *a: None)
    monkeypatch.setattr(transcribe, "_groq_transcribe", lambda c, language=None: [
        {"start": 1, "end": 4, "text": secret, "avg_logprob": -0.1, "no_speech_prob": 0.0, "language": "en"}])
    with caplog.at_level(logging.DEBUG):
        t = transcribe.transcribe([_chunk(tmp_path)], 60, "h")
    assert t.segments[0].text == secret
    assert "Ananya" not in caplog.text and "knight" not in caplog.text


def test_stale_temp_from_a_killed_run_is_swept_but_live_runs_are_kept(monkeypatch, tmp_path):
    import os
    import time
    monkeypatch.setattr(pipeline.tempfile, "gettempdir", lambda: str(tmp_path))
    old, live = tmp_path / "classeval_killed", tmp_path / "classeval_running"
    for d in (old, live):
        d.mkdir()
        (d / "audio.wav").write_bytes(b"x")
    stale = time.time() - pipeline.STALE_TEMP_S - 60
    os.utime(old, (stale, stale))
    assert pipeline.sweep_stale_temp() == 1
    assert not old.exists() and live.exists()


# ---------------------------------------------------------------- free-tier limits

class _R:
    def __init__(self, status, headers=None, body=None):
        self.status_code, self.headers, self._body = status, headers or {}, body or {}

    def json(self):
        return self._body


def test_daily_limit_fails_fast_instead_of_waiting(monkeypatch):
    from src import net
    slept = []
    monkeypatch.setattr(net.time, "sleep", slept.append)
    monkeypatch.setattr(net, "_throttle", net.Throttle(1000))
    daily = _R(429, {"retry-after": "1255"}, {"error": {"message": "Rate limit ... tokens per day (TPD): Limit 200000"}})
    with pytest.raises(ProviderError) as e:
        net.request_with_retry(lambda: daily)
    assert e.value.code == "daily_limit" and "21 min" in str(e.value) and slept == []


def test_short_rate_limit_still_waits_and_retries(monkeypatch):
    from src import net
    slept = []
    monkeypatch.setattr(net.time, "sleep", slept.append)
    monkeypatch.setattr(net, "_throttle", net.Throttle(1000))
    replies = iter([_R(429, {"retry-after": "12"}, {"error": {"message": "tokens per minute (TPM)"}}), _R(200)])
    assert net.request_with_retry(lambda: next(replies)).status_code == 200 and slept == [12.0]


def test_model_falls_back_when_daily_limit_is_hit(monkeypatch):
    calls = []

    def fake_chat(model, messages, max_tokens, effort):
        calls.append((model, effort))
        if model == "first":
            raise ProviderError("limit", code="daily_limit")
        return json.dumps({"summary": "ok", "dimensions": [], "confidence": 0.5})

    monkeypatch.setattr(llm, "_chat", fake_chat)
    llm.fallbacks_used.clear()
    out = llm.chat_json(["first", "second"], "sys", "u", Assessment, max_tokens=10, reasoning_effort="low")
    assert out.summary == "ok" and [m for m, _ in calls] == ["first", "second"]
    assert all(effort is None for _, effort in calls)          # reasoning option only sent to gpt-oss models
    assert llm.fallbacks_used and "second" in llm.fallbacks_used[0]


def test_sections_alternate_between_model_lanes_and_exhausted_lane_uses_a_free_model(monkeypatch):
    from src import condense as cd
    from src.schemas import Transcript
    monkeypatch.setitem(get_config()["llm"], "condense_model", "A")
    monkeypatch.setitem(get_config()["llm"], "condense_parallel_models", ["B"])
    monkeypatch.setitem(get_config()["llm"], "condense_fallback_models", ["C", "B"])
    chains = []
    monkeypatch.setattr(cd, "chat_json", lambda models, *a, **k: chains.append(models) or cd._RawDigest(summary="s"))
    segs = [Segment(start=i * 700 + 1, end=i * 700 + 5, text="hello", chunk_index=i) for i in range(4)]
    cd.condense(Transcript(segments=segs, duration_s=2800, provider="groq", model="m"), "h")
    assert sorted(c[0] for c in chains) == ["A", "A", "B", "B"]          # two lanes, alternating
    assert all(c[1] == "C" for c in chains if c[0] == "A")              # A's fallback avoids lane B first
