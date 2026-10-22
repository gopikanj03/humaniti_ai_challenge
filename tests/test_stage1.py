"""Stage 1 unit tests that need neither ffmpeg nor a network."""
import pytest
import requests

from src import net
from src.media import plan_chunks
from src.schemas import AudioChunk
from src.transcribe import merge_chunks


@pytest.mark.parametrize("duration", [30, 599, 600, 601, 1805, 3600, 7200])
def test_plan_chunks_covers_duration_without_gaps(duration):
    plan = plan_chunks(duration, target=600, min_len=10)
    assert plan[0][0] == 0
    assert plan[-1][1] == pytest.approx(duration)
    for (a0, a1), (b0, _) in zip(plan, plan[1:]):
        assert a1 == pytest.approx(b0)
    assert all(600 + 1e-6 >= e - s >= 10 for s, e in plan)


def test_plan_chunks_never_leaves_tiny_tail():
    # 601 s used to mean 600 + 1; equal splitting gives two ~300 s chunks.
    assert [round(e - s) for s, e in plan_chunks(601, 600, 10)] == [300, 300]


def _chunk(i, start, end):
    return AudioChunk(index=i, start_s=start, end_s=end, path=f"c{i}.flac", size_mb=1)


def test_merge_offsets_by_chunk_start_and_sorts():
    chunks = [_chunk(0, 0, 600), _chunk(1, 600, 1200)]
    per_chunk = [
        [{"start": 1.0, "end": 3.0, "text": " What is seven times eight? ", "avg_logprob": -0.2, "no_speech_prob": 0.01}],
        [{"start": 0.5, "end": 2.0, "text": "Fifty-six", "avg_logprob": -0.3, "no_speech_prob": 0.02}],
    ]
    segs = merge_chunks(chunks, per_chunk)
    assert [(s.start, s.end) for s in segs] == [(1.0, 3.0), (600.5, 602.0)]
    assert segs[0].text == "What is seven times eight?"
    assert not any(s.flagged for s in segs)


def test_merge_clamps_segment_to_chunk_end():
    segs = merge_chunks([_chunk(0, 0, 600)], [[{"start": 598, "end": 604, "text": "bye"}]])
    assert segs[0].end == 600


@pytest.mark.parametrize("seg", [
    {"start": 0, "end": 2, "text": "Thanks for watching!", "avg_logprob": -1.4, "no_speech_prob": 0.9},
    {"start": 0, "end": 2, "text": "ok ok ok ok ok ok ok ok", "compression_ratio": 3.1},
    {"start": 0, "end": 2, "text": "   "},
])
def test_low_confidence_segments_are_flagged(seg):
    assert merge_chunks([_chunk(0, 0, 600)], [[seg]])[0].flagged


class _Resp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}


def test_retry_honours_429_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(net.time, "sleep", sleeps.append)
    monkeypatch.setattr(net, "_throttle", net.Throttle(1000))
    responses = iter([_Resp(429, {"retry-after": "7"}), _Resp(200)])
    assert net.request_with_retry(lambda: next(responses)).status_code == 200
    assert sleeps == [7.0]


def test_non_retryable_error_raises_without_body(monkeypatch):
    monkeypatch.setattr(net, "_throttle", net.Throttle(1000))
    with pytest.raises(net.ProviderError, match="rejected the API key"):
        net.request_with_retry(lambda: _Resp(401))


def test_network_errors_exhaust_retries(monkeypatch):
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    monkeypatch.setattr(net, "_throttle", net.Throttle(1000))

    def boom():
        raise requests.ConnectionError("down")

    with pytest.raises(net.ProviderError, match="network error"):
        net.request_with_retry(boom)


# ---------------------------------------------------------------- link input

from src.download import check_url
from src.media import MediaError


@pytest.mark.parametrize("url", ["ftp://example.com/a.mp4", "example.com/a.mp4", "file:///C:/x.mp4",
                                 "http://127.0.0.1/a.mp4", "http://localhost:8501/", "http://192.168.1.5/a.mp4"])
def test_check_url_rejects_non_http_and_private_hosts(url):
    with pytest.raises(MediaError):
        check_url(url)


def test_dominant_language_weights_by_speech_time():
    from src.schemas import Segment
    from src.transcribe import dominant_language
    segs = [Segment(start=0, end=50, text="namaskaram", language="ml"),
            Segment(start=50, end=60, text="okay", language="en"),
            Segment(start=60, end=200, text="...", language="en", flagged=True)]  # flagged doesn't count
    assert dominant_language(segs) == "ml"


def test_language_lock_replaces_misdetected_chunk_only_if_more_confident(monkeypatch):
    from src import transcribe as tr
    monkeypatch.setattr(tr.cache, "load", lambda *a: None)
    monkeypatch.setattr(tr.cache, "save", lambda *a: None)
    chunks = [_chunk(0, 0, 600), _chunk(1, 600, 1200), _chunk(2, 1200, 1800)]
    seg = lambda lang, lp: {"start": 0, "end": 500, "text": "x", "avg_logprob": lp, "language": lang}  # noqa: E731
    per_chunk = [[seg("English", -0.9)], [seg("Malayalam", -0.4)], [seg("Malayalam", -0.5)]]
    calls = []

    def fake_groq(chunk, language=None):
        calls.append((chunk.index, language))
        return [seg("Malayalam", -0.6)]

    monkeypatch.setattr(tr, "_groq_transcribe", fake_groq)
    tr._lock_language(chunks, per_chunk, ["groq"] * 3, "hash")
    assert calls == [(0, "ml")]                         # only the disagreeing chunk is redone, forced to ml
    assert per_chunk[0][0]["language"] == "Malayalam"   # forced version was more confident, so it's kept

    per_chunk = [[seg("English", -0.2)], [seg("Malayalam", -0.4)], [seg("Malayalam", -0.5)]]
    tr._lock_language(chunks, per_chunk, ["groq"] * 3, "hash")
    assert per_chunk[0][0]["language"] == "English"     # a confident English stretch survives


def test_youtube_bot_check_is_not_reported_as_a_private_link():
    from src.download import _friendly, _is_bot_check
    err = "ERROR: [youtube] mbZIjNKlxuY: Sign in to confirm you're not a bot. Use --cookies-from-browser"
    assert _is_bot_check(err)
    assert "did not allow this server" in _friendly(err) and "private" not in _friendly(err).lower()
    assert "private" in _friendly("ERROR: This video is private").lower()


def test_bot_check_retries_with_other_player_types(monkeypatch, tmp_path):
    import yt_dlp
    from src import download
    tried = []

    class FakeYDL:
        def __init__(self, opts): self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download):
            tried.append((self.opts.get("extractor_args") or {}).get("youtube", {}).get("player_client"))
            if len(tried) < 2:
                raise yt_dlp.utils.DownloadError("Sign in to confirm you're not a bot")
            (tmp_path / "download.mp4").write_bytes(b"x")
            return {"duration": 50}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(download, "check_url", lambda u: u)
    assert download.download("https://youtube.com/shorts/x", tmp_path).name == "download.mp4"
    assert tried == [None, ["tv", "mweb"]]
