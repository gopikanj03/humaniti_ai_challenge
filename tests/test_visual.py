"""Stage 5 tests: frame sampling, visual summaries and visual-evidence verification (no video needed)."""
from src import evaluator, visual
from src.schemas import Assessment, ChunkDigest, DimensionScore, EvidenceItem, Event
from src.visual import FrameObs, VisualResult


def test_sample_times_respects_interval_and_cap():
    ts = visual.sample_times(3600, 45, 120)
    assert len(ts) == 80 and ts[0] == 22.5 and all(b - a == 45 for a, b in zip(ts, ts[1:]))
    assert len(visual.sample_times(4 * 3600, 45, 120)) <= 120     # long recordings: wider spacing


def _frames():
    f = []
    for i in range(10):
        f.append(FrameObs(t=i * 45, faces=1, largest_face=0.3, camera_view=True, logo=0.95 if i < 3 else 0.4,
                          changed=None if i == 0 else True))
    for i in range(10, 18):  # a long stretch of screen sharing with no face found
        f.append(FrameObs(t=i * 45, faces=0, largest_face=0.0, camera_view=False, changed=i % 2 == 0))
    return f


def test_summary_metrics_and_timestamped_observations():
    res = VisualResult(interval_s=45, frames=_frames())
    visual._summarise(res, logo_available=True)
    m = {x.name: x.value for x in res.metrics}
    assert m["frames_sampled"] == 18 and m["camera_view_share"] == 0.56 and m["face_visible_share"] == 0.56
    assert m["logo_in_camera_view_share"] == 0.3 and m["no_face_share"] == 0.44
    assert any("[00:00] tutor camera shown large; 8x8 logo visible" in o for o in res.observations)
    assert any(o.startswith("[07:30-12:45] no face detected") for o in res.observations)
    assert all("lighting" not in o.lower() for o in res.observations)   # not measured, by design


def test_visual_evidence_must_match_a_sampled_frame(monkeypatch):
    monkeypatch.setattr(evaluator, "load_rubric", lambda: {"dimensions": [
        {"id": "setup", "weight": 1, "requires": "visual"}]})
    obs = ["[06:22] tutor camera shown large; 8x8 logo visible.", "[10:00-14:30] no face detected ..."]
    vt = evaluator.visual_times(obs)
    assert vt == [382, 600, 870]
    a = Assessment(summary="", confidence=0.7, dimensions=[DimensionScore(dimension="setup", score=4, evidence=[
        EvidenceItem(timestamp="06:30", observation="logo visible", source="visual"),        # 8 s off: ok
        EvidenceItem(timestamp="whole class", observation="logo in 2/7", source="visual"),   # summary: ok
        EvidenceItem(timestamp="30:00", observation="tutor in a dark room", source="visual"),  # invented
    ])])
    digests = [ChunkDigest(chunk_index=0, start=0, end=600, summary="", events=[Event(t=1, type="question", detail="q")])]
    out, dropped = evaluator.verify(a, digests, has_visual=True, vtimes=vt)
    assert dropped == 1 and out.dimensions[0].score == 4 and len(out.dimensions[0].evidence) == 2
