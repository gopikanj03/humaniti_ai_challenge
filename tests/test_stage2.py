"""Stage 2 unit tests: scoring policy, evidence verification, quote matching, pseudonyms, metrics."""
import pytest

from src import evaluator, scoring
from src.condense import NameMap, parse_ts, quote_matches, verify_events, _RawDigest, _RawEvent
from src.metrics import _student_turns, event_metrics
from src.schemas import Assessment, ChunkDigest, DimensionScore, EvidenceItem, Event, Segment

RUBRIC = {"dimensions": [{"id": "a", "weight": 0.5}, {"id": "b", "weight": 0.3}, {"id": "c", "weight": 0.2}]}


def _assess(scores: dict, conf=0.8):
    return Assessment(summary="", confidence=conf,
                      dimensions=[DimensionScore(dimension=k, score=v) for k, v in scores.items()])


# ---------------------------------------------------------------- scoring

@pytest.mark.parametrize("value,label", [(0.0, "Poor"), (2.24, "Poor"), (2.25, "Average"), (2.99, "Average"),
                                         (3.0, "Good"), (3.75, "Excellent"), (5.0, "Excellent")])
def test_label_thresholds(value, label):
    th = {"Poor": 0.0, "Average": 2.25, "Good": 3.0, "Excellent": 3.75}
    assert scoring.label_for(value, th) == label


def test_weighted_score_all_present():
    r = scoring.score(_assess({"a": 4, "b": 2, "c": 3}), RUBRIC)
    assert r.score == pytest.approx(0.5 * 4 + 0.3 * 2 + 0.2 * 3)
    assert r.weight_coverage == 1.0 and r.missing_dimensions == []
    assert r.confidence == pytest.approx(0.8)


def test_missing_dimension_renormalises_and_lowers_confidence():
    r = scoring.score(_assess({"a": 4, "b": 2, "c": None}), RUBRIC)
    assert r.score == pytest.approx((0.5 * 4 + 0.3 * 2) / 0.8, abs=0.01)
    assert r.missing_dimensions == ["c"]
    assert r.weight_coverage == pytest.approx(0.8)
    assert r.confidence == pytest.approx(0.8 * 0.8)


def test_low_coverage_is_flagged_low_confidence():
    r = scoring.score(_assess({"a": 5, "b": None, "c": None}, conf=1.0), RUBRIC)
    assert r.weight_coverage == 0.5 and r.low_confidence


def test_nothing_scored_returns_one_label_with_zero_confidence():
    r = scoring.score(_assess({"a": None, "b": None, "c": None}), RUBRIC)
    assert r.label in scoring.LABELS and r.confidence == 0.0 and r.low_confidence


def test_quality_factors_reduce_confidence_with_reasons():
    r = scoring.score(_assess({"a": 3, "b": 3, "c": 3}, conf=1.0), RUBRIC, quality=[(0.5, "weak speakers")])
    assert r.confidence == 0.5 and "weak speakers" in r.confidence_reasons


def test_schema_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        DimensionScore(dimension="a", score=7)


# ---------------------------------------------------------------- evidence verification

def _digest(times):
    return [ChunkDigest(chunk_index=0, start=0, end=600, summary="",
                        events=[Event(t=t, type="question", detail="q") for t in times])]


def test_verify_drops_unmatched_evidence_and_unsupported_scores(monkeypatch):
    monkeypatch.setattr(evaluator, "load_rubric", lambda: {"dimensions": [
        {"id": "a", "weight": 1}, {"id": "b", "weight": 1}, {"id": "v", "weight": 1, "requires": "visual"}]})
    a = Assessment(summary="", confidence=0.7, dimensions=[
        DimensionScore(dimension="a", score=4, evidence=[EvidenceItem(timestamp="02:05", observation="ok")]),
        DimensionScore(dimension="b", score=4, evidence=[EvidenceItem(timestamp="09:00", observation="invented")]),
        DimensionScore(dimension="v", score=5, evidence=[EvidenceItem(timestamp="01:00", observation="logo",
                                                                      source="visual")]),
    ])
    out, dropped = evaluator.verify(a, _digest([120.0]), has_visual=False)
    scores = {d.dimension: d.score for d in out.dimensions}
    assert scores == {"a": 4, "b": None, "v": None}  # b: evidence fabricated; v: no visual input
    assert dropped == 2                                # b's invented item + v's visual claim with no frames


def test_verify_fills_dimensions_the_judge_omitted(monkeypatch):
    monkeypatch.setattr(evaluator, "load_rubric", lambda: {"dimensions": [{"id": "a", "weight": 1},
                                                                          {"id": "b", "weight": 1}]})
    out, _ = evaluator.verify(_assess({"a": None}), _digest([]), has_visual=False)
    assert [d.dimension for d in out.dimensions] == ["a", "b"]


# ---------------------------------------------------------------- condense helpers

def test_parse_ts():
    assert parse_ts("12:05") == 725 and parse_ts("1:02:03") == 3723 and parse_ts("abc") is None


@pytest.mark.parametrize("quote,text,ok", [
    ("what is the best move", "Okay, so what is the best move here?", True),
    ("What is the BEST move?", "so what is the best move here", True),      # case / punctuation
    ("നല്ല നീക്കം ഏതാണ്", "ശരി, നല്ല നീക്കം ഏതാണ്?", True),                 # Malayalam
    ("checkmate in two moves", "let's look at the board", False),           # fabricated
])
def test_quote_matching(quote, text, ok):
    assert quote_matches(quote, text) is ok


def test_verify_events_snaps_to_quoted_line_and_drops_fabrications():
    segs = [Segment(start=60, end=64, text="Good morning everyone"),
            Segment(start=70, end=75, text="Who can find the fork?"),
            Segment(start=80, end=82, text="Me, knight to f7")]
    raw = _RawDigest(summary="", events=[
        _RawEvent(t="01:12", type="question", detail="d", quote="Who can find the fork"),  # off by 2 s
        _RawEvent(t="01:20", type="student_response", detail="d", quote="bishop takes queen"),  # not said
        _RawEvent(t="05:00", type="activity", detail="d"),                                      # outside chunk
    ])
    kept, times = verify_events(raw, segs)
    assert kept[0] is not None and times[0] == 70
    assert kept[1] is None and kept[2] is None


def test_name_map_is_stable_and_scrubs_text():
    names = NameMap()
    names.teacher_names.add("reshma")
    assert names.student("Aadhi") == "Student 1"
    assert names.student("aadhi") == "Student 1"
    assert names.student("Reshma") is None                     # the teacher is not a student
    assert names.student("Anu Mary") == "Student 2"
    assert names.scrub("Aadhi, try again. Anu, well done. Reshma here") == \
        "Student 1, try again. Student 2, well done. the teacher here"


# ---------------------------------------------------------------- metrics

def test_student_turns_merge_consecutive_student_segments():
    segs = [Segment(start=0, end=5, text="t", speaker="teacher"),
            Segment(start=5, end=6, text="s", speaker="student"),
            Segment(start=6, end=8, text="s", speaker="mixed"),
            Segment(start=8, end=9, text="t", speaker="teacher"),
            Segment(start=9, end=10, text="s", speaker="student")]
    assert _student_turns(segs) == [(5, 8), (9, 10)]


def test_event_metrics_wait_time_and_individual_attention():
    segs = [Segment(start=10, end=12, text="Who knows?", speaker="teacher"),
            Segment(start=15, end=16, text="Me", speaker="student"),
            Segment(start=100, end=102, text="Any idea?", speaker="teacher")]
    events = [Event(t=10, type="question", speaker="teacher", detail="q", question_type="open", student="Student 1"),
              Event(t=100, type="question", speaker="teacher", detail="q", question_type="closed"),
              Event(t=200, type="homework", speaker="teacher", detail="hw")]
    m = {x.name: x.value for x in event_metrics([ChunkDigest(chunk_index=0, start=0, end=300, summary="",
                                                             events=events)], segs, duration=300)}
    assert m["questions_total"] == 2 and m["open_question_share"] == 0.5
    assert m["median_wait_time_s"] == 3.0 and m["question_response_rate"] == 0.5
    assert m["students_addressed_individually"] == 1 and m["homework_given"] is True


# ---------------------------------------------------------------- token budgets (free tier: 8k tokens/min)

def test_fit_to_budget_splits_long_chunks():
    from src.condense import fit_to_budget, _user_message
    from src.llm import estimate_tokens
    segs = [Segment(start=i * 5, end=i * 5 + 4, text="word " * 40) for i in range(100)]
    parts = fit_to_budget([segs], budget_tokens=800)
    assert len(parts) > 1 and sum(len(p) for p in parts) == 100
    assert all(estimate_tokens(_user_message(p)) <= 800 for p in parts)


def test_render_digests_keeps_high_priority_events_under_budget():
    from src.llm import estimate_tokens
    events = [Event(t=i, type="explanation", detail="long explanation " * 5) for i in range(40)]
    events += [Event(t=500, type="homework", detail="assigned puzzles"),
               Event(t=510, type="question", detail="asked Student 2 why", student="Student 2")]
    d = [ChunkDigest(chunk_index=0, start=0, end=600, summary="s", events=events)]
    text, omitted = evaluator.render_digests(d, budget_tokens=300)
    assert estimate_tokens(text) <= 300 and omitted > 0
    assert "homework" in text and "Student 2" in text


def test_estimate_tokens_counts_non_latin_scripts_heavily():
    from src.llm import estimate_tokens
    assert estimate_tokens("നല്ല നീക്കം" * 10) > estimate_tokens("good move " * 10)


def test_repair_truncated_json_keeps_complete_events():
    from src.llm import repair_truncated_json
    raw = '{"summary": "s", "events": [{"t": "01:00", "type": "question"}, {"t": "02:00", "type": "hom'
    assert repair_truncated_json(raw) == {"summary": "s", "events": [{"t": "01:00", "type": "question"}]}
    assert repair_truncated_json('{"a": "unterminated') is None


def test_raw_event_normalises_model_drift():
    e = _RawEvent(t=75, type="Praise", speaker="Teacher (likely)", detail="d", question_type="OPEN",
                  student_name="null")
    assert (e.t, e.type, e.speaker, e.question_type, e.student_name) == ("01:15", "praise_encouragement",
                                                                           "teacher", "open", None)


def test_render_digests_gives_rare_types_a_turn():
    # 46 student responses early in the class must not crowd out 2 late praise events.
    events = [Event(t=i, type="student_response", detail="answered the question") for i in range(46)]
    events += [Event(t=2000, type="praise_encouragement", detail="praised Student 1 for the fork", verified=True),
               Event(t=2100, type="praise_encouragement", detail="praised Student 2", verified=True)]
    d = [ChunkDigest(chunk_index=0, start=0, end=2400, summary="s", events=events)]
    text, omitted = evaluator.render_digests(d, budget_tokens=200)
    assert omitted > 0 and "praised Student 1" in text and "praised Student 2" in text


# ---------------------------------------------------------------- speakers

def test_speaker_turns_and_segment_labels():
    import numpy as np
    from src import speakers
    # Teacher speaks 0-10 s, a child 10-14 s, the teacher again 14-20 s (overlapping 1.5 s windows).
    windows = [(t, t + 1.5) for t in np.arange(0, 18.6, 0.75)]
    labels = np.array([0 if (a + b) / 2 < 10 or (a + b) / 2 >= 14 else 1 for a, b in windows])
    res = speakers.build_result(windows, labels, np.zeros((len(windows), 4)), [(0, 20)])
    assert [t.role for t in res.turns] == ["teacher", "student", "teacher"]
    assert res.teacher_share > 0.6
    segs = [Segment(start=1, end=8, text="t"), Segment(start=10.5, end=13.5, text="s"),
            Segment(start=8, end=12, text="both"), Segment(start=30, end=32, text="silence")]
    speakers.label_segments(segs, res)
    assert [s.speaker for s in segs] == ["teacher", "student", "mixed", None]


def test_too_little_speech_returns_no_separation(monkeypatch, tmp_path):
    import numpy as np
    from src import speakers
    monkeypatch.setattr(speakers, "read_wav", lambda p: (np.zeros(16000 * 5, dtype=np.float32), 16000))
    monkeypatch.setattr(speakers, "speech_regions", lambda a, sr: [(1.0, 1.4)])
    res = speakers.separate(tmp_path / "x.wav")
    assert res.teacher_share is None and res.warnings


# ---------------------------------------------------------------- cache policy: the app never writes

def test_app_never_writes_cache_but_reads_existing(monkeypatch, tmp_path):
    from src import cache
    from src.config import get_config
    c = get_config()["cache"]
    monkeypatch.setitem(c, "dir", str(tmp_path))
    monkeypatch.setitem(c, "write", False)
    cache.clear_memory()
    cache.save("abc123", "transcript_x", {"secret": "child name"})
    assert not any(tmp_path.rglob("*")), "app mode must not write anything to disk"
    assert cache.load("abc123", "transcript_x") == {"secret": "child name"}   # kept in memory for a retry

    cache.clear_memory()                                    # app restart: memory gone, nothing on disk
    assert cache.load("abc123", "transcript_x") is None

    monkeypatch.setitem(c, "write", True)                   # what dataset scripts do (enable_writes)
    cache.save("abc123", "transcript_x", {"ok": 1})
    monkeypatch.setitem(c, "write", False)
    assert cache.load("abc123", "transcript_x") == {"ok": 1}  # existing disk entries are still reused


def test_memory_keeps_only_the_last_few_recordings():
    from src import cache
    cache.clear_memory()
    for i in range(cache.MEMORY_RECORDINGS + 2):
        cache._remember(f"file{i}", "stage", i)
    assert cache.load("file0", "stage") is None and cache.load(f"file{cache.MEMORY_RECORDINGS + 1}", "stage") is not None
    cache.clear_memory()


def test_section_ranges_are_not_accepted_as_evidence():
    times = [600.0]
    assert evaluator._ts_ok("10:05", times) and evaluator._ts_ok("10:05-10:40", times)
    assert not evaluator._ts_ok("09:50-18:20", times)   # a whole section is not a specific moment


def test_judge_always_sees_event_evidence_within_the_real_budget():
    """Regression: rubric + metrics + style + anchors once filled the whole budget and the judge saw 0 events."""
    import re
    from src.config import get_config
    from src.llm import prompt_budget
    from src.schemas import Metric
    metrics = [Metric(name=f"metric_number_{i}", value=0.5, source="events", note="a note that is fairly long " * 3)
               for i in range(30)]
    events = [Event(t=60 + i * 20, type="question" if i % 2 else "explanation", detail="teacher asks why the knight "
                    "move is good and waits", quote="why is this the best move here", verified=True) for i in range(120)]
    digests = [ChunkDigest(chunk_index=0, start=0, end=3000, summary="a long summary " * 10, events=events)]
    visual = [f"[{i:02d}:00] tutor camera shown large; 8x8 logo visible." for i in range(12)]
    lc = get_config()["llm"]
    prompt = evaluator.build_prompt(digests, metrics, visual, {"duration": "50 min"},
                                    budget_tokens=prompt_budget(lc["judge_max_tokens"], evaluator.SYSTEM))
    assert len(re.findall(r"^\[\d", prompt, re.M)) >= 30
