"""Strict rules: recording-quality issues and teaching standards can only lower the label."""
from src import scoring
from src.schemas import Assessment, DimensionScore, EvidenceItem, Metric

GOOD_CLASS = {"video_height": 1080, "median_sharpness": 1500, "longest_frozen_s": 0, "near_black_frame_share": 0.0,
              "transcription_confidence": 0.68, "low_confidence_segment_share": 0.01, "speech_ratio": 0.6,
              "face_visible_share": 0.8, "homework_given": True, "recap_or_conclusion_in_last_10min": True,
              "camera_view_share": 0.2, "frames_sampled": 60, "logo_in_camera_view_share": 0.5,
              "questions_per_10min": 6.0, "praise_count": 6}


def _score(values: dict, dims: float = 4.2, errors: int = 0):
    a = Assessment(summary="", confidence=0.8,
                   dimensions=[DimensionScore(dimension=d["id"], score=dims) for d in scoring.load_rubric()["dimensions"]],
                   teaching_errors=[EvidenceItem(timestamp="10:00", observation="wrong rule") for _ in range(errors)])
    return scoring.score(a, metrics=[Metric(name=k, value=v, source="visual") for k, v in values.items()])


def test_no_rules_fire_on_a_good_class():
    r = _score(GOOD_CLASS)
    assert r.label == "Excellent" and r.rules_applied == [] and not r.quality_issue


def test_recording_quality_issue_makes_it_poor():
    for change in ({"video_height": 240}, {"median_sharpness": 40}, {"longest_frozen_s": 600},
                   {"near_black_frame_share": 0.5}, {"transcription_confidence": 0.4}, {"speech_ratio": 0.1}):
        r = _score({**GOOD_CLASS, **change})
        assert r.label == "Poor" and r.quality_issue and r.rules_applied[0].startswith("Recording quality issue"), change


def test_teaching_mistakes_cap_the_label():
    assert _score(GOOD_CLASS, errors=1).label == "Average"
    assert _score(GOOD_CLASS, errors=3).label == "Poor"


def test_single_standard_caps_at_average_and_three_make_it_poor():
    assert _score({**GOOD_CLASS, "homework_given": False, "recap_or_conclusion_in_last_10min": False}).label == "Average"
    assert _score({**GOOD_CLASS, "logo_in_camera_view_share": 0.0}).label == "Average"
    three = {**GOOD_CLASS, "logo_in_camera_view_share": 0.0, "questions_per_10min": 1.0, "praise_count": 0}
    r = _score(three)
    assert r.label == "Poor" and r.score_label == "Excellent" and len(r.rules_applied) == 4


def test_face_not_visible_is_poor():
    assert _score({**GOOD_CLASS, "face_visible_share": 0.1}).label == "Poor"


def test_rules_never_raise_a_label_and_skip_unmeasured_values():
    low = _score({**GOOD_CLASS, "homework_given": False, "recap_or_conclusion_in_last_10min": False}, dims=1.0)
    assert low.label == "Poor"                        # score already Poor; an "Average" cap can't raise it
    unmeasured = {k: None for k in GOOD_CLASS}
    assert _score(unmeasured).rules_applied == []      # n/a values never trigger a rule


def test_background_rule_needs_enough_camera_frames():
    few = {**GOOD_CLASS, "logo_in_camera_view_share": 0.0, "camera_view_share": 0.03}   # ~2 camera frames
    assert _score(few).label == "Excellent"
