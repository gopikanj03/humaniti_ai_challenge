"""Unit tests for the stage 4 dataset-script logic, on synthetic data (no recordings, feedback or API)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import calibrate  # noqa: E402
from themes import split_phrases, themes_for  # noqa: E402


def _run(rank, **dims):
    return {"rank": rank, "dims": dims}


def test_weighted_ignores_missing_dimensions():
    assert calibrate.weighted({"a": 4, "b": None}, {"a": 1, "b": 1}) == 4
    assert calibrate.weighted({"a": None}, {"a": 1}) is None


def test_thresholds_are_midpoints_and_monotonic():
    scores, ranks = [1.0, 1.4, 2.4, 2.6, 3.4, 3.6, 4.2, 4.4], [0, 0, 1, 1, 2, 2, 3, 3]
    th = calibrate.fit_thresholds(scores, ranks)
    assert th == {"Poor": 0.0, "Average": 1.85, "Good": 3.0, "Excellent": 3.9}
    assert [calibrate.predict(s, th) for s in scores] == ranks


def test_thresholds_handle_a_missing_label_and_inverted_means():
    th = calibrate.fit_thresholds([1.0, 3.0, 2.0], [0, 1, 3])   # no Good; Excellent scored below Average
    values = [th["Average"], th["Good"], th["Excellent"]]
    assert values == sorted(values)


def test_loo_never_uses_the_held_out_sample():
    # Correlation weights are refit per fold; a perfectly separating dimension should give exact LOO.
    runs = [_run(r, a=s, b=2.5) for r, s in [(0, 1.0), (0, 1.2), (1, 2.0), (1, 2.2), (2, 3.0), (2, 3.2),
                                             (3, 4.0), (3, 4.2)]]
    res = calibrate.evaluate(runs, "corr", lambda tr: calibrate.corr_weights(tr, ["a", "b"]))
    assert res["loo_within_one"] == 1.0 and res["rank_agreement"] > 0.9


def test_feedback_phrases_get_polarity_and_themes():
    ph = split_phrases("x/B1", "Good", "good rapport building, should give HWs, need more pleasant interaction")
    assert [p.negative for p in ph] == [False, True, True]
    assert "homework" in themes_for(ph[1].text) and "warmth_energy" in themes_for(ph[2].text)
    assert "branded_background" in themes_for("kept 8*8 background")
