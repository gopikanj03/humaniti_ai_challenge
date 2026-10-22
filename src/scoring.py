"""Weighted score -> label, computed in Python (the LLM only scores dimensions).

Missing-dimension policy
------------------------
A dimension is "missing" when the judge returned null or all of its evidence failed verification.
* The weighted score is the mean over SCORED dimensions, with their weights renormalised to sum to 1.
* Confidence is multiplied by the share of rubric weight that was scored ("weight coverage"), so a
  label resting on fewer dimensions is reported as less certain.
* If weight coverage < scoring.min_weight_coverage, or confidence < scoring.low_confidence_below,
  the result is flagged low-confidence and the UI says so prominently.
* If nothing at all could be scored (e.g. muted or unusable recording), the label is
  scoring.no_evidence_label with confidence 0. An unusable class recording is itself a quality issue
  in 8x8's own labelling ("quality issue" -> Poor), but the zero confidence makes clear that no
  teaching was assessed.
Exactly one label is always returned.

Strict rules (see config `rules`)
---------------------------------
After the weighted score, deterministic rules can only LOWER the label:
* recording-quality rules (resolution, blur, frozen or black video, unclear audio, too little speech)
  -> Poor, flagged as a recording quality issue;
* teaching rules (teaching mistakes, no homework and no recap, face not visible, no 8x8 background,
  too few questions, too little praise) -> at most Average, or Poor for the severe ones; breaking several
  "at most Average" rules -> Poor.
Every rule that fires is reported with its reason. Rules never fire on values that could not be measured.
"""
from __future__ import annotations

import json
from functools import lru_cache

from .config import ROOT, get_config
from .schemas import Assessment, Metric, ScoreResult

LABELS = ("Poor", "Average", "Good", "Excellent")


@lru_cache(maxsize=1)
def load_rubric() -> dict:
    with open(ROOT / get_config()["scoring"]["rubric"], encoding="utf-8") as f:
        return json.load(f)


def label_for(score: float, thresholds: dict[str, float]) -> str:
    label = LABELS[0]
    for name in LABELS:
        if score >= thresholds[name]:
            label = name
    return label


def apply_rules(values: dict, teaching_errors: int) -> tuple[str | None, list[str], bool]:
    """Return (label cap or None, reasons, recording_quality_issue) from measured values."""
    cfg = get_config().get("rules", {})
    if not cfg.get("enabled", False):
        return None, [], False
    q, t = cfg["quality"], cfg["teaching"]
    v = lambda name: values.get(name) if isinstance(values.get(name), (int, float)) else None  # noqa: E731

    checks = [
        (v("video_height") is not None and v("video_height") < q["min_video_height"],
         f"video resolution is only {v('video_height')}p"),
        (v("median_sharpness") is not None and v("median_sharpness") < q["min_median_sharpness"], "video is blurry"),
        (v("longest_frozen_s") is not None and v("longest_frozen_s") >= q["max_frozen_s"],
         f"video was frozen for about {round((v('longest_frozen_s') or 0) / 60)} min"),
        (v("near_black_frame_share") is not None and v("near_black_frame_share") > q["max_black_share"],
         f"video was black in {v('near_black_frame_share'):.0%} of sampled frames" if v("near_black_frame_share") is not None else ""),
        (v("transcription_confidence") is not None and v("transcription_confidence") < q["min_transcription_clarity"],
         "audio is unclear (speech hard to recognise)"),
        (v("low_confidence_segment_share") is not None and v("low_confidence_segment_share") > q["max_unclear_speech_share"],
         "much of the speech is unintelligible"),
        (v("speech_ratio") is not None and v("speech_ratio") < q["min_speech_ratio"],
         "very little audible speech in the recording"),
    ]
    quality = [reason for hit, reason in checks if hit]
    if quality:
        return "Poor", [f"Recording quality issue: {r}." for r in quality], True

    poor, average = [], []
    if teaching_errors >= t["errors_poor"]:
        poor.append(f"{teaching_errors} teaching mistakes were found.")
    elif teaching_errors >= t["errors_cap_average"]:
        average.append(f"{teaching_errors} teaching mistake{'s' if teaching_errors > 1 else ''} found.")
    if v("face_visible_share") is not None and v("face_visible_share") < t["min_face_visible_share"]:
        poor.append(f"The tutor's face was visible in only {v('face_visible_share'):.0%} of sampled frames.")
    if values.get("homework_given") is False and values.get("recap_or_conclusion_in_last_10min") is False:
        average.append("No homework and no recap or closing.")
    cam = round((v("camera_view_share") or 0) * (v("frames_sampled") or 0))
    if cam >= t["min_checked_camera_frames"] and v("logo_in_camera_view_share") == 0:
        average.append(f"No 8x8 background in any of the {cam} full-size camera frames.")
    if v("questions_per_10min") is not None and v("questions_per_10min") < t["min_questions_per_10min"]:
        average.append(f"Only {v('questions_per_10min')} questions per 10 minutes.")
    if v("praise_count") is not None and v("praise_count") < t["min_praise"]:
        average.append(f"Only {int(v('praise_count'))} moment(s) of praise.")
    if len(average) >= t["average_caps_to_poor"]:
        poor.append(f"{len(average)} required standards were not met.")
    if poor:
        return "Poor", poor + average, False
    if average:
        return "Average", average, False
    return None, [], False


def score(assessment: Assessment, rubric: dict | None = None, *,
          quality: list[tuple[float, str]] | None = None, metrics: list[Metric] | None = None) -> ScoreResult:
    """`quality` is a list of (factor in 0..1, reason) applied to confidence, e.g. weak speaker separation."""
    cfg = get_config()["scoring"]
    if not assessment.is_class_recording:
        reason = assessment.not_class_reason.strip() or "it does not contain a lesson."
        return ScoreResult(label="Poor", score=0.0, confidence=round(assessment.confidence, 2),
                           low_confidence=assessment.confidence < cfg["low_confidence_below"],
                           weight_coverage=0.0, not_a_class=True, score_label="Poor",
                           rules_applied=[f"This does not appear to be a class recording: {reason}"])
    rubric = rubric or load_rubric()
    weights = {d["id"]: float(d["weight"]) for d in rubric["dimensions"]}
    total_w = sum(weights.values())
    scored = {d.dimension: d.score for d in assessment.dimensions if d.dimension in weights and d.score is not None}
    missing = [dim for dim in weights if dim not in scored]
    coverage = sum(weights[d] for d in scored) / total_w if total_w else 0.0
    reasons: list[str] = []

    if not scored:
        return ScoreResult(label=cfg.get("no_evidence_label", "Poor"), score=0.0, confidence=0.0,
                           low_confidence=True, weight_coverage=0.0, missing_dimensions=missing,
                           confidence_reasons=["No rubric dimension could be scored from this recording."])

    weighted = sum(weights[d] * s for d, s in scored.items()) / sum(weights[d] for d in scored)
    conf = assessment.confidence * coverage
    if missing:
        reasons.append(f"Not scored: {', '.join(missing)} ({coverage:.0%} of rubric weight covered).")
    for factor, reason in quality or []:
        if factor < 1.0:
            conf *= max(0.0, factor)
            reasons.append(reason)

    conf = round(min(max(conf, 0.0), 1.0), 2)
    low = conf < cfg["low_confidence_below"] or coverage < cfg["min_weight_coverage"]
    score_label = label_for(weighted, cfg["thresholds"])
    cap, rules, quality_issue = apply_rules({m.name: m.value for m in metrics or []}, len(assessment.teaching_errors))
    label = score_label
    if cap is not None and LABELS.index(cap) < LABELS.index(score_label):
        label = cap
    return ScoreResult(label=label, score=round(weighted, 2), confidence=conf, low_confidence=low,
                       weight_coverage=round(coverage, 2), missing_dimensions=missing, confidence_reasons=reasons,
                       score_label=score_label, rules_applied=rules,
                       quality_issue=quality_issue)
