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
"""
from __future__ import annotations

import json
from functools import lru_cache

from .config import ROOT, get_config
from .schemas import Assessment, ScoreResult

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


def score(assessment: Assessment, rubric: dict | None = None, *,
          quality: list[tuple[float, str]] | None = None) -> ScoreResult:
    """`quality` is a list of (factor in 0..1, reason) applied to confidence, e.g. weak speaker separation."""
    cfg = get_config()["scoring"]
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
    return ScoreResult(label=label_for(weighted, cfg["thresholds"]), score=round(weighted, 2), confidence=conf,
                       low_confidence=low, weight_coverage=round(coverage, 2), missing_dimensions=missing,
                       confidence_reasons=reasons)
