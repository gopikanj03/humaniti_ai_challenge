"""Plain-language explanations of a result, for readers with no technical background.

Used by both the results page and the PDF so they say the same thing. Numbers become words ("Strong",
"Needs work", "Fairly sure"), each area is described as a simple question, and technical reasons are rewritten
as everyday sentences.
"""
from __future__ import annotations

import re

from .config import get_config
from .schemas import Assessment, ScoreResult

LABEL_MEANING = {
    "Excellent": "A very strong class: the tutor did nearly everything an 8x8 class should have, and did it well.",
    "Good": "A solid class: most important things were done well, with a few things to improve.",
    "Average": "The class met the basic standard, but several important things were missing or weak.",
    "Poor": "The class fell well short of the expected standard and needs clear improvement.",
}
QUALITY_ISSUE_MEANING = ("The recording itself (picture or sound) is too poor to judge the class fairly, so it is "
                         "marked Poor as a recording quality issue.")

AREA_QUESTION = {
    "teaching_accuracy": "Is what the tutor teaches correct?",
    "opening_rapport": "Does the class start in a warm, friendly way?",
    "individual_engagement": "Does the tutor involve every child, calling on them by name?",
    "questioning_interaction": "Does the tutor ask questions that get the children thinking and answering?",
    "warmth_energy": "Is the tutor encouraging and positive with the children?",
    "structure_closure": "Is the lesson well organised, with a recap at the end and homework?",
    "professional_setup": "Is the tutor's camera on, with the 8x8 background?",
}

RATINGS = [(4.0, "Strong"), (3.0, "Fine"), (2.0, "Needs work"), (0.0, "Weak")]

CERTAINTY_MEANING = {
    "Fairly sure": "The evidence was clear enough to rely on this result.",
    "Somewhat sure": "The result is likely right, but some of the evidence was unclear.",
    "Not very sure": "Treat this as a first look: a person should check the recording before acting on it.",
}

EVIDENCE_SOURCE = {"transcript": "Said in class", "metric": "Measured", "visual": "Seen on video"}


def display_time(ts: str) -> str:
    """Evidence about the class as a whole has no single moment: show "Overall" instead of a time."""
    return "Overall" if ts.strip().lower() in ("whole class", "whole-class", "overall") else ts


def rating(score: float | None) -> str:
    if score is None:
        return "Not checked"
    return next(word for limit, word in RATINGS if score >= limit)


def certainty(confidence: float) -> str:
    return "Fairly sure" if confidence >= 0.7 else "Somewhat sure" if confidence >= 0.5 else "Not very sure"


def label_meaning(result: ScoreResult) -> str:
    return QUALITY_ISSUE_MEANING if result.quality_issue else LABEL_MEANING[result.label]


def plain_reason(text: str) -> str:
    """Rewrite a technical confidence reason as an everyday sentence."""
    t = text.lower()
    if "voice separation" in t or "voices could not be separated" in t:
        return "It was hard to tell the tutor's voice from the children's, so figures about who spoke are rough."
    m = re.search(r"class language is (\w+)", text, re.I)
    if m:
        return f"The class was in {m.group(1)}; the automatic understanding of it is less precise than for English."
    if "clarity is low" in t:
        return "Parts of the audio were hard to make out."
    if "could not be verified" in t:
        return "Some moments could not be checked against what was actually said."
    if t.startswith("not scored"):
        return "Some areas could not be checked from this recording."
    if "no rubric dimension" in t:
        return "Nothing in the recording could be assessed."
    return text


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return m.group(1) if m else text


def main_reasons(result: ScoreResult, assessment: Assessment, rubric: dict, limit: int = 4) -> list[str]:
    """The few things that most explain the label, in plain words."""
    if result.quality_issue:
        return result.rules_applied[:limit]
    names = {d["id"]: d["name"] for d in rubric["dimensions"]}
    reasons = list(result.rules_applied)
    scored = [d for d in assessment.dimensions if d.score is not None]
    for d in sorted(scored, key=lambda d: d.score):
        if d.score < 3 and len(reasons) < limit:
            reasons.append(f"{names.get(d.dimension, d.dimension)} needs work: {_first_sentence(d.rationale)}")
    best = max(scored, key=lambda d: d.score, default=None)
    if best is not None and best.score >= 4 and len(reasons) < limit:
        reasons.append(f"{names.get(best.dimension, best.dimension)} was a strength: {_first_sentence(best.rationale)}")
    return reasons[:limit]


def scale(result: ScoreResult) -> list[tuple[str, float, float]]:
    """(label, from, to) bands of the 0-5 score, from the configured thresholds."""
    th = get_config()["scoring"]["thresholds"]
    order = ["Poor", "Average", "Good", "Excellent"]
    bounds = [th[x] for x in order] + [5.0]
    return [(order[i], bounds[i], bounds[i + 1]) for i in range(4)]
