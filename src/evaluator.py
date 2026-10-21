"""One judge call: rubric + metrics + chunk digests (+ visual notes, anchors, style) -> Assessment.

After the call, evidence is checked against what the judge was actually given:
  * transcript evidence must point within EVIDENCE_TOLERANCE_S of a digest event;
  * visual evidence is only allowed when visual observations were provided;
  * a dimension whose evidence is all rejected loses its score (it becomes "missing"),
    and dimensions that need an unavailable signal are forced to null.
"""
from __future__ import annotations

import json
import logging

from .condense import fmt_ts, parse_ts
from .config import ROOT, get_config
from .llm import chat_json, estimate_tokens, prompt_budget
from .schemas import Assessment, ChunkDigest, DimensionScore, Event, EvidenceItem, Metric
from .scoring import load_rubric

log = logging.getLogger(__name__)
PROMPT_VERSION = "judge-v2"
EVIDENCE_TOLERANCE_S = 20.0

DEFAULT_STYLE = """Write like an experienced 8x8 class reviewer: short, specific, practical. Each observation
names a concrete moment ("At 12:40 the teacher asked Student 3 to find the checkmate and waited for the
answer") rather than a generality ("good interaction"). Praise what worked, then give actionable
improvements ("should give homework", "give new joiners time to introduce themselves")."""

SYSTEM = """You assess the teaching quality of a recorded online chess class for children (8x8).
You receive a rubric, computed metrics, and timestamped event digests extracted from the transcript.
You do not have the raw transcript or audio.

Rules:
1. Evidence only. Every evidence item must cite a timestamp that appears in the digests (or say
   "whole class" for metric-based evidence). Never invent moments, quotes or numbers.
2. If a dimension cannot be judged from what you were given, set "score": null and explain in
   "limitations". Signals marked NOT AVAILABLE are missing, not negative.
3. Speaker labels and the metrics derived from them are automatic estimates; do not treat them as fact
   when their reliability is low.
4. Do not infer teaching quality from voice pitch, loudness, accent, gender or speaking speed.
   Silence is often productive (children solving puzzles). A quiet child is not necessarily disengaged.
5. Refer to children only by the pseudonyms used in the digests (Student 1, Student 2, ...).
   If the digests note that events were omitted for length, a missing kind of event is NOT evidence that
   it did not happen; the metric counts (e.g. praise_count, questions_total) cover the whole class.
   Never turn a gap in the data into a criticism of the teacher.
6. Scores use the rubric scale (0-5, decimals allowed). Be calibrated: 3 = adequate, not "good".

Return JSON:
{"summary": "2-3 sentence overall assessment",
 "dimensions": [{"dimension": "<rubric id>", "score": 0-5 or null, "rationale": "1-2 sentences",
                 "evidence": [{"timestamp": "mm:ss" | "mm:ss-mm:ss" | "whole class",
                               "observation": "specific, factual", "source": "transcript" | "metric" | "visual"}]}],
 "strengths": [<evidence items>], "improvements": [<evidence items>],
 "limitations": ["what could not be assessed and why"],
 "confidence": 0-1}
Include every rubric dimension exactly once."""


def _load_optional(name: str) -> str | None:
    """Gitignored artefacts from the dataset scripts (anchors, style). Absent until stage 4 runs."""
    path = ROOT / "config" / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def render_metrics(metrics: list[Metric]) -> str:
    lines = []
    for m in metrics:
        value = "NOT AVAILABLE" if m.value is None else m.value
        note = f" ({m.note})" if m.note else ""
        lines.append(f"- {m.name}: {value} {m.unit}".rstrip() + f" [source={m.source}, reliability={m.reliability}]{note}")
    return "\n".join(lines)


# When the digests don't fit the token budget, keep the events most tied to the rubric first.
# Within a priority level, types take turns (round-robin), so a frequent type such as student_response
# can't crowd out a rarer one that a rubric dimension depends on (e.g. praise for warmth).
EVENT_PRIORITY = {
    "individual_attention": 0, "question": 0, "homework": 0, "recap_conclusion": 0, "greeting_rapport": 0,
    "self_introduction": 0, "praise_encouragement": 0, "corrective_feedback": 0, "technical_issue": 0,
    "student_response": 1, "off_task": 1, "activity": 1, "explanation": 2,
}


def _event_line(e: Event, max_quote_words: int | None = None) -> str:
    who = e.speaker + (f" -> {e.student}" if e.student and e.speaker == "teacher" else
                       f" ({e.student})" if e.student else "")
    qt = f"/{e.question_type}" if e.question_type else ""
    quote = e.quote
    if quote and max_quote_words is not None:
        words = quote.split()
        quote = " ".join(words[:max_quote_words]) + (" ..." if len(words) > max_quote_words else "")
    return f"[{fmt_ts(e.t)}] {e.type}{qt}, {who}: {e.detail}" + (f' | "{quote}"' if quote else "")


def render_digests(digests: list[ChunkDigest], budget_tokens: int | None = None) -> tuple[str, int]:
    """Render digests in time order. If over budget, drop the lowest-priority events first.
    Returns (text, number of events omitted)."""
    def render(keep: set[int] | None, max_quote_words: int | None) -> str:
        out, idx = [], 0
        for d in digests:
            out.append(f"## {fmt_ts(d.start)}-{fmt_ts(d.end)}\nSummary: {d.summary}")
            for e in d.events:
                if keep is None or idx in keep:
                    out.append(_event_line(e, max_quote_words))
                idx += 1
        return "\n".join(out)

    full = render(None, None)
    if budget_tokens is None or estimate_tokens(full) <= budget_tokens:
        return full, 0
    events = [(i, e) for i, e in enumerate(e for d in digests for e in d.events)]
    # Position of each event within its own type (verified first, then spread through the class in time order).
    turn: dict[int, int] = {}
    by_type: dict[str, list[tuple[int, Event]]] = {}
    for i, e in events:
        by_type.setdefault(e.type, []).append((i, e))
    for items in by_type.values():
        for n, (i, _) in enumerate(sorted(items, key=lambda ie: (not ie[1].verified, ie[1].t))):
            turn[i] = n
    ranked = sorted(events, key=lambda ie: (EVENT_PRIORITY.get(ie[1].type, 2), turn[ie[0]], ie[1].t))
    # Fixed cost of the chunk summaries, then add events by priority while they fit.
    used = estimate_tokens(render(set(), 12))
    keep: set[int] = set()
    for i, e in ranked:
        cost = estimate_tokens(_event_line(e, 12)) + 1
        if used + cost > budget_tokens:
            continue
        keep.add(i)
        used += cost
    return render(keep, 12), len(events) - len(keep)


def build_prompt(digests: list[ChunkDigest], metrics: list[Metric], visual: list[str] | None,
                 meta: dict, budget_tokens: int | None = None) -> str:
    rubric = load_rubric()
    # The judge sees what each dimension means, not its weight: weighting is Python's job.
    dims = [{k: d[k] for k in ("id", "name", "description", "evidence_required", "requires") if k in d}
            for d in rubric["dimensions"]]
    parts = [
        "# Rubric\n" + json.dumps({"scale": rubric["scale"], "dimensions": dims}, ensure_ascii=False),
        "# Class\n" + "\n".join(f"- {k}: {v}" for k, v in meta.items()),
        "# Metrics\n" + render_metrics(metrics),
        "# Visual observations\n" + ("\n".join(f"- {v}" for v in visual) if visual else
                                     "NOT AVAILABLE (video frames were not analysed in this run)"),
        "# Writing style\n" + (_load_optional("style.json") or DEFAULT_STYLE),
    ]
    anchors = _load_optional("anchors.json")
    if anchors:
        parts.append("# Reference classes (one per label, paraphrased; for calibration only)\n" + anchors)
    head = "\n\n".join(parts)
    remaining = None if budget_tokens is None else budget_tokens - estimate_tokens(head) - 30
    text, omitted = render_digests(digests, remaining)
    note = f"\n({omitted} lower-priority events omitted to fit the length limit.)" if omitted else ""
    return head + "\n\n# Event digests\n" + (text or "NOT AVAILABLE (no usable speech)") + note


def _ts_ok(ts: str, event_times: list[float]) -> bool:
    first = ts.split("-")[0].strip()
    t = parse_ts(first)
    return t is not None and any(abs(t - et) <= EVIDENCE_TOLERANCE_S for et in event_times)


def _filter(items: list[EvidenceItem], event_times: list[float], has_visual: bool) -> tuple[list[EvidenceItem], int]:
    kept = []
    for it in items:
        if it.source == "visual":
            ok = has_visual
        elif it.source == "metric":
            ok = True  # metric evidence refers to the whole class; the numbers came from us
        else:
            ok = _ts_ok(it.timestamp, event_times)
        if ok:
            kept.append(it)
    return kept, len(items) - len(kept)


def verify(assessment: Assessment, digests: list[ChunkDigest], has_visual: bool) -> tuple[Assessment, int]:
    rubric = load_rubric()
    event_times = [e.t for d in digests for e in d.events]
    by_id = {d.dimension: d for d in assessment.dimensions}
    dims, dropped = [], 0
    for spec in rubric["dimensions"]:
        d = by_id.get(spec["id"]) or DimensionScore(dimension=spec["id"], rationale="Not returned by the judge.")
        evidence, n = _filter(d.evidence, event_times, has_visual)
        dropped += n
        score = d.score
        if spec.get("requires") == "visual" and not has_visual:
            score, evidence = None, []
        elif score is not None and not evidence:
            score = None  # a score with no surviving evidence is not supported
        dims.append(d.model_copy(update={"score": score, "evidence": evidence}))
    strengths, n1 = _filter(assessment.strengths, event_times, has_visual)
    improvements, n2 = _filter(assessment.improvements, event_times, has_visual)
    limitations = list(assessment.limitations)
    if dropped + n1 + n2:
        limitations.append(f"{dropped + n1 + n2} evidence item(s) were removed because they did not match any "
                           "moment in the class digests.")
    return assessment.model_copy(update={"dimensions": dims, "strengths": strengths, "improvements": improvements,
                                         "limitations": limitations}), dropped + n1 + n2


def evaluate(digests: list[ChunkDigest], metrics: list[Metric], meta: dict, file_hash: str,
             visual: list[str] | None = None) -> tuple[Assessment, int]:
    lcfg = get_config()["llm"]
    max_tokens = lcfg["judge_max_tokens"]
    user = build_prompt(digests, metrics, visual, meta, budget_tokens=prompt_budget(max_tokens, SYSTEM))
    raw = chat_json(lcfg["judge_model"], SYSTEM, user, Assessment, max_tokens=max_tokens,
                    reasoning_effort=lcfg.get("judge_reasoning_effort"), file_hash=file_hash,
                    cache_tag=PROMPT_VERSION)
    assessment, dropped = verify(raw, digests, has_visual=bool(visual))
    log.info("judge: %d dimensions scored, %d evidence items dropped",
             sum(d.score is not None for d in assessment.dimensions), dropped)
    return assessment, dropped
