"""HTML rendering for the progress tracker and the results page.

All text that came from the model or the transcript is passed through html.escape before it is
placed in markup. HTML is emitted without blank lines or indentation, because Streamlit's markdown
renderer ends an HTML block at a blank line and treats indented lines as code.
"""
from __future__ import annotations

import json
from html import escape

from .pipeline import PipelineResult, summarize
from .scoring import load_rubric
from .transcribe import LANGUAGE_NAMES

STEPS = [
    ("Downloading", "Download"),
    ("Validating file", "Validate"),
    ("Extracting audio", "Audio"),
    ("Transcribing + separating speakers", "Transcribe + speakers"),
    ("Summarising the class", "Summarise"),
    ("Computing metrics", "Metrics"),
    ("Assessing against the rubric", "Assess"),
]
LABEL_COLOUR = {"Excellent": "var(--accent)", "Good": "var(--accent-3)", "Average": "var(--accent-4)",
                "Poor": "var(--accent-2)"}
SOURCE_ICON = {"transcript": "💬", "metric": "📊", "visual": "🎥"}
STATUS_TEXT = {"used": "Used", "partial": "Partial", "missing": "Missing"}


def _compact(html: str) -> str:
    return "".join(line.strip() for line in html.splitlines())


# ---------------------------------------------------------------- progress

def progress_html(current: str, fraction: float, is_link: bool, done: bool = False) -> str:
    steps = [s for s in STEPS if is_link or s[0] != "Downloading"]
    names = [s[0] for s in steps]
    idx = names.index(current) if current in names else (len(steps) if done else 0)
    items = []
    for i, (_, label) in enumerate(steps):
        state = "done" if i < idx or done else "active" if i == idx else "todo"
        mark = "✓" if state == "done" else str(i + 1)
        items.append(f'<li class="cc-step {state}"><span class="cc-dot">{mark}</span><span>{escape(label)}</span></li>')
    overall = 1.0 if done else (idx + max(0.0, min(fraction, 1.0))) / len(steps)
    title = "Analysis complete" if done else f"{escape(current)}…"
    return _compact(f"""
<div class="cc-card cc-progress">
<div class="cc-progress-title">{title}</div>
<ol class="cc-steps">{''.join(items)}</ol>
<div class="cc-bar"><div class="cc-bar-fill" style="width:{overall * 100:.0f}%"></div></div>
</div>""")


# ---------------------------------------------------------------- results

def _confidence_word(c: float) -> str:
    return "High" if c >= 0.7 else "Medium" if c >= 0.5 else "Low"


def _chip(ts: str) -> str:
    return f'<span class="cc-chip">{escape(ts)}</span>'


def _evidence_list(items, empty: str = "") -> str:
    if not items:
        return f'<p class="cc-muted">{escape(empty)}</p>' if empty else ""
    lis = "".join(f'<li>{_chip(e.timestamp)}<span class="cc-src" title="{escape(e.source)}">'
                  f'{SOURCE_ICON.get(e.source, "")}</span>{escape(e.observation)}</li>' for e in items)
    return f'<ul class="cc-ev">{lis}</ul>'


def hero_html(res: PipelineResult) -> str:
    r, a = res.result, res.assessment
    s = summarize(res)
    colour = LABEL_COLOUR[r.label]
    conf_word = _confidence_word(r.confidence)
    lang = s["language"] or "unknown"
    reasons = "".join(f"<li>{escape(x)}</li>" for x in r.confidence_reasons)
    low = (f'<div class="cc-banner">Low confidence: treat this label as provisional. '
           f'{escape(" ".join(r.confidence_reasons))}</div>') if r.low_confidence else ""
    return _compact(f"""
<div class="cc-card cc-hero" style="--label:{colour}">
<div class="cc-hero-main">
<div class="cc-label-wrap"><div class="cc-kicker">Overall label</div><div class="cc-label">{escape(r.label)}</div></div>
<div class="cc-hero-stats">
<div class="cc-stat"><div class="cc-kicker">Score</div><div class="cc-stat-value">{r.score:.2f}<span>/5</span></div>
<div class="cc-scorebar"><div style="width:{r.score / 5 * 100:.0f}%"></div></div></div>
<div class="cc-stat"><div class="cc-kicker">Confidence</div><div class="cc-stat-value">{r.confidence:.0%}<span> {conf_word}</span></div>
<div class="cc-scorebar conf"><div style="width:{r.confidence * 100:.0f}%"></div></div></div>
</div>
</div>
<div class="cc-meta"><span>{s['duration_min']} min</span><span>{escape(lang)}</span><span>{s['events']} moments analysed</span>
<span>{r.weight_coverage:.0%} of rubric scored</span></div>
<p class="cc-summary">{escape(a.summary)}</p>
{low}
{f'<details class="cc-reasons"><summary>Why this confidence?</summary><ul>{reasons}</ul></details>' if reasons else ''}
</div>""")


def strengths_html(res: PipelineResult) -> str:
    a = res.assessment
    return _compact(f"""
<div class="cc-grid2">
<div class="cc-card"><h3 class="cc-h3">✨ What worked</h3>{_evidence_list(a.strengths, "No specific strengths were evidenced.")}</div>
<div class="cc-card"><h3 class="cc-h3">🎯 To improve</h3>{_evidence_list(a.improvements, "No specific improvements were evidenced.")}</div>
</div>""")


def dimensions_html(res: PipelineResult) -> str:
    rubric = {d["id"]: d for d in load_rubric()["dimensions"]}
    cards = []
    for d in res.assessment.dimensions:
        spec = rubric.get(d.dimension, {})
        name = spec.get("name", d.dimension)
        if d.score is None:
            score = '<span class="cc-pill missing">Not scored</span>'
            bar = ""
        else:
            score = f'<span class="cc-dim-score">{d.score:.1f}<span>/5</span></span>'
            bar = f'<div class="cc-scorebar"><div style="width:{d.score / 5 * 100:.0f}%"></div></div>'
        empty = "No supporting evidence was available for this dimension." if d.score is None else ""
        cards.append(f"""
<div class="cc-card cc-dim">
<div class="cc-dim-head"><div><div class="cc-dim-name">{escape(name)}</div>
<div class="cc-muted cc-small">weight {spec.get('weight', 0):.0%}</div></div>{score}</div>
{bar}
<p class="cc-rationale">{escape(d.rationale)}</p>
{_evidence_list(d.evidence, empty)}
</div>""")
    return _compact('<h2 class="cc-h2">Rubric breakdown</h2><div class="cc-dims">' + "".join(cards) + "</div>")


def signals_html(res: PipelineResult) -> str:
    rows = "".join(f'<li><span class="cc-pill {escape(s.status)}">{STATUS_TEXT.get(s.status, s.status)}</span>'
                   f'<div><div class="cc-sig-name">{escape(s.name)}</div>'
                   f'<div class="cc-muted cc-small">{escape(s.detail)}</div></div></li>' for s in res.signals)
    lims = "".join(f"<li>{escape(x)}</li>" for x in res.assessment.limitations + res.warnings)
    return _compact(f"""
<div class="cc-grid2">
<div class="cc-card"><h3 class="cc-h3">Signals used / missing</h3><ul class="cc-signals">{rows}</ul></div>
<div class="cc-card"><h3 class="cc-h3">Limitations</h3><ul class="cc-lims">{lims or '<li>None reported.</li>'}</ul></div>
</div>""")


METRIC_LABELS = {
    "speech_ratio": "Speech in recording", "long_silences": "Long silences (≥30 s)",
    "longest_silence_s": "Longest silence", "first_speech_s": "Time before first speech",
    "teacher_talk_share": "Teacher talk share", "student_talk_share": "Student talk share",
    "student_turns_per_10min": "Student turns per 10 min", "median_student_turn_s": "Median student turn",
    "speaker_confidence": "Speaker separation confidence", "questions_total": "Teacher questions",
    "questions_per_10min": "Questions per 10 min", "open_question_share": "Open questions share",
    "question_response_rate": "Questions answered", "median_wait_time_s": "Median wait after question",
    "students_addressed_individually": "Children addressed individually", "students_named_total": "Children named",
    "praise_count": "Praise moments", "greeting_or_rapport_in_first_10min": "Greeting / rapport early",
    "homework_given": "Homework given", "recap_or_conclusion_in_last_10min": "Recap / closing at the end",
    "technical_issues": "Technical issues", "unverifiable_events_dropped": "Unverifiable moments dropped",
    "language": "Language", "transcription_confidence": "Transcription clarity",
    "low_confidence_segment_share": "Low-confidence speech share",
}
PERCENT_METRICS = {"speech_ratio", "teacher_talk_share", "student_talk_share", "open_question_share",
                   "question_response_rate", "speaker_confidence", "transcription_confidence",
                   "low_confidence_segment_share"}


def _fmt_metric(m) -> str:
    v = m.value
    if v is None:
        return '<span class="cc-pill missing">Not available</span>'
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if m.name == "language":
        return escape(LANGUAGE_NAMES.get(v, str(v)))
    if m.name in PERCENT_METRICS and isinstance(v, (int, float)):
        return f"{v:.0%}"
    return escape(f"{v} {m.unit}".strip() if m.unit in ("s",) else str(v))


def metrics_html(res: PipelineResult) -> str:
    rows = "".join(
        f'<tr><td>{escape(METRIC_LABELS.get(m.name, m.name))}</td><td>{_fmt_metric(m)}</td>'
        f'<td><span class="cc-rel {m.reliability}">{m.reliability}</span></td>'
        f'<td class="cc-muted cc-small">{escape(m.note)}</td></tr>' for m in res.metrics)
    return _compact(f'<table class="cc-table"><thead><tr><th>Metric</th><th>Value</th><th>Reliability</th>'
                    f'<th>Note</th></tr></thead><tbody>{rows}</tbody></table>')


def report_json(res: PipelineResult) -> str:
    """Downloadable report: the assessment and metrics only. No transcript, quotes are pseudonymised."""
    return json.dumps({
        "label": res.result.label, "score": res.result.score, "confidence": res.result.confidence,
        "low_confidence": res.result.low_confidence, "confidence_reasons": res.result.confidence_reasons,
        "missing_dimensions": res.result.missing_dimensions, "summary": res.assessment.summary,
        "dimensions": [d.model_dump() for d in res.assessment.dimensions],
        "strengths": [e.model_dump() for e in res.assessment.strengths],
        "improvements": [e.model_dump() for e in res.assessment.improvements],
        "limitations": res.assessment.limitations,
        "metrics": [m.model_dump() for m in res.metrics],
        "signals": [s.model_dump() for s in res.signals],
    }, indent=2, ensure_ascii=False)
