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
from .metrics import METRIC_LABELS
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
SOURCE_LABEL = {"transcript": "Transcript", "metric": "Metric", "visual": "Video"}
STATUS_TEXT = {"used": "Used", "partial": "Partial", "missing": "Missing"}


def _compact(html: str) -> str:
    return "".join(line.strip() for line in html.splitlines())


# ---------------------------------------------------------------- progress

# Short, friendly lines per step. They rotate with a CSS animation, so they keep moving even while the
# server is waiting (e.g. on rate limits) and no new progress update arrives.
STEP_MESSAGES = {
    "Downloading": ["Fetching the recording…", "Almost there…", "Getting the video ready…"],
    "Validating file": ["Checking the recording…", "Making sure the audio is readable…", "Measuring its length…"],
    "Extracting audio": ["Separating the audio from the video…", "Preparing the sound for listening…",
                         "Cutting the class into ~10-minute parts…"],
    "Transcribing + separating speakers": ["Listening to the class…", "Turning speech into text…",
                                           "Telling the teacher's voice from the children's…",
                                           "Looking at video frames for the camera and 8x8 background…"],
    "Summarising the class": ["Reading the class section by section…", "Spotting questions and answers…",
                              "Noting praise, feedback and homework…", "Checking every quote against the transcript…"],
    "Computing metrics": ["Counting questions and answers…", "Measuring talk time and wait time…"],
    "Assessing against the rubric": ["Scoring rapport, engagement and questioning…",
                                     "Checking warmth, structure and setup…",
                                     "Making sure every observation has evidence…"],
}


# A drawn tick (not a text symbol), so no emoji-like characters appear anywhere in the UI.
CHECK_SVG = ('<svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.6 2.6L10 3.5" '
             'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def progress_html(current: str, fraction: float, is_link: bool, done: bool = False, detail: str = "") -> str:
    steps = [s for s in STEPS if is_link or s[0] != "Downloading"]
    names = [s[0] for s in steps]
    idx = names.index(current) if current in names else (len(steps) if done else 0)
    items = []
    for i, (_, label) in enumerate(steps):
        state = "done" if i < idx or done else "active" if i == idx else "todo"
        mark = CHECK_SVG if state == "done" else str(i + 1)
        items.append(f'<li class="cc-step {state}"><span class="cc-dot">{mark}</span><span>{escape(label)}</span></li>')
    overall = 1.0 if done else (idx + max(0.0, min(fraction, 1.0))) / len(steps)
    if done:
        live = '<div class="cc-live done"><span>All steps finished</span></div>'
        title = "Analysis complete"
    else:
        msgs = STEP_MESSAGES.get(current, ["Working…"])
        period = 3 * len(msgs)  # seconds per full cycle (3 s per message)
        spans = "".join(f'<span style="animation-name:cc-cycle{len(msgs)};animation-duration:{period}s;'
                        f'animation-delay:{3 * i}s">{escape(m)}</span>'
                        for i, m in enumerate(msgs))
        live = (f'<div class="cc-live"><span class="cc-spinner"></span><span class="cc-rotator">{spans}</span></div>'
                + (f'<div class="cc-detail">{escape(detail)}</div>' if detail else ""))
        title = f"Step {idx + 1} of {len(steps)}: {escape(current)}"
    return _compact(f"""
<div class="cc-card cc-progress">
<div class="cc-progress-head"><div class="cc-progress-title">{title}</div><div class="cc-pct">{overall:.0%}</div></div>
<ol class="cc-steps">{''.join(items)}</ol>
<div class="cc-bar"><div class="cc-bar-fill" style="width:{overall * 100:.0f}%"></div></div>
{live}
</div>""")


TIPS = [
    "Your recording is analysed privately and deleted as soon as the analysis finishes.",
    "Every observation is tied to a timestamp you can check in the video.",
    "Quiet stretches are often children solving puzzles, so silence is never counted against the tutor.",
    "Video frames are checked on this machine only and are never sent anywhere.",
    "Warmth is judged from what the tutor says, never from their voice or accent.",
]


def live_tips_html(palette: dict) -> str:
    """Typical duration and rotating tips, animated in the browser (keeps moving while the server waits)."""
    tips = json.dumps(TIPS)
    return f"""<!doctype html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
<style>
html,body{{margin:0;background:transparent;font-family:'Plus Jakarta Sans',system-ui,sans-serif;}}
.row{{color:{palette['muted']};font-size:13.5px;padding:2px 2px;}}
.tip{{transition:opacity .5s;}}
</style></head><body>
<div class="row">A 45–60 min class usually takes 3–4 minutes.</div>
<div class="row"><span class="tip" id="tip">Tip: {escape(TIPS[0])}</span></div>
<script>
const tips = {tips};
let i = 0;
setInterval(() => {{
  const el = document.getElementById('tip'); el.style.opacity = 0;
  setTimeout(() => {{ i = (i + 1) % tips.length; el.textContent = 'Tip: ' + tips[i]; el.style.opacity = 1; }}, 500);
}}, 7000);
</script></body></html>"""


# ---------------------------------------------------------------- results

def _confidence_word(c: float) -> str:
    return "High" if c >= 0.7 else "Medium" if c >= 0.5 else "Low"


def _chip(ts: str) -> str:
    return f'<span class="cc-chip">{escape(ts)}</span>'


def _evidence_list(items, empty: str = "") -> str:
    if not items:
        return f'<p class="cc-muted">{escape(empty)}</p>' if empty else ""
    lis = "".join(f'<li>{_chip(e.timestamp)}<span class="cc-src {escape(e.source)}">'
                  f'{SOURCE_LABEL.get(e.source, e.source)}</span><span>{escape(e.observation)}</span></li>'
                  for e in items)
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
{f'<details class="cc-reasons"><summary>Why This Confidence?</summary><ul>{reasons}</ul></details>' if reasons else ''}
</div>""")


def strengths_html(res: PipelineResult) -> str:
    a = res.assessment
    return _compact(f"""
<div class="cc-grid2">
<div class="cc-card"><h3 class="cc-h3">Strengths</h3>{_evidence_list(a.strengths, "No specific strengths were evidenced.")}</div>
<div class="cc-card"><h3 class="cc-h3">Areas To Improve</h3>{_evidence_list(a.improvements, "No specific improvements were evidenced.")}</div>
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
    return _compact('<h2 class="cc-h2">How The Class Scored</h2><div class="cc-dims">' + "".join(cards) + "</div>")


def signals_html(res: PipelineResult) -> str:
    rows = "".join(f'<li><span class="cc-pill {escape(s.status)}">{STATUS_TEXT.get(s.status, s.status)}</span>'
                   f'<div><div class="cc-sig-name">{escape(s.name)}</div>'
                   f'<div class="cc-muted cc-small">{escape(s.detail)}</div></div></li>' for s in res.signals)
    lims = "".join(f"<li>{escape(x)}</li>" for x in res.assessment.limitations + res.warnings)
    return _compact(f"""
<div class="cc-grid2">
<div class="cc-card"><h3 class="cc-h3">Signals Used / Missing</h3><ul class="cc-signals">{rows}</ul></div>
<div class="cc-card"><h3 class="cc-h3">Limitations</h3><ul class="cc-lims">{lims or '<li>None reported.</li>'}</ul></div>
</div>""")


PERCENT_METRICS = {"speech_ratio", "teacher_talk_share", "student_talk_share", "open_question_share",
                   "question_response_rate", "speaker_confidence", "transcription_confidence",
                   "low_confidence_segment_share", "camera_view_share", "face_visible_share", "no_face_share",
                   "logo_in_camera_view_share", "near_black_frame_share", "static_screen_share"}


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
