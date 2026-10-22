"""HTML rendering for the progress tracker and the results page.

All text that came from the model or the transcript is passed through html.escape before it is
placed in markup. HTML is emitted without blank lines or indentation, because Streamlit's markdown
renderer ends an HTML block at a blank line and treats indented lines as code.
"""
from __future__ import annotations

import json
from html import escape

from .pipeline import PipelineResult, summarize
from . import explain
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


def live_tips_html(palettes: dict) -> str:
    """Typical duration and rotating tips, animated in the browser (keeps moving while the server waits).
    The text colour follows the page's light/dark switch, which lives in the parent document."""
    tips = json.dumps(TIPS)
    muted = json.dumps({name: p["muted"] for name, p in palettes.items()})
    return f"""<!doctype html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
<style>
html,body{{margin:0;background:transparent;font-family:'Plus Jakarta Sans',system-ui,sans-serif;}}
.row{{font-size:13.5px;padding:2px 2px;}}
.tip{{transition:opacity .5s;}}
</style></head><body>
<div class="row">A 45–60 min class usually takes about 5–7 minutes.</div>
<div class="row"><span class="tip" id="tip">Tip: {escape(TIPS[0])}</span></div>
<script>
const tips = {tips}, muted = {muted};
let i = 0;
function colour() {{
  let t = 'light';
  try {{ t = window.parent.document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'; }} catch (e) {{}}
  document.body.style.color = muted[t];
}}
colour(); setInterval(colour, 400);
setInterval(() => {{
  const el = document.getElementById('tip'); el.style.opacity = 0;
  setTimeout(() => {{ i = (i + 1) % tips.length; el.textContent = 'Tip: ' + tips[i]; el.style.opacity = 1; }}, 500);
}}, 7000);
</script></body></html>"""


# ---------------------------------------------------------------- results

def _confidence_word(c: float) -> str:
    return "High" if c >= 0.7 else "Medium" if c >= 0.5 else "Low"


def _chip(ts: str) -> str:
    return f'<span class="cc-chip">{escape(explain.display_time(ts))}</span>'


def _evidence_list(items, empty: str = "") -> str:
    if not items:
        return f'<p class="cc-muted">{escape(empty)}</p>' if empty else ""
    lis = "".join(f'<li>{_chip(e.timestamp)}<span class="cc-src {escape(e.source)}">'
                  f'{explain.EVIDENCE_SOURCE.get(e.source, e.source)}</span><span>{escape(e.observation)}</span></li>'
                  for e in items)
    return f'<ul class="cc-ev">{lis}</ul>'


def _scale_html(r) -> str:
    bands = explain.scale(r)
    # The highlighted band and the marker always show the FINAL result (never the score-alone label).
    segs = "".join(f'<div class="cc-band{" on" if name == r.label else ""}" style="flex:{hi - lo:.2f}">'
                   f'<span>{escape(name)}</span></div>' for name, lo, hi in bands)
    pos = min(max(explain.bar_position(r) / 5, 0), 1) * 100
    return (f'<div class="cc-scale"><div class="cc-bands">{segs}</div>'
            f'<div class="cc-marker" style="left:{pos:.1f}%"></div></div>')


def hero_html(res: PipelineResult) -> str:
    r, a = res.result, res.assessment
    s = summarize(res)
    rubric = load_rubric()
    colour = LABEL_COLOUR[r.label]
    sure = explain.certainty(r.confidence)
    lang = s["language"] or "unknown"
    badge = ('<div class="cc-qi">Not a class recording</div>' if r.not_a_class else
             '<div class="cc-qi">Recording quality issue</div>' if r.quality_issue else "")
    why = "".join(f"<li>{escape(x)}</li>" for x in explain.main_reasons(r, a, rubric))
    sure_why = "".join(f"<li>{escape(explain.plain_reason(x))}</li>" for x in r.confidence_reasons)
    lowered = (f'<p class="cc-muted cc-small">Required standards that were not met are listed first.</p>'
               if explain.lowered(r) else "")
    return _compact(f"""
<div class="cc-card cc-hero" style="--label:{colour}">
<div class="cc-hero-main">
<div class="cc-label-wrap"><div class="cc-kicker">Overall result</div><div class="cc-label">{escape(r.label)}</div>{badge}</div>
<div class="cc-meaning"><div class="cc-kicker">What this means</div><p>{escape(explain.label_meaning(r))}</p></div>
</div>
<div class="cc-scorebox"><div class="cc-kicker">Where this class falls</div>{_scale_html(r)}
<p class="cc-muted cc-small cc-scoreline">{escape(explain.score_line(r))}</p></div>
<div class="cc-sure"><span class="cc-kicker">How sure is this result?</span>&nbsp;<b>{sure}.</b>&nbsp;
<span class="cc-muted">{escape(explain.CERTAINTY_MEANING[sure])}</span>
{f'<details class="cc-reasons"><summary>Why?</summary><ul>{sure_why}</ul></details>' if sure_why else ''}</div>
<h3 class="cc-h3 cc-sub">In Short</h3>
<p class="cc-summary">{escape(a.summary)}</p>
{f'<h3 class="cc-h3 cc-sub">Why This Result</h3>{lowered}<ul class="cc-why">{why}</ul>' if why else ''}
<div class="cc-meta"><span>{s['duration_min']} min class</span><span>{escape(lang)}</span>
<span>Times like 12:05 mean minutes:seconds into the recording</span></div>
</div>""")


def strengths_html(res: PipelineResult) -> str:
    a = res.assessment
    return _compact(f"""
<div class="cc-grid2">
<div class="cc-card"><h3 class="cc-h3">What Went Well</h3>{_evidence_list(a.strengths, "No specific strengths were evidenced.")}</div>
<div class="cc-card"><h3 class="cc-h3">What To Improve</h3>{_evidence_list(a.improvements, "No specific improvements were evidenced.")}</div>
</div>""")


def dimensions_html(res: PipelineResult) -> str:
    rubric = {d["id"]: d for d in load_rubric()["dimensions"]}
    cards = []
    for d in res.assessment.dimensions:
        spec = rubric.get(d.dimension, {})
        word = explain.rating(d.score)
        number = "" if d.score is None else f'<span class="cc-muted cc-small">{d.score:.1f} / 5</span>'
        examples = (f'<details class="cc-examples"><summary>Show examples ({len(d.evidence)})</summary>'
                    f'{_evidence_list(d.evidence)}</details>' if d.evidence else "")
        cards.append(f"""
<div class="cc-card cc-dim">
<div class="cc-dim-head"><div class="cc-dim-name">{escape(spec.get('name', d.dimension))}</div>
<span class="cc-rating {word.lower().replace(' ', '-')}">{word}</span></div>
<div class="cc-question">{escape(explain.AREA_QUESTION.get(d.dimension, ''))} {number}</div>
<p class="cc-rationale">{escape(d.rationale or 'This area could not be checked from this recording.')}</p>
{examples}
</div>""")
    return _compact('<h2 class="cc-h2">Area By Area</h2><div class="cc-dims">' + "".join(cards) + "</div>")


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
