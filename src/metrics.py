"""Computed signals. Each Metric records its source and reliability; unmeasurable values are None.

Deliberately NOT computed: pitch, loudness, accent, gender or speaking rate as quality signals.
Silence is reported neutrally: in a chess class, quiet stretches are often children solving puzzles.
"""
from __future__ import annotations

import math
import re
import statistics

from .audio import audio_metrics
from .schemas import ChunkDigest, Event, Metric, Segment, Transcript
from .speakers import SpeakerResult

METRIC_LABELS = {
    "speech_ratio": "Speech in recording", "long_silences": "Long silences (≥30 s)",
    "longest_silence_s": "Longest silence", "first_speech_s": "Time before first speech",
    "teacher_talk_share": "Teacher talk share", "student_talk_share": "Student talk share",
    "student_turns_per_10min": "Student turns per 10 min", "median_student_turn_s": "Median student turn",
    "speaker_confidence": "Speaker separation confidence", "questions_total": "Teacher questions",
    "questions_per_10min": "Questions per 10 min", "open_question_share": "Open questions share",
    "question_response_rate": "Questions answered", "median_wait_time_s": "Median wait after question",
    "students_addressed_individually": "Children addressed individually", "students_named_total": "Children named",
    "praise_count": "Praise moments", "name_calls_per_10min": "Children called by name (per 10 min)",
    "praise_per_10min": "Praise (per 10 min)", "greeting_or_rapport_in_first_10min": "Greeting / rapport early",
    "homework_given": "Homework given", "recap_or_conclusion_in_last_10min": "Recap / closing at the end",
    "technical_issues": "Technical issues", "unverifiable_events_dropped": "Unverifiable moments dropped",
    "language": "Language", "transcription_confidence": "Transcription clarity",
    "low_confidence_segment_share": "Low-confidence speech share",
    "frames_sampled": "Video frames sampled", "camera_view_share": "Tutor camera full-size",
    "face_visible_share": "Face visible (any size)", "no_face_share": "No face detected",
    "logo_in_camera_view_share": "8x8 background (camera frames)", "near_black_frame_share": "Black frames",
    "static_screen_share": "Screen unchanged between samples",
    "median_sharpness": "Video sharpness", "longest_frozen_s": "Longest frozen video", "video_height": "Video resolution",
}


def humanize(text: str) -> str:
    """Replace raw metric names (e.g. "recap_or_conclusion_in_last_10min = true") with plain words."""
    def name(match: re.Match) -> str:
        key = match.group(1)
        return METRIC_LABELS.get(key, key.replace("_", " ")).lower()
    text = re.sub(r"\b([a-z]+(?:_[a-z0-9]+)+)\b", name, text)
    text = re.sub(r"\s*\((?:visual )?metrics?\)", "", text)  # tags like "(metric)" mean nothing to readers
    text = re.sub(r"\s*=\s*(true|True)\b", ": yes", text)
    text = re.sub(r"\s*=\s*(false|False)\b", ": no", text)
    return re.sub(r"\s+=\s+", ": ", text)  # "name = value" only; leaves "2+2=4" alone


RESPONSE_WINDOW_S = 20.0    # a student turn starting within this after a question counts as a response


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


def speaker_metrics(segments: list[Segment], spk: SpeakerResult | None, duration: float) -> list[Metric]:
    if spk is None or spk.teacher_share is None:
        return [Metric(name="teacher_talk_share", value=None, source="speakers", reliability="low",
                       note="speaker separation unavailable")]
    rel = "high" if spk.confidence >= 0.7 else "medium" if spk.confidence >= 0.4 else "low"
    turns = _student_turns(segments)
    lengths = [b - a for a, b in turns]
    per10 = len(turns) / max(duration / 600, 1e-6)
    return [
        Metric(name="teacher_talk_share", value=_r(spk.teacher_share), source="speakers", reliability=rel,
               note="estimated share of speech time by the main (teacher) voice"),
        Metric(name="student_talk_share", value=_r(1 - spk.teacher_share), source="speakers", reliability=rel),
        Metric(name="student_turns_per_10min", value=_r(per10, 1), source="speakers", reliability=rel,
               note="runs of consecutive student segments in the transcript"),
        Metric(name="median_student_turn_s", value=_r(statistics.median(lengths) if lengths else None, 1),
               unit="s", source="speakers", reliability=rel),
        Metric(name="speaker_confidence", value=_r(spk.confidence), source="speakers", reliability=rel),
    ]


def _student_turns(segments: list[Segment]) -> list[tuple[float, float]]:
    turns: list[list[float]] = []
    prev_student = False
    for s in segments:
        is_student = s.speaker in ("student", "mixed")
        if is_student and prev_student and turns:
            turns[-1][1] = s.end
        elif is_student:
            turns.append([s.start, s.end])
        prev_student = is_student
    return [(a, b) for a, b in turns]


def event_metrics(digests: list[ChunkDigest], segments: list[Segment], duration: float) -> list[Metric]:
    events = [e for d in digests for e in d.events]
    if not digests:
        return [Metric(name="questions_total", value=None, source="events", reliability="low",
                       note="no transcript digests available")]
    by = lambda typ: [e for e in events if e.type == typ]  # noqa: E731
    questions = [e for e in by("question") if e.speaker != "student"]
    open_q = [e for e in questions if e.question_type == "open"]
    waits, answered = _wait_times(questions, segments)
    students_addressed = {e.student for e in events
                          if e.student and e.type in ("question", "individual_attention", "corrective_feedback",
                                                      "praise_encouragement")}
    students_heard = {e.student for e in events if e.student}
    first_10min = [e for e in events if e.t <= 600]
    last_10min = [e for e in events if e.t >= duration - 600]
    rel = "medium"  # extracted by a language model from an imperfect transcript
    m = [
        Metric(name="questions_total", value=len(questions), unit="count", source="events", reliability=rel),
        Metric(name="questions_per_10min", value=_r(len(questions) / max(duration / 600, 1e-6), 1),
               source="events", reliability=rel),
        Metric(name="open_question_share", value=_r(len(open_q) / len(questions)) if questions else None,
               source="events", reliability=rel),
        Metric(name="question_response_rate", value=_r(answered / len(questions)) if questions else None,
               source="speakers", reliability="low" if not questions else rel,
               note=f"student speech within {RESPONSE_WINDOW_S:.0f}s of a question"),
        Metric(name="median_wait_time_s", value=_r(statistics.median(waits), 1) if waits else None, unit="s",
               source="speakers", reliability=rel, note="gap between a question and the next student turn"),
        Metric(name="students_addressed_individually", value=len(students_addressed), unit="children",
               source="events", reliability=rel, note="distinct named children the teacher spoke to directly"),
        Metric(name="students_named_total", value=len(students_heard), unit="children", source="events",
               reliability=rel),
        Metric(name="praise_count", value=len(by("praise_encouragement")), unit="count", source="events",
               reliability=rel),
        Metric(name="greeting_or_rapport_in_first_10min",
               value=any(e.type in ("greeting_rapport", "self_introduction") for e in first_10min),
               source="events", reliability=rel),
        Metric(name="homework_given", value=bool(by("homework")), source="events", reliability=rel),
        Metric(name="recap_or_conclusion_in_last_10min",
               value=any(e.type in ("recap_conclusion", "homework") for e in last_10min),
               source="events", reliability=rel),
        Metric(name="technical_issues", value=len(by("technical_issue")), unit="count", source="events",
               reliability=rel),
        Metric(name="unverifiable_events_dropped", value=sum(d.dropped_events for d in digests), unit="count",
               source="events", note="events whose quote could not be found in the transcript"),
    ]
    return m


def _wait_times(questions: list[Event], segments: list[Segment]) -> tuple[list[float], int]:
    """For each question, time from the end of its segment to the next student/mixed segment."""
    waits, answered = [], 0
    for q in questions:
        seg = next((s for s in segments if s.start <= q.t + 0.5 and s.end >= q.t - 0.5), None)
        q_end = seg.end if seg else q.t
        nxt = next((s for s in segments if s.start >= q_end - 0.2 and s.speaker in ("student", "mixed")), None)
        if nxt and nxt.start - q_end <= RESPONSE_WINDOW_S:
            answered += 1
            waits.append(max(0.0, nxt.start - q_end))
    return waits, answered


def transcript_metrics(t: Transcript) -> list[Metric]:
    usable = t.usable_segments()
    probs = [math.exp(s.avg_logprob) for s in usable if s.avg_logprob is not None]
    flagged = sum(s.flagged for s in t.segments)
    clarity = statistics.mean(probs) if probs else None
    return [
        Metric(name="language", value=t.language, source="transcript"),
        Metric(name="transcription_confidence", value=_r(clarity), source="transcript",
               note="mean Whisper token probability; an audio-clarity proxy, not a teaching signal"),
        Metric(name="low_confidence_segment_share", value=_r(flagged / len(t.segments)) if t.segments else None,
               source="transcript"),
    ]


# Praise phrases in English plus a few common Hindi / Malayalam ones (transcripts can be mixed-language).
PRAISE_RE = re.compile(
    r"\b(well done|very good|good job|good work|good answer|excellent|great|awesome|brilliant|perfect|superb|"
    r"fantastic|amazing|super|nice (move|work|one|job|try)|that'?s (right|correct)|you got it|correct answer|"
    r"shabash|shabaash|bahut (accha|achha|badhiya)|badhiya|kollam|adipoli|nannayi|mitukk\w*)\b", re.I)


def whole_class_counts(segments: list[Segment], spk: SpeakerResult | None, names) -> dict:
    """Count, over the WHOLE transcript, the children the tutor calls by name and the tutor's praise.

    The event digests keep at most ~20 moments per section, so quick name-calls and praise drop out of them
    and were undercounted. Names come from the digests' name map (raw names stay in memory, only pseudonyms
    are reported); lines confidently attributed to children are skipped.
    """
    from .condense import _norm

    reliable = spk is not None and spk.teacher_share is not None and spk.confidence >= 0.4
    tutor = [x for x in segments if x.speaker != "student"] if reliable else list(segments)
    called: dict[str, int] = {}
    for raw, alias in (names.students.items() if names else []):
        parts = [raw] + [w for w in raw.split() if len(w) >= 3]
        pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in parts) + r")(?!\w)")
        hits = sum(1 for x in tutor if pattern.search(_norm(x.text)))
        if hits:
            called[alias] = hits
    praise = sum(1 for x in tutor if PRAISE_RE.search(x.text) and not x.text.strip().endswith("?"))
    return {"called": called, "praise": praise}


def compute_all(t: Transcript, spk: SpeakerResult | None, digests: list[ChunkDigest], names=None) -> list[Metric]:
    regions = spk.speech_regions if spk else []
    segs = t.usable_segments()
    metrics = (audio_metrics(regions, t.duration_s) + speaker_metrics(segs, spk, t.duration_s)
               + event_metrics(digests, segs, t.duration_s) + transcript_metrics(t))
    if not digests:
        return metrics
    whole = whole_class_counts(segs, spk, names)
    per10 = max(t.duration_s / 600, 1e-6)
    for m in metrics:
        # Whole-class counts replace digest counts when they find more (digests only sample the class).
        if m.name == "students_addressed_individually" and len(whole["called"]) > (m.value or 0):
            m.value, m.note = len(whole["called"]), "distinct children the tutor called by name (whole class)"
        if m.name == "praise_count" and whole["praise"] > (m.value or 0):
            m.value, m.note = whole["praise"], "tutor lines with praise across the whole class"
    metrics.append(Metric(name="name_calls_per_10min", value=_r(sum(whole["called"].values()) / per10, 1),
                          source="transcript", reliability="medium",
                          note="times the tutor said a child's name, per 10 minutes"))
    metrics.append(Metric(name="praise_per_10min", value=_r(whole["praise"] / per10, 1), source="transcript",
                          reliability="medium", note="tutor lines with praise, per 10 minutes"))
    return metrics
