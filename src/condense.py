"""Per-chunk digests: a small fast model turns each ~10 min of transcript into timestamped events.

The judge never sees the raw transcript, only these digests. To keep them honest:
  * every quote must be found (verbatim, or near-verbatim) in the transcript, else the event is dropped;
  * timestamps are snapped to the transcript line that contains the quote;
  * children's names are replaced by stable pseudonyms ("Student 1", ...) and the teacher's by
    "the teacher" before anything leaves this module.
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from .config import get_config
from .llm import chat_json, estimate_tokens, prompt_budget
from .schemas import EVENT_TYPES, ChunkDigest, Event, Segment, Transcript

log = logging.getLogger(__name__)
PROMPT_VERSION = "condense-v2"
SPEAKER_TAG = {"teacher": "T", "student": "S", "mixed": "M", None: "?"}

SYSTEM = f"""You extract evidence from part of a recorded online chess class for children (company: 8x8).
The transcript may be in any language (often Malayalam or Hindi mixed with English) and contains
speech-recognition errors. Each line is "[mm:ss] TAG: text". TAG is an automatic estimate of who is
speaking: T = probably the teacher, S = probably a child, M = mixed, ? = unknown. Tags can be wrong;
use the content too.

Return JSON:
{{"summary": "2-3 factual English sentences on what happened in this part",
  "teacher_name": "teacher's name if they say it, else null",
  "events": [{{"t": "mm:ss", "type": one of {list(EVENT_TYPES)},
              "speaker": "teacher" | "student" | "unclear",
              "detail": "factual English description, max 15 words",
              "quote": "words copied exactly from the line at t (original language, max 12 words), or null",
              "question_type": "open" | "closed" | null,
              "student_name": "the child's name as heard, if a specific child is addressed or speaks, else null"}}]}}

Rules:
- Only report what is in the transcript. Never invent events, quotes, names or timestamps.
- "t" must be the timestamp of a line shown. "quote" must be copied character-for-character from that line.
- Prefer events that show teaching quality: greetings and rapport, introductions (including new
  children), explanations, questions to the class or to a named child, children's answers, praise,
  corrections, puzzles/activities, homework, recap and closing, technical problems.
- question_type: "open" invites reasoning ("why", "how", "what would you do"); "closed" asks for a
  yes/no or a single fact. Only for type "question".
- Describe what was said, not how it sounded: do not comment on voice, accent, pitch, loudness or speed.
- At most 20 events, most informative first. Keep the JSON compact. If the part is silent or unclear, return few or no events and say so in the summary."""


# Small models drift from the requested vocabulary; map common variants instead of failing the chunk.
TYPE_SYNONYMS = {
    "greeting": "greeting_rapport", "rapport": "greeting_rapport", "introduction": "self_introduction",
    "intro": "self_introduction", "teacher_question": "question", "student_answer": "student_response",
    "answer": "student_response", "response": "student_response", "student_question": "student_response",
    "praise": "praise_encouragement", "encouragement": "praise_encouragement", "feedback": "corrective_feedback",
    "correction": "corrective_feedback", "puzzle": "activity", "game": "activity", "practice": "activity",
    "recap": "recap_conclusion", "conclusion": "recap_conclusion", "closing": "recap_conclusion",
    "summary": "recap_conclusion", "assignment": "homework", "technical": "technical_issue",
    "instruction": "explanation", "teaching": "explanation",
}


class _RawEvent(BaseModel):
    t: str
    type: str                      # validated against EVENT_TYPES after normalisation; unknown -> dropped
    speaker: str = "unclear"
    detail: str = ""
    quote: str | None = None
    question_type: str | None = None
    student_name: str | None = None

    @field_validator("t", mode="before")
    @classmethod
    def _t(cls, v):
        return fmt_ts(float(v)) if isinstance(v, (int, float)) else str(v)

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        v = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in EVENT_TYPES else TYPE_SYNONYMS.get(v, v)

    @field_validator("speaker", mode="before")
    @classmethod
    def _speaker(cls, v):
        v = str(v or "").lower()
        return "teacher" if "teach" in v or "tutor" in v else "student" if "stud" in v or "child" in v else "unclear"

    @field_validator("question_type", mode="before")
    @classmethod
    def _qt(cls, v):
        v = str(v or "").lower()
        return v if v in ("open", "closed") else None

    @field_validator("quote", "student_name", mode="before")
    @classmethod
    def _optional_text(cls, v):
        v = None if v is None else str(v).strip()
        return v if v and v.lower() not in ("null", "none", "n/a") else None


class _RawDigest(BaseModel):
    summary: str
    teacher_name: str | None = None
    events: list[_RawEvent] = Field(default_factory=list)


# ---------------------------------------------------------------- helpers

def fmt_ts(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_ts(text: str) -> float | None:
    parts = text.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def _norm(text: str) -> str:
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.casefold())
    return " ".join(text.split())


def quote_matches(quote: str, text: str, min_ratio: float = 0.85) -> bool:
    """True if `quote` appears in `text` verbatim or nearly so (tolerates small transcription noise)."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    m = difflib.SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size / len(q) >= min_ratio


def chunk_groups(transcript: Transcript) -> list[list[Segment]]:
    groups: dict[int, list[Segment]] = {}
    for s in transcript.usable_segments():
        groups.setdefault(s.chunk_index, []).append(s)
    return [groups[k] for k in sorted(groups)]


def format_chunk(segments: list[Segment]) -> str:
    return "\n".join(f"[{fmt_ts(s.start)}] {SPEAKER_TAG.get(s.speaker, '?')}: {s.text}" for s in segments)


def _user_message(segments: list[Segment]) -> str:
    return (f"Part of the class from {fmt_ts(segments[0].start)} to {fmt_ts(segments[-1].end)}:\n\n"
            f"{format_chunk(segments)}")


def fit_to_budget(groups: list[list[Segment]], budget_tokens: int) -> list[list[Segment]]:
    """Split groups whose prompt would exceed the token budget into consecutive halves (recursively)."""
    out: list[list[Segment]] = []
    for g in groups:
        if len(g) > 1 and estimate_tokens(_user_message(g)) > budget_tokens:
            mid = len(g) // 2
            out += fit_to_budget([g[:mid], g[mid:]], budget_tokens)
        else:
            out.append(g)
    return out


# ---------------------------------------------------------------- verification + pseudonyms

class NameMap:
    """Maps raw child names to stable pseudonyms, in order of first appearance."""

    def __init__(self) -> None:
        self.students: dict[str, str] = {}
        self.teacher_names: set[str] = set()

    @staticmethod
    def _key(name: str) -> str:
        return _norm(name)

    def student(self, name: str | None) -> str | None:
        if not name or not self._key(name) or self._key(name) in self.teacher_names:
            return None
        key = self._key(name)
        if key not in self.students:
            self.students[key] = f"Student {len(self.students) + 1}"
        return self.students[key]

    def scrub(self, text: str | None) -> str | None:
        """Replace every known name (and each part of multi-word names) in free text."""
        if not text:
            return text
        replacements = [(n, "the teacher") for n in self.teacher_names]
        replacements += list(self.students.items())
        # Longest names first so "Anu Mary" is replaced before "Anu".
        for raw, alias in sorted(replacements, key=lambda kv: -len(kv[0])):
            for part in [raw] + [p for p in raw.split() if len(p) >= 3]:
                text = re.sub(rf"(?<!\w){re.escape(part)}(?!\w)", alias, text, flags=re.IGNORECASE)
        return text


def verify_events(raw: _RawDigest, segments: list[Segment]) -> tuple[list[_RawEvent | None], list[float]]:
    """Return per-event snapped timestamps, or None for events that must be dropped."""
    start, end = segments[0].start, segments[-1].end
    kept, times = [], []
    for ev in raw.events:
        t = parse_ts(ev.t)
        if t is None or ev.type not in EVENT_TYPES or not (start - 5 <= t <= end + 5):
            kept.append(None); times.append(0.0)  # noqa: E702
            continue
        if ev.quote:
            # Look near the claimed time first, then anywhere in the chunk (models often misplace by a line or two).
            near = sorted(segments, key=lambda s: abs(s.start - t))
            hit = next((s for s in near[:6] if quote_matches(ev.quote, s.text)), None) or \
                next((s for s in segments if quote_matches(ev.quote, s.text)), None)
            if hit is None:
                kept.append(None); times.append(0.0)  # noqa: E702
                continue
            t = hit.start
        kept.append(ev)
        times.append(t)
    return kept, times


# ---------------------------------------------------------------- entry point

def condense(transcript: Transcript, file_hash: str,
             on_progress: Callable[[int, int], None] | None = None) -> tuple[list[ChunkDigest], NameMap]:
    lcfg = get_config()["llm"]
    max_tokens = lcfg["condense_max_tokens"]
    groups = fit_to_budget(chunk_groups(transcript), prompt_budget(max_tokens, SYSTEM))
    # Each Groq model has its own per-minute allowance, so sections alternate between "lanes" (models) and
    # run in parallel. A lane whose model is at its cap falls back to a model not already used as a lane.
    lanes = [lcfg["condense_model"]] + [m for m in lcfg.get("condense_parallel_models", [])
                                        if m != lcfg["condense_model"]]
    fallbacks = lcfg.get("condense_fallback_models", [])

    def chain(i: int) -> list[str]:
        primary = lanes[i % len(lanes)]
        order = [primary] + [m for m in fallbacks if m not in lanes] + [m for m in lanes + fallbacks if m != primary]
        return list(dict.fromkeys(order))

    def run(i: int, segments: list[Segment]) -> _RawDigest:
        return chat_json(chain(i), SYSTEM, _user_message(segments), _RawDigest, max_tokens=max_tokens,
                         reasoning_effort=lcfg.get("condense_reasoning_effort"), file_hash=file_hash,
                         cache_tag=PROMPT_VERSION)

    # Progress is reported from this (the caller's) thread: UIs such as Streamlit can't be updated
    # from worker threads.
    with ThreadPoolExecutor(max_workers=max(2, len(lanes))) as pool:
        futures = [pool.submit(run, i, g) for i, g in enumerate(groups)]
        for n, _ in enumerate(as_completed(futures), start=1):
            if on_progress:
                on_progress(n, len(groups))
        raws = [f.result() for f in futures]

    # ... but names are mapped sequentially, so pseudonyms are numbered in class order on every run.
    names = NameMap()
    for raw in raws:
        if raw.teacher_name:
            names.teacher_names.add(NameMap._key(raw.teacher_name))
    for raw in raws:
        for ev in raw.events:
            names.student(ev.student_name)

    digests = []
    for i, (raw, segments) in enumerate(zip(raws, groups)):
        kept, times = verify_events(raw, segments)
        events = [
            Event(t=t, type=ev.type, speaker=ev.speaker, detail=names.scrub(ev.detail) or "",
                  quote=names.scrub(ev.quote), question_type=ev.question_type if ev.type == "question" else None,
                  student=names.student(ev.student_name), verified=bool(ev.quote))
            for ev, t in zip(kept, times) if ev is not None
        ]
        dropped = sum(ev is None for ev in kept)
        digests.append(ChunkDigest(chunk_index=i, start=segments[0].start, end=segments[-1].end,
                                   summary=names.scrub(raw.summary) or "", events=sorted(events, key=lambda e: e.t),
                                   dropped_events=dropped))
    n_events = sum(len(d.events) for d in digests)
    n_dropped = sum(d.dropped_events for d in digests)
    log.info("condense: %d chunks, %d events kept, %d dropped as unverifiable, %d students named",
             len(digests), n_events, n_dropped, len(names.students))
    return digests, names
