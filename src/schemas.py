"""Pydantic models shared across pipeline stages."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MediaInfo(BaseModel):
    duration_s: float
    has_video: bool
    width: int | None = None
    height: int | None = None
    has_audio: bool
    size_mb: float
    container: str


class AudioChunk(BaseModel):
    index: int
    start_s: float          # global offset of this chunk in the recording
    end_s: float
    path: str               # temp file; deleted after the run
    size_mb: float


class Segment(BaseModel):
    start: float            # global seconds
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    language: str | None = None   # as detected by Whisper for this segment's chunk
    speaker: str | None = None    # "teacher" | "student" | "mixed" | None (estimated, see speakers.py)
    teacher_fraction: float | None = None
    chunk_index: int = 0
    # True when Whisper confidence suggests a hallucination or unclear audio.
    # Flagged segments are kept for transparency but must not be cited as evidence.
    flagged: bool = False


class Transcript(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    duration_s: float
    provider: str
    model: str
    language: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def usable_segments(self) -> list[Segment]:
        return [s for s in self.segments if not s.flagged and s.text.strip()]


# ---------------------------------------------------------------- condense (per-chunk events)

EVENT_TYPES = (
    "greeting_rapport",       # greeting, small talk, warm-up before content
    "self_introduction",      # teacher or children introducing themselves (incl. new joiners)
    "explanation",            # teacher explains a concept, move or rule
    "question",               # teacher asks a question
    "student_response",       # a child answers or contributes
    "individual_attention",   # teacher addresses / checks on a specific child
    "praise_encouragement",
    "corrective_feedback",    # teacher corrects a mistake or guides a child to the answer
    "activity",               # puzzle, game, practice, quiz, video
    "homework",               # homework / practice assigned
    "recap_conclusion",       # summary of the lesson, closing
    "technical_issue",        # audio/video/connection problems
    "off_task",               # distraction, long unrelated talk
)
EventType = Literal[EVENT_TYPES]  # type: ignore[valid-type]


class Event(BaseModel):
    t: float                                   # global seconds
    type: EventType
    speaker: Literal["teacher", "student", "unclear"] = "unclear"
    detail: str                                # short English description
    quote: str | None = None                   # verbatim, original language
    question_type: Literal["open", "closed"] | None = None
    student: str | None = None                 # pseudonym ("Student 3") after name mapping
    verified: bool = False                     # quote found in the transcript near t


class ChunkDigest(BaseModel):
    chunk_index: int
    start: float
    end: float
    summary: str
    events: list[Event] = Field(default_factory=list)
    dropped_events: int = 0                    # events removed because they couldn't be verified


# ---------------------------------------------------------------- metrics

class Metric(BaseModel):
    name: str
    value: float | int | bool | str | None     # None = could not be measured
    unit: str = ""
    source: Literal["audio", "transcript", "speakers", "events", "visual"]
    reliability: Literal["high", "medium", "low"] = "high"
    note: str = ""


# ---------------------------------------------------------------- evaluator output

class EvidenceItem(BaseModel):
    timestamp: str                             # "mm:ss" or "mm:ss-mm:ss"
    observation: str
    source: Literal["transcript", "metric", "visual"] = "transcript"


class DimensionScore(BaseModel):
    dimension: str
    score: float | None = None                 # 0-5, None = insufficient evidence
    rationale: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)

    @field_validator("score")
    @classmethod
    def _range(cls, v: float | None) -> float | None:
        if v is not None and not 0 <= v <= 5:
            raise ValueError("score must be between 0 and 5, or null")
        return v


class Assessment(BaseModel):
    summary: str
    dimensions: list[DimensionScore]
    strengths: list[EvidenceItem] = Field(default_factory=list)
    improvements: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    teaching_errors: list[EvidenceItem] = Field(default_factory=list)  # factual mistakes the tutor taught
    is_class_recording: bool = True            # False: not a lesson at all (film, vlog, music, gameplay, noise...)
    not_class_reason: str = ""
    confidence: float = Field(ge=0, le=1)


class ScoreResult(BaseModel):
    label: Literal["Poor", "Average", "Good", "Excellent"]
    score: float                               # weighted 0-5
    confidence: float                          # 0-1
    low_confidence: bool
    weight_coverage: float                     # share of rubric weight that had a score
    missing_dimensions: list[str] = Field(default_factory=list)
    confidence_reasons: list[str] = Field(default_factory=list)
    score_label: str | None = None             # label from the weighted score alone, before rules
    rules_applied: list[str] = Field(default_factory=list)   # strict rules that lowered the label, with reasons
    quality_issue: bool = False                # recording's technical quality is too poor to assess fairly
    not_a_class: bool = False                  # the recording does not appear to be a class at all
