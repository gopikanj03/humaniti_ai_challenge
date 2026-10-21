"""Pydantic models shared across pipeline stages."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MediaInfo(BaseModel):
    duration_s: float
    has_video: bool
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
