"""Shared helpers for the offline dataset scripts: list samples and parse Zoom VTTs.

Layout (agreed with the dataset owner):
    data/<label folder>/<batch> ... Part - YYYY-MM-DD.mp4   + matching *-transcript*.vtt
    data/feedbacks.pdf                                       one feedback line per labelled class
Folders map to labels: excellent, good, average -> same; weak and "quality issue" -> Poor;
demo -> unlabelled. Samples are keyed by folder + batch, because batch B435 appears twice
(two different classes and tutors).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Windows consoles default to cp1252; model output can contain characters like non-breaking hyphens.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = ROOT / "data"

FOLDER_LABELS = {"excellent": "Excellent", "good": "Good", "average": "Average",
                 "weak": "Poor", "quality issue": "Poor", "demo": None}


@dataclass
class Sample:
    key: str            # e.g. "good/B552"
    folder: str
    label: str | None   # None for unlabelled demo classes
    video: Path
    vtt: Path | None
    batch: str | None
    date: str | None    # from the filename, which is more reliable than the PDF


def list_samples() -> list[Sample]:
    samples = []
    for folder, label in FOLDER_LABELS.items():
        d = DATA / folder
        if not d.is_dir():
            continue
        vtts = list(d.glob("*.vtt"))
        for video in sorted(d.glob("*.mp4")):
            batch_m = re.match(r"(B\d+)", video.name, re.I)
            batch = batch_m.group(1).upper() if batch_m else None
            date_m = re.search(r"(\d{4}-\d{2}-\d{2})", video.name)
            if batch:
                vtt = next((v for v in vtts if v.name.upper().startswith(batch)), None)
                key = f"{folder}/{batch}"
            else:  # demo files: match on the language word in the name
                word = video.stem.lower().split()[-1]
                vtt = next((v for v in vtts if word in v.name.lower()), None)
                key = f"{folder}/{word}"
            samples.append(Sample(key, folder, label, video, vtt, batch, date_m.group(1) if date_m else None))
    return samples


@dataclass
class Cue:
    start: float
    end: float
    speaker: str
    text: str


def _ts(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(path: Path) -> list[Cue]:
    txt = path.read_text(encoding="utf-8-sig")
    cues = []
    for a, b, body in re.findall(r"(\d+:\d\d:\d\d\.\d+) --> (\d+:\d\d:\d\d\.\d+)[^\n]*\n(.+?)(?:\n\s*\n|\Z)", txt, re.S):
        body = " ".join(body.strip().splitlines())
        m = re.match(r"([^:]{1,60}):\s*(.*)", body)
        speaker, text = (m.group(1).strip(), m.group(2)) if m else ("", body)
        cues.append(Cue(_ts(a), _ts(b), speaker, text))
    return cues


def vtt_teacher(cues: list[Cue]) -> str:
    """The speaker with the most talk time (the tutor in 10/10 labelled classes)."""
    talk: dict[str, float] = {}
    for c in cues:
        talk[c.speaker] = talk.get(c.speaker, 0.0) + c.end - c.start
    return max(talk, key=talk.get)


# ---------------------------------------------------------------- feedback PDF

HEADING_FOLDER = {"excellent": "excellent", "good": "good", "average": "average", "weak": "weak",
                  "known quality issues": "quality issue"}


@dataclass
class Feedback:
    key: str            # folder/batch, matching Sample.key
    label: str
    text: str           # the reviewer's note with the tutor's name removed
    tutor_tokens: list[str]  # name parts, kept only in memory to scrub generated text; never printed


def load_feedback(pdf: Path = DATA / "feedbacks.pdf") -> dict[str, Feedback]:
    from pypdf import PdfReader

    text = " ".join(page.extract_text() for page in PdfReader(str(pdf)).pages)
    text = re.sub(r"\s+", " ", text)
    out: dict[str, Feedback] = {}
    # Split into "*Heading -*" sections, then "N. Bxxx (date) NAME ---- note" entries.
    for heading, body in re.findall(r"\*\s*([A-Za-z ]+?)\s*-\s*\*(.*?)(?=\*\s*[A-Za-z ]+?\s*-\s*\*|$)", text):
        folder = HEADING_FOLDER[heading.strip().lower()]
        for batch, name, note in re.findall(r"\d+\.\s*(B\d+)\s*\([^)]*\)\s*(.*?)\s*-{3,}\s*(.*?)(?=\s\d+\.\s*B\d+|$)", body):
            key = f"{folder}/{batch.upper()}"
            out[key] = Feedback(key=key, label=FOLDER_LABELS[folder], text=note.strip(),
                                tutor_tokens=[t for t in re.split(r"\W+", name) if len(t) >= 3])
    return out


def scrub_names(text: str, tokens: list[str]) -> str:
    for t in sorted(tokens, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(t)}(?!\w)", "the tutor", text, flags=re.IGNORECASE)
    return text


# Held out from every tuning step (calibrate.py); chosen before any scores were seen, one from the upper
# and one from the lower half of the label scale. Only used for the final held-out check.
HOLDOUT = ["good/B435", "weak/B558"]
RUNS = DATA / "runs"


def run_path(key: str) -> Path:
    return RUNS / (key.replace("/", "__").replace(" ", "_") + ".json")
