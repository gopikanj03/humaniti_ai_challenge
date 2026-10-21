"""Shared theme logic for the feedback scripts: split notes into phrases, cluster them, name the clusters.

Each reviewer note is a comma-separated list of short phrases ("good rapport building, should give HWs").
Phrases are clustered on TF-IDF of normalised words; clusters are then matched against THEMES, a small
hand-written table that names each recurring theme and says how (or whether) it can be assessed from a
recording. Clusters that match no theme are reported as new candidates rather than silently dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words normalised so that "given HWs" and "should give homework" land together.
NORMALISE = {
    r"\bhws?\b": "homework", r"\bhomeworks\b": "homework", r"\bgiven\b": "give", r"\bgiving\b": "give",
    r"\bjoines\b": "joiners", r"\bjoiners?\b": "joiners", r"\bintroduction\b": "intro",
    r"\bindividually\b": "individual", r"\bone-to-one\b": "individual", r"\bcalling\b": "call",
    r"\basking\b": "ask", r"\basked\b": "ask", r"\bquestions\b": "question", r"\bkept\b": "keep",
    r"\b8\s*\*\s*8\b": "8x8", r"\bhappier\b": "happy", r"\binteractions?\b": "interaction",
    r"\bconsidering\b": "consider", r"\bconsideration\b": "consider", r"\bclass\b": "",
}
NEGATIVE = re.compile(r"^(need|needs|should|can |could|without|no |not |can't|cannot|directly|sometimes|some )|"
                      r"\b(without|can't|turned off|issues?)\b", re.I)


@dataclass
class Theme:
    id: str
    name: str
    pattern: str            # regex over normalised text (feedback phrases or generated observations)
    category: str           # MEASURABLE | JUDGEABLE | NOT_OBSERVABLE
    dimension: str | None   # rubric dimension it informs
    signal: str             # metric(s) or evidence used
    rationale: str = ""


THEMES = [
    Theme("branded_background", "8x8 branded background", r"background|8x8|logo|banner", "MEASURABLE",
          "professional_setup", "visual: 8x8 logo detected in sampled frames (stage 5)",
          "Visible on screen; a local frame check can detect the logo."),
    Theme("camera_face", "Camera on, face visible", r"camera|face|turned off|can't see", "MEASURABLE",
          "professional_setup", "visual: face-present ratio in sampled frames (stage 5)"),
    Theme("lighting", "Lighting / video clarity", r"\blight\b|lighting|clarity", "MEASURABLE",
          "professional_setup", "visual: frame brightness / sharpness (stage 5)",
          "'clarity issues while keeping the background' reads as video clarity, not explanation clarity."),
    Theme("homework", "Homework assigned", r"homework", "MEASURABLE", "structure_closure",
          "homework_given (event extracted from the transcript)"),
    Theme("individual_questions", "Asks children individually", r"individual|call|one to one",
          "MEASURABLE", "individual_engagement", "students_addressed_individually, question events",
          "Countable from named questions, but quality of the exchange still needs judgement."),
    Theme("include_everyone", "Considers every child", r"everyone|every one|each and every|consider",
          "MEASURABLE", "individual_engagement", "students_addressed_individually / students_named_total"),
    Theme("rapport_intro", "Intro and rapport building", r"rapport|intro\b|intro |good intro|revision",
          "JUDGEABLE", "opening_rapport", "events in the first ~10 minutes, read by the judge"),
    Theme("newcomer_intro", "Time for new joiners to introduce themselves", r"joiners|self-intro|self intro",
          "JUDGEABLE", "opening_rapport", "self_introduction events"),
    Theme("warmth_energy", "Pleasant, happy, energetic interaction", r"pleasant|happy|energetic|energy",
          "JUDGEABLE", "warmth_energy", "praise/encouragement events and wording, never voice pitch or volume",
          "Reviewers may hear tone of voice; we deliberately judge only what is said."),
    Theme("conclusion", "Clear conclusion", r"conclusion|closing|recap", "JUDGEABLE", "structure_closure",
          "recap_conclusion events in the last ~10 minutes"),
    Theme("interaction", "Interaction during class", r"interaction", "JUDGEABLE", "questioning_interaction",
          "question / response events"),
    Theme("teaching_aids", "Topic-related videos / aids", r"video|aids?\b", "JUDGEABLE", "structure_closure",
          "activity events"),
    Theme("prior_warnings", "Prior warnings about quality", r"informed|several times|warned", "NOT_OBSERVABLE",
          None, "none: refers to history outside the recording"),
]


def normalise(text: str) -> str:
    t = text.lower().strip(" .")
    for pat, rep in NORMALISE.items():
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Phrase:
    sample: str
    label: str
    text: str               # original phrase (tutor name already removed by load_feedback)
    norm: str
    negative: bool
    themes: list[str] = field(default_factory=list)


def split_phrases(key: str, label: str, note: str) -> list[Phrase]:
    parts = [p.strip() for p in re.split(r",|;|\.\s", note) if len(p.strip()) > 3]
    return [Phrase(key, label, p, normalise(p), bool(NEGATIVE.search(p.strip()))) for p in parts]


def themes_for(text: str) -> list[str]:
    norm = normalise(text)
    return [t.id for t in THEMES if re.search(t.pattern, norm)]


def cluster_phrases(phrases: list[Phrase], threshold: float = 0.8) -> list[list[Phrase]]:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer

    stop = ["good", "need", "more", "should", "can", "the", "a", "for", "of", "in", "with", "was", "were",
            "there", "and", "to", "be", "have", "while", "also", "better", "proper", "keep", "give"]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop).fit_transform([p.norm for p in phrases])
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                     linkage="average").fit_predict(vec.toarray())
    groups: dict[int, list[Phrase]] = {}
    for p, lab in zip(phrases, labels):
        groups.setdefault(int(lab), []).append(p)
    return sorted(groups.values(), key=len, reverse=True)
