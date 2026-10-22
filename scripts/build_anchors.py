"""Build one compact reference summary per label (metrics + paraphrased feedback gist) for the judge.

Writes config/anchors.json (gitignored), which evaluator.py adds to the judge prompt as few-shot anchors.
Held-out recordings are excluded. Tutor names are removed before the notes go to the model, which is
asked to paraphrase (not quote) and to describe what distinguishes each level. No transcript is used.
Note the leakage this implies: a training recording's own note contributes to its label's anchor, so
in-sample agreement is optimistic; only the held-out check is clean.

    python scripts/build_anchors.py
"""
from __future__ import annotations

import json
import statistics

from pydantic import BaseModel

from dataset import HOLDOUT, ROOT, load_feedback, scrub_names
from runs_data import LABELS, load_runs
from src.config import get_config
from src.llm import chat_json

KEY_METRICS = ["questions_per_10min", "open_question_share", "students_addressed_individually",
               "praise_count", "homework_given", "greeting_or_rapport_in_first_10min",
               "recap_or_conclusion_in_last_10min"]

SYSTEM = """You summarise how human reviewers at 8x8 (online chess classes for children) describe classes at
each quality level. For each label, write a 2-sentence gist of what characterises that level and what
separates it from the level below, in your own words. Do not quote the notes, do not include names.
Return JSON: {"Excellent": "...", "Good": "...", "Average": "...", "Poor": "..."}"""


class Gists(BaseModel):
    Excellent: str
    Good: str
    Average: str
    Poor: str


def main() -> None:
    feedback = {k: f for k, f in load_feedback().items() if k not in HOLDOUT}
    notes = "\n".join(f"[{f.label}] {scrub_names(f.text, f.tutor_tokens)}"
                      for f in sorted(feedback.values(), key=lambda f: LABELS.index(f.label), reverse=True))
    lcfg = get_config()["llm"]
    gists = chat_json([lcfg["judge_model"]] + lcfg.get("judge_fallback_models", []), SYSTEM, notes, Gists,
                      max_tokens=1500, reasoning_effort="low")

    runs = load_runs()
    anchors = {}
    for lab in reversed(LABELS):
        rs = [r for r in runs if r["human_label"] == lab]
        metrics = {}
        for m in KEY_METRICS:
            vals = [float(r["metric_values"][m]) for r in rs if isinstance(r["metric_values"].get(m), (int, float))]
            if vals:
                metrics[m] = round(statistics.mean(vals), 2)
        anchors[lab] = {"gist": getattr(gists, lab), "typical_metrics": metrics, "based_on_recordings": len(rs)}
    out = ROOT / "config" / "anchors.json"
    out.write_text(json.dumps(anchors, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(anchors, indent=1, ensure_ascii=False))
    print(f"\nwrote {out} (gitignored). The judge now uses these anchors; re-run the judge step for comparable results.")


if __name__ == "__main__":
    main()
