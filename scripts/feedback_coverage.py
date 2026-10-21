"""Does ClassCritique's output cover what the human reviewer said, and is its evidence real?

Coverage: themes named in the human note (themes.py) that also appear in the generated assessment.
Visual themes are reported separately while the visual stage is off; not-observable themes are excluded.
Evidence precision: each transcript-sourced evidence item must point at a transcript line within 20 s,
and any quoted words must actually appear in the transcript near that time.

    python scripts/feedback_coverage.py
"""
from __future__ import annotations

import json
import re

from dataset import DATA, HOLDOUT, load_feedback
from runs_data import load_runs
from src.condense import parse_ts, quote_matches
from themes import THEMES, themes_for

VISUAL = {t.id for t in THEMES if t.dimension == "professional_setup"}
NOT_OBS = {t.id for t in THEMES if t.category == "NOT_OBSERVABLE"}
NAMES = {t.id: t.name for t in THEMES}


def transcript_for(file_hash12: str) -> list[tuple[float, str]]:
    """Union of cached transcript chunks (auto + language-forced versions) as (global start, text)."""
    segs = []
    for d in (DATA / "cache").glob(f"{file_hash12}*"):
        for f in d.glob("transcript_*.json"):
            start = float(re.search(r"_(\d+)-\d+", f.stem).group(1))
            segs += [(start + s["start"], s["text"]) for s in json.loads(f.read_text(encoding="utf-8"))]
    return sorted(segs)


def main() -> None:
    feedback = load_feedback()
    totals = {"covered": 0, "observable": 0, "grounded": 0, "evidence": 0, "quotes_ok": 0, "quotes": 0}
    for r in load_runs(include_holdout=True):
        fb = feedback[r["key"]]
        human = set(themes_for(fb.text)) - NOT_OBS
        a = r["assessment"]
        items = a["strengths"] + a["improvements"] + [e for d in a["dimensions"] for e in d["evidence"]]
        generated_text = " ".join([a["summary"]] + [d["rationale"] for d in a["dimensions"]] +
                                  [e["observation"] for e in items])
        generated = set(themes_for(generated_text))
        observable = human - VISUAL
        covered = observable & generated
        totals["covered"] += len(covered); totals["observable"] += len(observable)

        segs = transcript_for(r["summary"]["file_hash"])
        grounded = quotes = quotes_ok = n = 0
        for e in items:
            t = parse_ts(e["timestamp"].split("-")[0].strip()) if e["source"] == "transcript" else None
            if t is None:
                continue
            n += 1
            near = [txt for st, txt in segs if abs(st - t) <= 20]
            grounded += bool(near)
            for q in re.findall(r'"([^"]{8,})"', e["observation"]):
                quotes += 1
                quotes_ok += any(quote_matches(q, txt, 0.8) for txt in near) or \
                    any(quote_matches(q, txt, 0.8) for st, txt in segs if abs(st - t) <= 60)
        for k, v in (("grounded", grounded), ("evidence", n), ("quotes_ok", quotes_ok), ("quotes", quotes)):
            totals[k] += v
        tag = " (held out)" if r["key"] in HOLDOUT else ""
        missed = [NAMES[t] for t in observable - generated]
        print(f"{r['key']:20s}{tag:11s} coverage {len(covered)}/{len(observable)}"
              f"{'  missed: ' + ', '.join(missed) if missed else ''}"
              f"{'  | visual themes not assessed: ' + str(len(human & VISUAL)) if human & VISUAL else ''}")
        print(f"{'':31s} evidence grounded {grounded}/{n}, quotes found {quotes_ok}/{quotes}")
    if totals["observable"]:
        print(f"\nOverall: theme coverage {totals['covered']}/{totals['observable']} "
              f"({totals['covered'] / totals['observable']:.0%}); evidence grounded "
              f"{totals['grounded']}/{totals['evidence']}; quotes verified {totals['quotes_ok']}/{totals['quotes']}")
        print("Note: quotes are pseudonymised (Student N) and transcripts can be noisy, so a few true quotes may "
              "fail the fuzzy match; failures are worth a manual look, not automatic proof of fabrication.")


if __name__ == "__main__":
    main()
