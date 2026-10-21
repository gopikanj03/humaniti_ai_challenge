"""Flag samples whose feedback tone or content seems to contradict their label.

Tone = share of praise phrases minus share of criticism phrases in the note. Labels should broadly
order by tone. Flags are for the dataset owner to review; nothing is excluded or down-weighted until
they confirm (calibrate.py reads data/reports/consistency_decisions.json, which only they create).

    python scripts/check_consistency.py
"""
from __future__ import annotations

import json
import statistics

from dataset import DATA, load_feedback
from themes import THEMES, split_phrases, themes_for

RANK = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
NOT_OBSERVABLE = {t.id for t in THEMES if t.category == "NOT_OBSERVABLE"}


def main() -> None:
    rows = []
    for f in load_feedback().values():
        ph = split_phrases(f.key, f.label, f.text)
        pos, neg = sum(not p.negative for p in ph), sum(p.negative for p in ph)
        unobservable = [t for p in ph for t in themes_for(p.text) if t in NOT_OBSERVABLE]
        rows.append({"key": f.key, "label": f.label, "praise": pos, "criticism": neg,
                     "tone": round((pos - neg) / max(len(ph), 1), 2), "unobservable": unobservable})

    mean_tone = {lab: statistics.mean(r["tone"] for r in rows if r["label"] == lab) for lab in RANK}
    print("Mean tone by label:", {k: round(v, 2) for k, v in mean_tone.items()}, "\n")
    flags = []
    for r in sorted(rows, key=lambda r: (-RANK[r["label"]], -r["tone"])):
        reasons = []
        # Tone closer to a label at least two levels away from its own.
        nearest = min(RANK, key=lambda lab: abs(mean_tone[lab] - r["tone"]))
        if abs(RANK[nearest] - RANK[r["label"]]) >= 2:
            reasons.append(f"tone {r['tone']:+.2f} looks like '{nearest}', not '{r['label']}'")
        if r["label"] in ("Excellent", "Good") and r["criticism"] > r["praise"]:
            reasons.append("more criticism than praise for a high label")
        if r["label"] == "Poor" and r["praise"] > r["criticism"]:
            reasons.append("more praise than criticism for a Poor label")
        if r["unobservable"]:
            reasons.append("label partly rests on things a recording can't show (" + ", ".join(r["unobservable"]) + ")")
        mark = "FLAG" if reasons else "ok  "
        print(f"{mark} {r['key']:20s} {r['label']:10s} +{r['praise']} -{r['criticism']} tone {r['tone']:+.2f}"
              + (f"  <- {'; '.join(reasons)}" if reasons else ""))
        if reasons:
            flags.append({**r, "reasons": reasons})
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    (DATA / "reports" / "consistency.json").write_text(json.dumps(flags, indent=1), encoding="utf-8")
    print(f"\n{len(flags)} flagged. Nothing is excluded until you confirm; to exclude or down-weight, create "
          "data/reports/consistency_decisions.json like {\"exclude\": [\"average/B516\"], \"downweight\": []}.")


if __name__ == "__main__":
    main()
