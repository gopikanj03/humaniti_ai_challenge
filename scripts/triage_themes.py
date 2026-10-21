"""Classify each feedback theme as MEASURABLE, JUDGEABLE or NOT_OBSERVABLE and say how it is used.

MEASURABLE -> has a computed metric; JUDGEABLE -> a rubric dimension the judge scores from evidence;
NOT_OBSERVABLE -> listed under limitations. Run derive_rubric.py first.

    python scripts/triage_themes.py
"""
from __future__ import annotations

import json

from dataset import DATA
from themes import THEMES


def main() -> None:
    by_theme = json.loads((DATA / "reports" / "themes.json").read_text(encoding="utf-8"))["by_theme"]
    rows = []
    for cat in ("MEASURABLE", "JUDGEABLE", "NOT_OBSERVABLE"):
        print(f"\n== {cat} ==")
        for t in THEMES:
            if t.category != cat or t.id not in by_theme:
                continue
            b = by_theme[t.id]
            print(f"  {t.name:44s} {b['mentions']:2d} mentions (+{b['praise']}/-{b['criticism']})  "
                  f"-> {t.dimension or 'limitations'} via {t.signal}")
            if t.rationale:
                print(f"      note: {t.rationale}")
            rows.append({"theme": t.id, "name": t.name, "category": cat, "dimension": t.dimension,
                         "signal": t.signal, "mentions": b["mentions"]})
    if "unmatched" in by_theme:
        print(f"\n{by_theme['unmatched']['mentions']} phrase(s) matched no theme: review data/reports/themes.json")
    (DATA / "reports" / "triage.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nwrote data/reports/triage.json")


if __name__ == "__main__":
    main()
