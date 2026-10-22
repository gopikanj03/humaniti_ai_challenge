"""Load saved run results (data/runs) for the analysis scripts."""
from __future__ import annotations

import json

from dataset import DATA, HOLDOUT, RUNS

RANK = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
LABELS = ["Poor", "Average", "Good", "Excellent"]


def load_runs(include_holdout: bool = False, base=RUNS) -> list[dict]:
    excluded = set()
    decisions = DATA / "reports" / "consistency_decisions.json"
    if decisions.exists():  # only created by the dataset owner after reviewing check_consistency.py
        excluded = set(json.loads(decisions.read_text(encoding="utf-8")).get("exclude", []))
    runs = []
    for p in sorted(base.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["human_label"] is None or r["key"] in excluded:
            continue
        if r["key"] in HOLDOUT and not include_holdout:
            continue
        r["rank"] = RANK[r["human_label"]]
        r["metric_values"] = {m["name"]: m["value"] for m in r["metrics"]}
        r["dims"] = {d["dimension"]: d["score"] for d in r["assessment"]["dimensions"]}
        runs.append(r)
    return runs
