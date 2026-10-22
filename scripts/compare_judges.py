"""Compare judge models on the labelled classes: which agrees best with 8x8's labels?

Both runs use the same summaries (cached), so only the judge differs. Prints a per-class table and, per judge:
exact matches, matches within one level, rank agreement (Spearman) and the two held-out classes.
No transcript text or names are printed.

    python scripts/compare_judges.py                                   # data/runs vs data/runs/judge_120b
    python scripts/compare_judges.py --a runs --b runs/judge_120b --names gpt-oss-20b gpt-oss-120b
"""
from __future__ import annotations

import argparse

from scipy.stats import spearmanr

from dataset import DATA, HOLDOUT
from runs_data import LABELS, RANK, load_runs


def summarise(runs: dict[str, dict]) -> dict:
    keys = sorted(runs)
    human = [RANK[runs[k]["human_label"]] for k in keys]
    ours = [LABELS.index(runs[k]["result"]["label"]) for k in keys]
    rho = spearmanr(human, ours).statistic if len(set(ours)) > 1 else float("nan")
    held = [k for k in keys if k in HOLDOUT]
    return {"n": len(keys), "exact": sum(h == o for h, o in zip(human, ours)),
            "within_one": sum(abs(h - o) <= 1 for h, o in zip(human, ours)), "rho": rho,
            "held_exact": sum(RANK[runs[k]["human_label"]] == LABELS.index(runs[k]["result"]["label"]) for k in held),
            "held_n": len(held)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs")
    ap.add_argument("--b", default="runs/judge_120b")
    ap.add_argument("--names", nargs=2, default=["gpt-oss-20b", "gpt-oss-120b"])
    args = ap.parse_args()
    sets = []
    for folder in (args.a, args.b):
        runs = {r["key"]: r for r in load_runs(include_holdout=True, base=DATA / folder)}
        sets.append(runs)
    common = sorted(set(sets[0]) & set(sets[1]))
    if not common:
        print("No classes judged by both models yet."); return

    na, nb = args.names
    print(f"{'class':22s} {'8x8':10s} {na:>14s} {nb:>14s}")
    for k in common:
        tag = " (held out)" if k in HOLDOUT else ""
        cells = []
        for runs in sets:
            r = runs[k]["result"]
            mark = "ok" if r["label"] == runs[k]["human_label"] else "  "
            cells.append(f"{r['label']} {r['score']:.1f} {mark}")
        print(f"{k + tag:22s} {runs[k]['human_label']:10s} {cells[0]:>14s} {cells[1]:>14s}")

    print()
    for name, runs in zip((na, nb), sets):
        s = summarise({k: runs[k] for k in common})
        secs = [runs[k].get("seconds") for k in common if runs[k].get("seconds")]
        print(f"{name:14s} exact {s['exact']}/{s['n']}  within one level {s['within_one']}/{s['n']}  "
              f"rank agreement {s['rho']:+.2f}  held-out exact {s['held_exact']}/{s['held_n']}"
              + (f"  avg time per class {sum(secs) / len(secs):.0f}s" if secs else ""))
    print("\nCaution: 10 classes; one class changes the exact-match count by 10 points. Prefer the judge that is "
          "better on rank agreement AND held-out classes, not just exact matches.")


if __name__ == "__main__":
    main()
