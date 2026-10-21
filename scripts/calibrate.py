"""Tune dimension weights and label thresholds with simple, leave-one-out-validated methods.

No model is trained: with ~8 labelled recordings anything more flexible would just memorise them.
  * Thresholds: midpoints between the mean weighted scores of adjacent labels (kept monotonic).
  * Weights: compare a few fixed candidates (current rubric, equal, feedback-mention draft) plus one
    data-driven one (each dimension's positive Spearman correlation with the label), recomputed inside
    every leave-one-out fold so the held-out sample never influences its own prediction.
  * The held-out recordings (dataset.HOLDOUT) are scored once, at the end, with the chosen setting.
Prints rank agreement (Spearman), exact and within-one LOO accuracy, and a confusion matrix.
Nothing is written to config unless --apply is given.

    python scripts/calibrate.py [--apply]
"""
from __future__ import annotations

import argparse
import json

from scipy.stats import spearmanr

from dataset import DATA, HOLDOUT, ROOT
from runs_data import LABELS, load_runs


def weighted(dims: dict, weights: dict) -> float | None:
    present = {d: s for d, s in dims.items() if s is not None and weights.get(d, 0) > 0}
    if not present:
        return None
    return sum(weights[d] * s for d, s in present.items()) / sum(weights[d] for d in present)


def fit_thresholds(scores: list[float], ranks: list[int]) -> dict[str, float]:
    means = []
    for i in range(4):
        vals = [s for s, r in zip(scores, ranks) if r == i]
        means.append(sum(vals) / len(vals) if vals else None)
    # Fill labels with no examples by interpolating, then force increasing means.
    known = [(i, m) for i, m in enumerate(means) if m is not None]
    for i in range(4):
        if means[i] is None:
            lo = max((k for k in known if k[0] < i), default=None, key=lambda k: k[0])
            hi = min((k for k in known if k[0] > i), default=None, key=lambda k: k[0])
            means[i] = (lo[1] + 0.5) if lo and not hi else (hi[1] - 0.5) if hi and not lo else (lo[1] + hi[1]) / 2
    for i in range(1, 4):
        means[i] = max(means[i], means[i - 1] + 0.05)
    th = {"Poor": 0.0}
    for i in range(1, 4):
        th[LABELS[i]] = round((means[i - 1] + means[i]) / 2, 2)
    return th


def predict(score: float, th: dict) -> int:
    return max(i for i, lab in enumerate(LABELS) if score >= th[lab])


def corr_weights(runs: list[dict], dims: list[str]) -> dict[str, float]:
    w = {}
    for d in dims:
        pairs = [(r["dims"].get(d), r["rank"]) for r in runs if r["dims"].get(d) is not None]
        if len(pairs) >= 4 and len({x for x, _ in pairs}) > 1:
            w[d] = max(0.0, spearmanr([x for x, _ in pairs], [y for _, y in pairs]).statistic)
        else:
            w[d] = 0.0
    total = sum(w.values())
    return {d: (v / total if total else 1 / len(dims)) for d, v in w.items()}


def evaluate(runs: list[dict], name: str, make_weights) -> dict:
    preds, scores_all = [], []
    for i, held in enumerate(runs):
        train = runs[:i] + runs[i + 1:]
        w = make_weights(train)
        train_scores = [(weighted(r["dims"], w), r["rank"]) for r in train]
        train_scores = [(s, y) for s, y in train_scores if s is not None]
        th = fit_thresholds([s for s, _ in train_scores], [y for _, y in train_scores])
        s = weighted(held["dims"], w)
        preds.append(None if s is None else predict(s, th))
        scores_all.append(s)
    ranks = [r["rank"] for r in runs]
    ok = [(p, y) for p, y in zip(preds, ranks) if p is not None]
    rho = spearmanr([s for s in scores_all if s is not None],
                    [y for s, y in zip(scores_all, ranks) if s is not None]).statistic
    return {"name": name, "loo_exact": sum(p == y for p, y in ok) / len(ok),
            "loo_within_one": sum(abs(p - y) <= 1 for p, y in ok) / len(ok), "rank_agreement": rho,
            "preds": preds, "ranks": ranks}


def confusion(preds: list[int], ranks: list[int]) -> str:
    lines = ["human \\ predicted  " + "  ".join(f"{lab[:4]:>4s}" for lab in LABELS)]
    for y, lab in enumerate(LABELS):
        row = [sum(1 for p, r in zip(preds, ranks) if r == y and p == x) for x in range(4)]
        lines.append(f"{lab:18s} " + "  ".join(f"{c:4d}" for c in row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the chosen weights/thresholds to config")
    args = ap.parse_args()

    runs = load_runs()
    if len(runs) < 5:
        print(f"Only {len(runs)} labelled runs; finish scripts/run_samples.py first."); return
    rubric = json.loads((ROOT / "config" / "rubric.json").read_text(encoding="utf-8"))
    dims = [d["id"] for d in rubric["dimensions"]]
    current = {d["id"]: d["weight"] for d in rubric["dimensions"]}
    draft_path = ROOT / "config" / "rubric.draft.json"
    draft = ({d["id"]: d["proposed_weight"] for d in json.loads(draft_path.read_text())["dimensions"]}
             if draft_path.exists() else current)

    print(f"Calibrating on {len(runs)} labelled recordings (held out: {', '.join(HOLDOUT)})")
    print("WARNING: tiny sample. Each recording is ~12% of the data; expect results to move a lot with one more "
          "label. These are sanity checks, not a validated model.\n")
    candidates = [
        ("current rubric weights", lambda tr: current),
        ("equal weights", lambda tr: {d: 1.0 for d in dims}),
        ("feedback-mention weights (draft)", lambda tr: draft),
        ("correlation weights (fit per fold)", lambda tr: corr_weights(tr, dims)),
    ]
    results = [evaluate(runs, n, f) for n, f in candidates]
    for r in results:
        print(f"{r['name']:36s} LOO exact {r['loo_exact']:.0%}   within one label {r['loo_within_one']:.0%}   "
              f"rank agreement (Spearman) {r['rank_agreement']:+.2f}")

    # Prefer simple weights unless the data-driven one is clearly better (it is the most likely to overfit).
    best = max(results, key=lambda r: (round(r["loo_within_one"], 2), round(r["loo_exact"], 2),
                                       r["name"] != "correlation weights (fit per fold)"))
    print(f"\nChosen: {best['name']}\nLeave-one-out confusion matrix:\n{confusion(best['preds'], best['ranks'])}")

    maker = dict(candidates)[best["name"]]
    final_w = maker(runs)
    scored = [(weighted(r["dims"], final_w), r["rank"]) for r in runs]
    final_th = fit_thresholds([s for s, _ in scored if s is not None], [y for s, y in scored if s is not None])
    norm = {d: round(final_w.get(d, 0) / sum(final_w.values()), 2) for d in dims}
    print(f"\nWeights: {norm}\nThresholds: {final_th}")

    held = [r for r in load_runs(include_holdout=True) if r["key"] in HOLDOUT]
    print("\nHeld-out check (never used above):")
    for r in held:
        s = weighted(r["dims"], final_w)
        pred = "n/a" if s is None else LABELS[predict(s, final_th)]
        print(f"  {r['key']:20s} human={r['human_label']:10s} predicted={pred:10s} score={s if s is None else round(s, 2)}")
    if not held:
        print("  (held-out recordings not run yet)")

    out = {"chosen": best["name"], "weights": norm, "thresholds": final_th,
           "candidates": [{k: v for k, v in r.items() if k not in ("preds", "ranks")} for r in results]}
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    (DATA / "reports" / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote data/reports/calibration.json")
    if args.apply:
        for d in rubric["dimensions"]:
            d["weight"] = norm[d["id"]]
        (ROOT / "config" / "rubric.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")
        import re
        cfg_path = ROOT / "config" / "config.yaml"
        cfg = cfg_path.read_text(encoding="utf-8")
        for lab, v in final_th.items():
            cfg = re.sub(rf"(\n    {lab}: )[\d.]+", rf"\g<1>{v}", cfg)
        cfg_path.write_text(cfg, encoding="utf-8")
        print("applied to config/rubric.json and config/config.yaml")
    else:
        print("Not applied. Re-run with --apply to write these to config.")


if __name__ == "__main__":
    main()
