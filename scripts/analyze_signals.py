"""Which signals separate the label levels, and which mislead? (Spearman correlation with label rank.)

Uses the labelled runs EXCEPT the held-out recordings. With ~8 samples, |rho| < 0.6 is not
distinguishable from noise; treat everything here as a hint, not a finding.

    python scripts/analyze_signals.py
"""
from __future__ import annotations

import json
import math

from scipy.stats import spearmanr

from dataset import DATA
from runs_data import LABELS, load_runs

# Direction a reasonable reviewer would expect (+1: higher is better). Used to spot misleading signals.
EXPECTED = {"questions_per_10min": 1, "open_question_share": 1, "question_response_rate": 1,
            "students_addressed_individually": 1, "praise_count": 1, "homework_given": 1,
            "greeting_or_rapport_in_first_10min": 1, "recap_or_conclusion_in_last_10min": 1,
            "student_talk_share": 1, "student_turns_per_10min": 1, "technical_issues": -1,
            "teacher_talk_share": -1, "long_silences": 0, "speech_ratio": 0, "transcription_confidence": 0}


def _num(v):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None


def main() -> None:
    runs = load_runs()
    print(f"{len(runs)} labelled runs (held-out recordings excluded)\n")
    if len(runs) < 4:
        print("Too few runs yet; finish scripts/run_samples.py first."); return
    signals = {}
    for r in runs:
        for k, v in {**r["metric_values"], **{f"dim:{d}": s for d, s in r["dims"].items()},
                     "overall_score": r["result"]["score"]}.items():
            if _num(v) is not None:
                signals.setdefault(k, []).append((_num(v), r["rank"]))

    rows = []
    for name, pairs in signals.items():
        if len(pairs) < 4 or len({x for x, _ in pairs}) < 2:
            continue
        rho = spearmanr([x for x, _ in pairs], [y for _, y in pairs]).statistic
        exp = EXPECTED.get(name, 1 if name.startswith("dim:") or name == "overall_score" else 0)
        verdict = ("separates" if abs(rho) >= 0.6 else "weak" if abs(rho) >= 0.3 else "no signal")
        if exp and rho * exp <= -0.3:
            verdict = "MISLEADS (wrong direction)"
        means = {lab: round(sum(x for x, y in pairs if y == i) / max(1, sum(1 for _, y in pairs if y == i)), 2)
                 for i, lab in enumerate(LABELS)}
        rows.append({"signal": name, "n": len(pairs), "rho": round(rho, 2), "verdict": verdict, "means": means})
    rows.sort(key=lambda r: -abs(r["rho"]))
    print(f"{'signal':40s} {'n':>2s} {'rho':>6s}  {'verdict':28s} mean by label (P/A/G/E)")
    for r in rows:
        m = r["means"]
        print(f"{r['signal']:40s} {r['n']:2d} {r['rho']:+6.2f}  {r['verdict']:28s} "
              f"{m['Poor']} / {m['Average']} / {m['Good']} / {m['Excellent']}")
    (DATA / "reports").mkdir(parents=True, exist_ok=True)
    (DATA / "reports" / "signals.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nCaution: n is tiny. A single recording can flip a correlation; use this to choose what to look at, "
          "not as proof.")


if __name__ == "__main__":
    main()
