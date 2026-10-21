"""Cluster recurring themes in the human feedback and propose rubric dimensions.

Writes data/reports/themes.json (phrase clusters; gitignored, derived from feedback) and
config/rubric.draft.json (dimension proposals with theme names and counts only) for human review.
The live rubric (config/rubric.json) is never overwritten.

    python scripts/derive_rubric.py
"""
from __future__ import annotations

import json
from collections import Counter

from dataset import DATA, ROOT, load_feedback
from themes import THEMES, cluster_phrases, split_phrases, themes_for

REPORTS = DATA / "reports"


def main() -> None:
    feedback = load_feedback()
    phrases = [p for f in feedback.values() for p in split_phrases(f.key, f.label, f.text)]
    for p in phrases:
        p.themes = themes_for(p.text)
    clusters = cluster_phrases(phrases)

    print(f"{len(feedback)} notes -> {len(phrases)} phrases -> {len(clusters)} clusters\n")
    report = []
    for c in clusters:
        theme_votes = Counter(t for p in c for t in p.themes)
        theme = theme_votes.most_common(1)[0][0] if theme_votes else None
        samples = sorted({p.sample for p in c})
        entry = {"theme": theme, "size": len(c), "samples": len(samples),
                 "negative": sum(p.negative for p in c), "positive": sum(not p.negative for p in c),
                 "labels": dict(Counter(p.label for p in c)), "phrases": [p.norm for p in c]}
        report.append(entry)
        flag = "" if theme else "   <-- no named theme: candidate new dimension"
        print(f"[{theme or '?':22s}] {len(c):2d} phrases in {len(samples)} notes "
              f"(+{entry['positive']} / -{entry['negative']}){flag}")

    # Per theme: how often it is praised vs criticised, and in which label bands.
    by_theme: dict[str, dict] = {}
    for p in phrases:
        for t in p.themes or ["unmatched"]:
            d = by_theme.setdefault(t, {"mentions": 0, "praise": 0, "criticism": 0, "labels": Counter()})
            d["mentions"] += 1
            d["criticism" if p.negative else "praise"] += 1
            d["labels"][p.label] += 1

    dims: dict[str, dict] = {}
    for t in THEMES:
        if t.dimension and t.id in by_theme:
            d = dims.setdefault(t.dimension, {"themes": [], "mentions": 0})
            d["themes"].append(t.name)
            d["mentions"] += by_theme[t.id]["mentions"]
    total = sum(d["mentions"] for d in dims.values())
    draft = {
        "note": "DRAFT from scripts/derive_rubric.py. Weights proportional to how often reviewers mention "
                "each dimension's themes. Review before copying anything into config/rubric.json.",
        "dimensions": [{"id": k, "themes": v["themes"], "feedback_mentions": v["mentions"],
                        "proposed_weight": round(v["mentions"] / total, 2)}
                       for k, v in sorted(dims.items(), key=lambda kv: -kv[1]["mentions"])],
        "not_observable": [t.name for t in THEMES if t.category == "NOT_OBSERVABLE" and t.id in by_theme],
        "unmatched_phrases": by_theme.get("unmatched", {}).get("mentions", 0),
    }
    print("\nProposed dimensions (weight = share of reviewer mentions):")
    for d in draft["dimensions"]:
        print(f"  {d['id']:24s} {d['proposed_weight']:.2f}  ({d['feedback_mentions']} mentions: {', '.join(d['themes'])})")
    print("  not observable:", draft["not_observable"], "| unmatched phrases:", draft["unmatched_phrases"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "themes.json").write_text(json.dumps({"clusters": report, "by_theme": {
        k: {**v, "labels": dict(v["labels"])} for k, v in by_theme.items()}}, indent=1), encoding="utf-8")
    (ROOT / "config" / "rubric.draft.json").write_text(json.dumps(draft, indent=1), encoding="utf-8")
    print("\nwrote data/reports/themes.json and config/rubric.draft.json")


if __name__ == "__main__":
    main()
