"""Extract 8x8 reviewers' feedback style and save it as instructions for the judge prompt.

Measures (a) structure and length, (b) specificity (references to moments vs general statements),
(c) vocabulary for praise and criticism. Writes config/style.json (gitignored). Vocabulary is reduced
to short theme terms; whole notes are never copied.

    python scripts/extract_style.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter

from dataset import ROOT, load_feedback
from themes import split_phrases

MOMENT = re.compile(r"\b(beginning|start|end|during|while|when|time|minute|sometimes)\b", re.I)
OPENERS = re.compile(r"^(need(s)? (a |more )?|should (give )?|can |good |kept |consider |individually |"
                      r"directly |some |no |sometimes )", re.I)


def main() -> None:
    notes = list(load_feedback().values())
    phrases = [p for f in notes for p in split_phrases(f.key, f.label, f.text)]
    words = [len(f.text.split()) for f in notes]
    per_note = [len(split_phrases(f.key, f.label, f.text)) for f in notes]
    specific = sum(bool(MOMENT.search(p.text)) for p in phrases)

    def openers(neg: bool) -> list[str]:
        c = Counter(m.group(0).strip().lower() for p in phrases if p.negative == neg
                    for m in [OPENERS.match(p.text.strip())] if m)
        return [w for w, _ in c.most_common(6)]

    stats = {"notes": len(notes), "mean_words": round(statistics.mean(words), 1),
             "mean_phrases": round(statistics.mean(per_note), 1),
             "praise_share": round(sum(not p.negative for p in phrases) / len(phrases), 2),
             "moment_references": f"{specific} of {len(phrases)} phrases"}
    praise_terms = ["rapport building", "one-to-one interaction", "considers everyone in the class",
                    "asks questions individually", "kept the 8x8 background", "gave homework", "good intro"]
    criticism_terms = ["need more pleasant / energetic interaction", "should give homework",
                       "need a better intro and conclusion", "give new joiners time to introduce themselves",
                       "went straight into revision without rapport building", "camera / face not visible",
                       "sit somewhere with better light"]
    instructions = (
        "Write like an 8x8 class reviewer. Their notes are short lists of phrases "
        f"(about {stats['mean_phrases']:.0f} per class, ~{stats['mean_words']:.0f} words in total): what the tutor did "
        "well, then what to change, in plain practical language. Praise usually names a habit "
        f"(openers such as: {', '.join(openers(False))}); criticism is phrased as an action "
        f"(openers such as: {', '.join(openers(True))}). Reviewers' own vocabulary: praise - "
        f"{'; '.join(praise_terms)}. Criticism - {'; '.join(criticism_terms)}. "
        f"Reviewers rarely cite moments (only {stats['moment_references']} do); you must go further and tie every "
        "observation to a timestamp, while keeping their short, practical tone. Do not copy their wording verbatim.")
    # Compact version for the judge prompt (the free tier leaves little room beside the evidence).
    short = ("Write like an 8x8 reviewer: short, practical phrases; what worked, then what to change, phrased as "
             "actions (e.g. 'give new joiners time to introduce themselves', 'should give homework'). Use their "
             "terms: rapport building, one-to-one interaction, considers everyone, asks questions individually, "
             "8x8 background, pleasant/energetic interaction. Unlike them, tie every point to a timestamp. "
             "Paraphrase; never copy notes.")
    out = {"stats": stats, "instructions": instructions, "short_instructions": short}
    (ROOT / "config" / "style.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1)); print("\n" + instructions + "\n\nwrote config/style.json")


if __name__ == "__main__":
    main()
