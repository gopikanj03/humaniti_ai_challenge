"""Run the full pipeline on every dataset recording, saving a text-free result per sample.

Resumable: samples with a saved result are skipped, and pipeline stages are cached (this script, unlike
the app, enables cache writes), so a run stopped by free-tier rate limits continues where it left off.
Saved results contain the assessment, scores and metrics only (pseudonymised; no transcript).

    python scripts/run_samples.py              # labelled samples, then demos
    python scripts/run_samples.py --only weak/B482          # quick check on one sample
    python scripts/run_samples.py --force good/B552
"""
from __future__ import annotations

import argparse
import json
import time

from dataset import RUNS, list_samples, run_path
from src import cache, pipeline
from src.config import setup_logging
from src.net import ProviderError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], help="re-run these sample keys even if saved")
    ap.add_argument("--labelled-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=[], help="run just these sample keys (quick checks)")
    args = ap.parse_args()
    setup_logging()
    cache.enable_writes()
    RUNS.mkdir(parents=True, exist_ok=True)

    samples = sorted(list_samples(), key=lambda s: s.label is None)  # labelled first
    if args.labelled_only:
        samples = [s for s in samples if s.label]
    if args.only:
        samples = [s for s in samples if s.key in args.only]
    for s in samples:
        out = run_path(s.key)
        if out.exists() and s.key not in args.force:
            print(f"skip  {s.key} (already done)")
            continue
        t0 = time.time()
        print(f"run   {s.key} ...", flush=True)
        try:
            res = pipeline.run(s.video)
        except ProviderError as e:
            if "429" in str(e) or "413" in str(e):
                print(f"STOP  rate limit reached at {s.key}: {e}. Re-run later to resume.")
                return
            print(f"FAIL  {s.key}: {e}")
            continue
        record = {
            "key": s.key, "human_label": s.label, "summary": pipeline.summarize(res),
            "result": res.result.model_dump(), "assessment": res.assessment.model_dump(),
            "metrics": [m.model_dump() for m in res.metrics], "signals": [x.model_dump() for x in res.signals],
            "events_by_type": _event_counts(res),
        }
        out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"done  {s.key}: {res.result.label} ({res.result.score:.2f}, conf {res.result.confidence:.2f}) "
              f"human={s.label} in {time.time() - t0:.0f}s", flush=True)


def _event_counts(res) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in res.digests:
        for e in d.events:
            counts[e.type] = counts.get(e.type, 0) + 1
    return counts


if __name__ == "__main__":
    main()
