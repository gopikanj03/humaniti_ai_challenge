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
from src.config import get_config
from src.llm import LLMOutputError
from src.net import ProviderError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], help="re-run these sample keys even if saved")
    ap.add_argument("--labelled-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=[], help="run just these sample keys (quick checks)")
    ap.add_argument("--wait", action="store_true",
                    help="on a free-tier usage limit, wait until it resets and retry instead of stopping")
    ap.add_argument("--max-wait-hours", type=float, default=12.0)
    ap.add_argument("--judge-model", help="judge every class with this model (overrides config for this run)")
    ap.add_argument("--out", default="", help="save results under data/runs/<out>/ (e.g. one folder per judge model)")
    ap.add_argument("--consistent", action="store_true",
                    help="final checks: judge only with the configured judge model (no fallback; wait instead) and "
                         "summarise with the primary model only, so every class is scored the same way")
    args = ap.parse_args()
    setup_logging()
    cache.enable_writes()
    out_dir = RUNS / args.out if args.out else RUNS
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if args.judge_model:
        get_config()["llm"]["judge_model"] = args.judge_model
    if args.consistent:
        llm_cfg = get_config()["llm"]
        llm_cfg["judge_fallback_models"] = []
        llm_cfg["condense_parallel_models"] = []
    samples = sorted(list_samples(), key=lambda s: s.label is None)  # labelled first
    if args.labelled_only:
        samples = [s for s in samples if s.label]
    if args.only:
        samples = [s for s in samples if s.key in args.only]
    for s in samples:
        out = run_path(s.key, out_dir)
        if out.exists() and s.key not in args.force:
            print(f"skip  {s.key} (already done)")
            continue
        t0 = time.time()
        print(f"run   {s.key} ...", flush=True)
        res, bad_answers = None, 0
        while res is None:
            try:
                res = pipeline.run(s.video)
            except LLMOutputError:
                bad_answers += 1
                if bad_answers <= 2:
                    print(f"retry {s.key}: unusable model answer", flush=True)
                    continue
                print(f"FAIL  {s.key}: model answer unusable three times")
                break
            except ProviderError as e:
                limited = e.code == "daily_limit" or "429" in str(e) or "413" in str(e)
                if limited and args.wait and time.time() - started < args.max_wait_hours * 3600:
                    pause = min(max(e.retry_after or 300, 60), 1800)
                    print(f"wait  {s.key}: usage limit, retrying in {pause / 60:.0f} min", flush=True)
                    time.sleep(pause)   # finished steps are cached, so the retry resumes where it stopped
                    continue
                print(f"{'STOP' if limited else 'FAIL'}  {s.key}: {e}" + (". Re-run later to resume." if limited else ""))
                if limited:
                    return
                break
        if res is None:
            continue
        record = {
            "key": s.key, "human_label": s.label, "summary": pipeline.summarize(res),
            "result": res.result.model_dump(), "assessment": res.assessment.model_dump(),
            "metrics": [m.model_dump() for m in res.metrics], "signals": [x.model_dump() for x in res.signals],
            "events_by_type": _event_counts(res), "judge_model": get_config()["llm"]["judge_model"],
            "seconds": round(time.time() - t0),
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
