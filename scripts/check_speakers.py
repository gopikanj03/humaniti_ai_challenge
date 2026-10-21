"""Measure teacher/student separation against the Zoom VTTs (which know who spoke).

Runs fully locally (no API). Embeddings are cached per recording so threshold sweeps are fast.
Prints numbers only: no names or transcript text.

    python scripts/check_speakers.py                  # all samples with a VTT
    python scripts/check_speakers.py --only good/B552
"""
from __future__ import annotations

import argparse
import shutil
import tempfile

import numpy as np

from dataset import DATA, list_samples, parse_vtt, vtt_teacher
from src import media, speakers
from src.cache import file_sha256
from src.config import get_config

THRESHOLDS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
STEP = 0.5  # seconds between comparison points


def embeddings_for(sample) -> tuple[list, np.ndarray, list]:
    cache_dir = DATA / "cache" / file_sha256(sample.video)[:16]
    cache_file = cache_dir / "speaker_embeddings.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        return [tuple(w) for w in z["windows"]], z["embs"], [tuple(r) for r in z["regions"]]
    work = tempfile.mkdtemp(prefix="classeval_")
    try:
        wav = media.extract_audio(sample.video, work)
        audio, sr = speakers.read_wav(wav)
        regions = speakers.speech_regions(audio, sr)
        s = get_config()["speakers"]
        windows = speakers._windows(regions, s["window_s"], s["step_s"], s["min_region_s"])
        embs = speakers._embed(audio, sr, windows)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, windows=np.array(windows), embs=embs, regions=np.array(regions))
    return windows, embs, regions


def compare(result: speakers.SpeakerResult, cues) -> dict:
    teacher = vtt_teacher(cues)
    t_true = s_true = t_hit = s_hit = missed = 0
    for c in cues:
        t = c.start
        while t < c.end:
            frac = speakers.teacher_fraction(result.turns, t, t + STEP)
            if frac is None:
                missed += 1
            elif c.speaker == teacher:
                t_true += 1
                t_hit += frac >= 0.5
            else:
                s_true += 1
                s_hit += frac < 0.5
            t += STEP
    total_vtt = sum(c.end - c.start for c in cues)
    vtt_share = sum(c.end - c.start for c in cues if c.speaker == teacher) / total_vtt
    t_recall = t_hit / max(t_true, 1)
    s_recall = s_hit / max(s_true, 1)
    return {"teacher_recall": t_recall, "student_recall": s_recall, "balanced_acc": (t_recall + s_recall) / 2,
            "vtt_share": vtt_share, "our_share": result.teacher_share or 0.0,
            "coverage": 1 - missed / max(t_true + s_true + missed, 1), "confidence": result.confidence}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    args = ap.parse_args()

    samples = [s for s in list_samples() if s.vtt and (not args.only or s.key == args.only)]
    # The Malayalam demo VTT is Zoom's failed English captioning, but speaker attribution is still valid.
    rows: dict[float, list[dict]] = {th: [] for th in THRESHOLDS}
    for s in samples:
        windows, embs, regions = embeddings_for(s)
        cues = parse_vtt(s.vtt)
        print(f"\n{s.key:22s} windows={len(windows)}")
        for th in THRESHOLDS:
            labels = speakers.cluster(embs, th)
            res = speakers.build_result(windows, labels, embs, regions)
            r = compare(res, cues)
            rows[th].append(r)
            print(f"  thr={th:.1f} clusters={res.n_voice_clusters:3d} teacher_recall={r['teacher_recall']:.2f} "
                  f"student_recall={r['student_recall']:.2f} balanced={r['balanced_acc']:.2f} "
                  f"share ours/vtt={r['our_share']:.2f}/{r['vtt_share']:.2f} coverage={r['coverage']:.2f} "
                  f"conf={r['confidence']:.2f}")

    print("\n=== mean over samples ===")
    for th, rs in rows.items():
        if rs:
            m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
            err = np.mean([abs(r["our_share"] - r["vtt_share"]) for r in rs])
            print(f"thr={th:.1f} balanced_acc={m('balanced_acc'):.3f} teacher_recall={m('teacher_recall'):.2f} "
                  f"student_recall={m('student_recall'):.2f} |share error|={err:.3f}")


if __name__ == "__main__":
    main()
