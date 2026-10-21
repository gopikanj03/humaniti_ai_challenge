"""Local visual signals from sparse frames: tutor camera, 8x8 branded background, black frames, static screen.

Frames are sampled every ~45 s, decoded by ffmpeg straight into memory (never written to disk) and analysed
with OpenCV on this machine; nothing is sent to any service. Results describe SAMPLED moments only.

Deliberately not measured: lighting quality from face brightness. Face brightness depends on skin tone,
and frame brightness depends on the (dark) 8x8 background, so neither is a fair lighting signal. We only
report fully black frames (camera covered or video off).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .config import ROOT, get_config
from .schemas import Metric

log = logging.getLogger(__name__)


class FrameObs(BaseModel):
    t: float
    faces: int
    largest_face: float          # face height / frame height
    camera_view: bool            # tutor camera shown large
    logo: float | None = None    # best template match (only checked in camera-view frames)
    near_black: bool = False
    changed: bool | None = None  # vs the previous sample; None for the first


class VisualResult(BaseModel):
    interval_s: float
    frames: list[FrameObs] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)   # timestamped, for the judge
    metrics: list[Metric] = Field(default_factory=list)
    vision_model_status: str = ""
    warnings: list[str] = Field(default_factory=list)


def _fmt(t: float) -> str:
    t = int(round(t))
    return f"{t // 60:02d}:{t % 60:02d}" if t < 3600 else f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def sample_times(duration: float, interval: float, max_frames: int) -> list[float]:
    if duration <= 0:
        return []
    step = max(interval, duration / max_frames)
    return [round(t, 1) for t in np.arange(min(step / 2, duration / 2), duration, step)]


def grab_frame(video: str | Path, t: float, width: int) -> np.ndarray | None:
    """One frame at time t, scaled to `width`, decoded in memory (BGR)."""
    proc = subprocess.run([get_config()["ffmpeg_bin"], "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
                           "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "bmp", "-"],
                          capture_output=True, timeout=60)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)


class LogoMatcher:
    def __init__(self, frame_width: int):
        vcfg = get_config()["visual"]
        path = ROOT / vcfg["logo_template"]
        tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.available = tmpl is not None
        base = frame_width / 1920  # the template was cut from a 1920 px wide frame
        self.templates = [] if tmpl is None else [
            cv2.resize(tmpl, None, fx=base * s, fy=base * s, interpolation=cv2.INTER_AREA)
            for s in vcfg["logo_scales"]]

    def score(self, gray: np.ndarray) -> float:
        best = 0.0
        for t in self.templates:
            if t.shape[0] < gray.shape[0] and t.shape[1] < gray.shape[1]:
                best = max(best, float(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED).max()))
        return best


def vision_model_status() -> str:
    """Report whether a vision model could be used (we still analyse locally either way)."""
    model = get_config()["visual"].get("vision_model")
    if not model:
        return "not used: frames contain children's faces and names, so they are analysed locally only"
    return f"configured ({model}) but not enabled in this prototype; local analysis only"


def analyse(video: str | Path, duration: float) -> VisualResult:
    vcfg = get_config()["visual"]
    width = vcfg["frame_width"]
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "", (width, width),
                                    0.6, 0.3, 5000)
    matcher = LogoMatcher(width)
    frames: list[FrameObs] = []
    prev_small = None
    for t in sample_times(duration, vcfg["interval_s"], vcfg["max_frames"]):
        img = grab_frame(video, t, width)
        if img is None:
            continue
        h, w = img.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        heights = [] if faces is None else [f[3] / h for f in faces]
        largest = max(heights, default=0.0)
        camera = largest >= vcfg["camera_face_min_height"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        changed = None if prev_small is None else float(np.abs(small - prev_small).mean()) >= vcfg["static_diff"]
        prev_small = small
        frames.append(FrameObs(
            t=t, faces=len(heights), largest_face=round(largest, 3), camera_view=camera,
            logo=round(matcher.score(gray), 2) if camera and matcher.available else None,
            near_black=float(gray.mean()) < vcfg["near_black_mean"], changed=changed))
    res = VisualResult(interval_s=vcfg["interval_s"], frames=frames, vision_model_status=vision_model_status())
    if not matcher.available:
        res.warnings.append("8x8 logo template missing; background check skipped.")
    _summarise(res, matcher.available)
    log.info("visual: %d frames, %d camera-view, %d with logo", len(frames),
             sum(f.camera_view for f in frames), sum((f.logo or 0) >= vcfg["logo_threshold"] for f in frames))
    return res


def _summarise(res: VisualResult, logo_available: bool) -> None:
    thr = get_config()["visual"]["logo_threshold"]
    fr = res.frames
    n = len(fr)
    if n == 0:
        res.warnings.append("No video frames could be read.")
        res.metrics = [Metric(name="frames_sampled", value=0, source="visual", reliability="low")]
        return
    cam = [f for f in fr if f.camera_view]
    logo = [f for f in cam if (f.logo or 0) >= thr]
    no_face = [f for f in fr if f.faces == 0]
    black = [f for f in fr if f.near_black]
    static = [f for f in fr if f.changed is False]
    rel = "medium" if n >= 20 else "low"
    res.metrics = [
        Metric(name="frames_sampled", value=n, unit="frames", source="visual",
               note=f"one every ~{res.interval_s:.0f} s; sampled moments, not continuous observation"),
        Metric(name="camera_view_share", value=round(len(cam) / n, 2), source="visual", reliability=rel,
               note="tutor camera shown full-size; low values are normal while screen sharing, not a fault"),
        Metric(name="face_visible_share", value=round((n - len(no_face)) / n, 2), source="visual",
               reliability=rel, note="any face visible, including the small camera tile during screen sharing"),
        Metric(name="no_face_share", value=round(len(no_face) / n, 2), source="visual", reliability="low",
               note="no face detected: screen sharing with a tiny or hidden camera tile, or camera off"),
        Metric(name="logo_in_camera_view_share",
               value=round(len(logo) / len(cam), 2) if cam and logo_available else None, source="visual",
               reliability=rel if len(cam) >= 3 else "low",
               note="8x8 branded background detected, among camera-view frames"),
        Metric(name="near_black_frame_share", value=round(len(black) / n, 2), source="visual", reliability=rel,
               note="fully black frames (camera covered / video off). Lighting quality is not measured"),
        Metric(name="static_screen_share", value=round(len(static) / max(n - 1, 1), 2), source="visual",
               reliability=rel, note="screen unchanged between consecutive samples; not a fault by itself"),
    ]
    obs = [f"Sampled {n} frames, one every ~{res.interval_s:.0f} s (moments, not continuous video).",
           f"A face is visible in {n - len(no_face)}/{n} sampled frames. The tutor camera is shown full-size in "
           f"{len(cam)}/{n}; in most other frames the tutor is a small tile beside the screen-shared board, which "
           "is normal for these classes and not a fault.",
           f"The 8x8 background can only be checked when the camera is full-size, so it was checked in "
           f"{len(cam)} frames" + (f" and detected in {len(logo)} of them." if cam and logo_available else ".")]
    for f in cam[:12]:  # timestamped details for the judge to cite
        state = "8x8 logo visible" if (f.logo or 0) >= thr else "8x8 logo not detected"
        obs.append(f"[{_fmt(f.t)}] tutor camera shown large; {state}.")
    for a, b in _runs([f for f in fr], lambda f: f.faces == 0, min_len=6):
        obs.append(f"[{_fmt(a)}-{_fmt(b)}] no face detected in consecutive samples (screen share with hidden "
                   "camera tile, or camera off; cannot tell which).")
    for f in black[:5]:
        obs.append(f"[{_fmt(f.t)}] frame is black (camera covered or video off).")
    res.observations = obs


def _runs(frames: list[FrameObs], cond, min_len: int) -> list[tuple[float, float]]:
    out, start, count, last = [], None, 0, None
    for f in frames:
        if cond(f):
            start = f.t if start is None else start
            count += 1
            last = f.t
        else:
            if count >= min_len:
                out.append((start, last))
            start, count = None, 0
    if count >= min_len:
        out.append((start, last))
    return out
