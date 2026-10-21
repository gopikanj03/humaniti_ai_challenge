"""Create the 8x8 logo template used by src/visual.py for local logo detection.

Crops the right-hand "8" of the 8x8 branded background from one frame of a dataset recording where the
tutor's camera is full-size and that part of the logo is not covered. The crop box was chosen so it
contains only the logo (no person); the script refuses to save if a face overlaps it.

    python scripts/make_logo_template.py
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

from dataset import DATA, ROOT
from src.config import get_config

SOURCE = DATA / "good" / "B552 Beginner Part  - 2026-09-02.mp4"
TIME_S = 420
BOX = (1222, 240, 1418, 622)  # x0, y0, x1, y1 in the 1920x1080 frame


def main() -> None:
    raw = subprocess.run([get_config()["ffmpeg_bin"], "-v", "error", "-ss", str(TIME_S), "-i", str(SOURCE),
                          "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "", (w, h), 0.5, 0.3, 5000)
    _, faces = det.detect(img)
    x0, y0, x1, y1 = BOX
    for f in [] if faces is None else faces:
        fx, fy, fw, fh = map(int, f[:4])
        if fx < x1 and fx + fw > x0 and fy < y1 and fy + fh > y0:
            raise SystemExit("a face overlaps the template box; refusing to save")
    out = ROOT / "assets" / "8x8_logo_template.png"
    cv2.imwrite(str(out), cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    print("wrote", out)


if __name__ == "__main__":
    main()
