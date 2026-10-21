"""Prepare the ClassCritique logo (assets/logo.png) from a source image.

Two kinds of source are handled automatically:
  * already transparent (PNG with alpha): the soft semi-transparent glow is removed, because on the dark
    theme it turns into a grey haze; only the solid artwork is kept, with a short fade for smooth edges;
  * on a black background: the largest non-black region is found (ignoring stray marks such as a
    watermark), a circle is fitted to it and everything outside is made transparent.
The result is cropped to the artwork and padded to a square (it is also the browser-tab icon).

    python scripts/make_logo.py assets/logo_source.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
OUT_SIZE = 512


GLOW_ALPHA = (170, 230)  # alpha below 170 -> removed, above 230 -> kept, linear fade in between


def trim_glow(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("RGBA")).astype(float)
    lo, hi = GLOW_ALPHA
    a[..., 3] = np.clip((a[..., 3] - lo) / (hi - lo), 0, 1) * 255
    out = Image.fromarray(a.astype("uint8"), "RGBA")
    return out.crop(out.getbbox())


def square(img: Image.Image) -> Image.Image:
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.alpha_composite(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)


def prepare(src: Path) -> Image.Image:
    img = Image.open(src)
    if img.mode in ("RGBA", "LA") and np.asarray(img.convert("RGBA"))[..., 3].min() < 255:
        return square(trim_glow(img))
    return cut_out(img)


def cut_out(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    rgb = np.asarray(img).astype(int)
    # The logo's dark purple is still far brighter than the black background.
    mask = rgb.sum(axis=2) > 60
    labels, n = ndimage.label(mask)
    if n == 0:
        raise SystemExit("no logo found in the image")
    largest = 1 + int(np.argmax(ndimage.sum(mask, labels, range(1, n + 1))))
    ys, xs = np.nonzero(labels == largest)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = min(xs.max() - xs.min(), ys.max() - ys.min()) / 2 - 2  # shrink slightly to drop the dark edge
    box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
    logo = img.crop(box).convert("RGBA")

    # Anti-aliased circular alpha: draw the circle 4x larger, then downsample.
    side = logo.width
    alpha = Image.new("L", (side * 4, side * 4), 0)
    ImageDraw.Draw(alpha).ellipse([0, 0, side * 4 - 1, side * 4 - 1], fill=255)
    logo.putalpha(alpha.resize((side, side), Image.LANCZOS))
    return logo.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/make_logo.py <source image>")
    out = ROOT / "assets" / "logo.png"
    out.parent.mkdir(exist_ok=True)
    prepare(Path(sys.argv[1])).save(out)
    print("wrote", out)
