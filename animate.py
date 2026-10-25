"""Animated DROITS RÉSERVÉS: the photo stays intact, the glitch passes through it.

A slow scan melts a thin band of pixels as it travels down the frame, and two
short bursts (RGB split, slice shifts, streaks, mosaic) break the image before
it settles again. The loop is seamless.

    python3 animate.py            -> output/droits-reserves-18-passage.mp4
"""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(ROOT, "source", "blonde.png")
OUT = os.path.join(ROOT, "output", "droits-reserves-18-passage.mp4")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
LABEL = "D R O I T S   R É S E R V É S   # 1 8    ·    P A S S A G E"

W, H = 1080, 1620
FPS, SECONDS = 24, 8
N = FPS * SECONDS
BURSTS = [(2.1, 0.45), (5.6, 0.35)]   # (start s, length s)


def load():
    img = Image.open(SOURCE).convert("RGB").resize((W, H), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255
    # gentle S-curve and deeper shadows so the pastel photo doesn't read flat
    a = np.clip((a - 0.5) * 1.18 + 0.47, 0, 1)
    a = a * a * (3 - 2 * a) * 0.5 + a * 0.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.28 * (((xx / W - 0.45) ** 2 + (yy / H - 0.4) ** 2) * 2.2)
    return np.clip(a * vig[..., None], 0, 1)


def lum(a):
    return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def melt(a, y0, y1, cols):
    """Pixel-sort columns `cols` between rows y0 and y1 by brightness."""
    y0, y1 = max(0, y0), min(H, y1)
    if y1 - y0 < 2:
        return
    band = a[y0:y1, cols]
    idx = np.argsort(lum(band), axis=0)
    a[y0:y1, cols] = np.take_along_axis(band, idx[..., None], axis=0)


def burst_level(t):
    """0..1 envelope: sharp attack, quick decay, for each burst."""
    e = 0.0
    for s, d in BURSTS:
        u = (t - s) / d
        if 0 <= u <= 1:
            e = max(e, np.sin(np.pi * u) ** 0.5)
    return e


def frame(base, i, label_img):
    t = i / FPS
    rng = np.random.default_rng(1800 + i)
    a = base.copy()

    # 1. the scan: a thin band melting as it moves down, once per loop
    sy = int((i / N) * (H + 300)) - 150
    col_rng = np.random.default_rng(42)          # same columns every frame -> coherent drips
    cols = np.flatnonzero(col_rng.random(W) < 0.35)
    melt(a, sy - 70, sy, cols)
    a[max(0, sy):min(H, sy + 2)] = np.clip(a[max(0, sy):min(H, sy + 2)] * 1.25 + 0.05, 0, 1)

    # 2. bursts
    e = burst_level(t)
    if e > 0:
        shift = int(e * 9)
        a[..., 0] = np.roll(a[..., 0], shift, 1)
        a[..., 2] = np.roll(a[..., 2], -shift, 1)
        for _ in range(int(2 + e * 6)):                      # slice shifts
            y = int(rng.integers(0, H - 40))
            h = int(rng.integers(6, 60))
            a[y:y + h] = np.roll(a[y:y + h], int(rng.integers(-60, 60) * e), 1)
        for _ in range(int(e * 4)):                          # vertical streaks
            x, w = int(rng.integers(0, W - 120)), int(rng.integers(10, 120))
            y, h = int(rng.integers(0, H - 400)), int(rng.integers(80, 400))
            a[y:y + h, x:x + w] = a[y, x:x + w][None]
        if e > 0.6:                                          # one mosaic block
            b = 18
            x, y = int(rng.integers(0, W - 216)) // b * b, int(rng.integers(0, H - 144)) // b * b
            blk = a[y:y + 144, x:x + 216]
            m = blk.reshape(8, b, 12, b, 3).mean((1, 3))
            blk[:] = np.repeat(np.repeat(m, b, 0), b, 1)
    elif rng.random() < 0.05:                                # rare micro-glitch
        y = int(rng.integers(0, H - 20))
        a[y:y + 12] = np.roll(a[y:y + 12], int(rng.integers(3, 14)), 1)

    # 3. living grain
    g = rng.normal(0, 0.03, (H, W, 1)).astype(np.float32)
    a = np.clip(a + g, 0, 1)

    out = (a * 255).astype(np.uint8)
    m = label_img[..., 3:] / 255
    out[-80:] = (out[-80:] * (1 - m) + label_img[..., :3] * m).astype(np.uint8)
    return out


def label():
    img = Image.new("RGBA", (W, 80), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((40, 30), LABEL, font=ImageFont.truetype(FONT, 18),
                             fill=(70, 70, 70, 200))
    return np.asarray(img, dtype=np.float32)


if __name__ == "__main__":
    base, lab = load(), label()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    for i in range(N):
        ff.stdin.write(frame(base, i, lab).tobytes())
    ff.stdin.close()
    ff.wait()
    print(OUT)
