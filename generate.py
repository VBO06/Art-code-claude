"""Generative glitch portraits in the spirit of DROITS RÉSERVÉS.

Each piece is fully determined by its seed and palette, so it can be
regenerated identically:  python3 generate.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1200, 1800          # working resolution (glitches are pixel-exact here)
SCALE = 2                  # final output 2400x3600, nearest-neighbour upscale
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


# ---------------------------------------------------------------- base figure

def blur(a, r):
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255


def shape_mask(draw_fn, r=0):
    img = Image.new("L", (W, H), 0)
    draw_fn(ImageDraw.Draw(img))
    m = np.asarray(img, dtype=np.float32) / 255
    return blur(m, r) if r else m


def sphere_shade(cx, cy, rx, ry, light):
    """Lambert shading of an ellipsoid, returns (shade, mask)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx - cx) / rx, (yy - cy) / ry
    d = nx ** 2 + ny ** 2
    nz = np.sqrt(np.clip(1 - d, 0, 1))
    lx, ly, lz = light
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    return shade, (d <= 1).astype(np.float32)


def figure(rng, pose):
    """A soft-lit bust: head, hair, neck, coat with lapels, closed eyes."""
    cx = W * pose["x"]
    tilt = pose["tilt"]
    light = np.array(pose["light"], dtype=np.float32)
    light /= np.linalg.norm(light)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = pose["bg"] + pose["bg_grad"] * (yy / H) + 0.04 * (xx / W)
    a = bg.copy()

    # coat
    hy = H * 0.30                      # head centre
    coat = shape_mask(lambda d: d.polygon([
        (cx - 560, H), (cx - 470, H * 0.58), (cx - 230, H * 0.47),
        (cx + 230, H * 0.47), (cx + 470, H * 0.58), (cx + 560, H)], fill=255), 6)
    coat_tone = 0.10 + 0.10 * np.clip((xx - (cx - 500)) / 1000, 0, 1)[::, ::-1] * light[0]
    a = a * (1 - coat) + coat_tone * coat
    # folds in the coat
    for _ in range(9):
        x0 = cx + rng.uniform(-480, 480)
        fold = shape_mask(lambda d: d.line(
            [(x0, H * rng.uniform(0.55, 0.7)), (x0 + rng.uniform(-90, 90), H)],
            fill=255, width=int(rng.uniform(6, 30))), 18)
        a += fold * coat * rng.uniform(-0.06, 0.10)

    # shirt V and lapels
    shirt = shape_mask(lambda d: d.polygon([
        (cx - 150, H * 0.47), (cx + 150, H * 0.47), (cx + 20, H * 0.74)], fill=255), 4)
    a = a * (1 - shirt) + (0.78 + 0.1 * (yy / H)) * shirt
    for side in (-1, 1):
        lapel = shape_mask(lambda d: d.polygon([
            (cx + side * 150, H * 0.47), (cx + side * 300, H * 0.52),
            (cx + side * 120, H * 0.66), (cx + side * 20, H * 0.75)], fill=255), 5)
        a = a * (1 - lapel) + (0.05 + 0.08 * (side * light[0] < 0)) * lapel

    # neck
    neck_shade, neck = sphere_shade(cx + tilt * 60, H * 0.44, 115, 150, light)
    neck = blur(neck, 4)
    a = a * (1 - neck) + (0.15 + 0.7 * neck_shade) * neck

    # head
    hx = cx + tilt * 120
    head_shade, head = sphere_shade(hx, hy, 230, 300, light)
    head = blur(head, 3)
    skin = 0.12 + 0.85 * head_shade ** 0.8
    a = a * (1 - head) + skin * head

    # hair mass
    hair = shape_mask(lambda d: d.chord(
        [hx - 260, hy - 330, hx + 260, hy + 120], 180 + tilt * 10, 360 + tilt * 10, fill=255), 10)
    hair *= (yy < hy - 40 + 120 * np.clip((np.abs(xx - hx) - 120) / 140, 0, 1)).astype(np.float32)
    hair = blur(hair, 6)
    a = a * (1 - hair) + (0.06 + 0.12 * head_shade) * hair

    # features: brows, closed eyes, nose shadow, lips
    def stroke(points, width, tone, r):
        m = shape_mask(lambda d: d.line(points, fill=255, width=width, joint="curve"), r)
        return a * (1 - m) + tone * m

    ey = hy + 10
    for s in (-1, 1):
        ex = hx + s * 95 + tilt * 25
        a = stroke([(ex - 55, ey - 45), (ex, ey - 58), (ex + 55, ey - 48)], 14, 0.12, 4)
        a = stroke([(ex - 45, ey), (ex, ey + 12), (ex + 45, ey + 2)], 6, 0.08, 2)
    a = stroke([(hx + tilt * 40 + 10, ey + 20), (hx + tilt * 40 + 25, ey + 120),
                (hx + tilt * 40 - 15, ey + 135)], 16, 0.18, 9)
    a = stroke([(hx - 60, ey + 200), (hx + tilt * 30, ey + 190), (hx + 60, ey + 200)], 12, 0.15, 5)
    # photographic texture: multi-scale noise, stronger in the mid-tones
    tex = sum(blur(rng.random((H, W)).astype(np.float32), r) * w
              for r, w in ((40, 2.5), (8, 1.2), (1.5, 0.6)))
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    a = a + tex * 0.05 * (0.3 + 4 * a * (1 - a))
    return np.clip(a, 0, 1)


def dither_grain(a, rng, amount):
    """Blend in a stochastic 1-bit dither, the speckled look of the series."""
    d = (rng.random(a.shape) < a).astype(np.float32)
    return a * (1 - amount) + d * amount


# ---------------------------------------------------------------- glitch ops

def rect(rng, wmin, wmax, hmin, hmax):
    w = int(rng.uniform(wmin, wmax) * W)
    h = int(rng.uniform(hmin, hmax) * H)
    x = int(rng.integers(0, W - w))
    y = int(rng.integers(0, H - h))
    return x, y, w, h


def pixel_sort(a, rng, n, lo=0.25, hi=0.85, vertical=True):
    """Sort runs of pixels whose brightness is in [lo, hi] (Asendorf style)."""
    src = a if vertical else a.T
    for _ in range(n):
        x, y, w, h = rect(rng, 0.03, 0.22, 0.08, 0.45) if vertical else rect(rng, 0.08, 0.45, 0.03, 0.18)
        if not vertical:
            x, y, w, h = y, x, h, w
        desc = rng.random() < 0.5
        for c in range(x, x + w):
            col = src[y:y + h, c]
            m = (col > lo) & (col < hi)
            edges = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
            for s, e in zip(edges[::2], edges[1::2]):
                seg = np.sort(col[s:e])
                col[s:e] = seg[::-1] if desc else seg
    return a


def smear(a, rng, n):
    """Repeat one row downwards (or one column sideways) -> streak bars."""
    for _ in range(n):
        x, y, w, h = rect(rng, 0.04, 0.25, 0.03, 0.2)
        if rng.random() < 0.65:
            a[y:y + h, x:x + w] = a[y, x:x + w][None, :]
        else:
            a[y:y + h, x:x + w] = a[y:y + h, x][:, None]
    return a


def stutter(a, rng, n):
    """Vertical streaks that keep a little of each column's jitter."""
    for _ in range(n):
        x, y, w, h = rect(rng, 0.01, 0.08, 0.04, 0.3)
        row = a[y, x:x + w].copy()
        jitter = rng.normal(0, 0.06, (h, w)).cumsum(0) * 0.08
        a[y:y + h, x:x + w] = np.clip(row[None, :] + jitter, 0, 1)
    return a


def displace(a, rng, n):
    for _ in range(n):
        x, y, w, h = rect(rng, 0.05, 0.35, 0.02, 0.15)
        dx = int(rng.integers(-W // 6, W // 6))
        dy = int(rng.integers(-H // 20, H // 20))
        sx = int(np.clip(x + dx, 0, W - w))
        sy = int(np.clip(y + dy, 0, H - h))
        a[y:y + h, x:x + w] = a[sy:sy + h, sx:sx + w].copy()
    return a


def mosaic(a, rng, n):
    for _ in range(n):
        x, y, w, h = rect(rng, 0.03, 0.15, 0.02, 0.1)
        b = int(rng.choice([6, 10, 16, 24]))
        blk = a[y:y + h, x:x + w]
        hh, ww = blk.shape[0] // b * b, blk.shape[1] // b * b
        if hh and ww:
            m = blk[:hh, :ww].reshape(hh // b, b, ww // b, b).mean((1, 3))
            blk[:hh, :ww] = np.repeat(np.repeat(m, b, 0), b, 1)
    return a


def solid_blocks(a, rng, n, tones):
    for _ in range(n):
        x, y, w, h = rect(rng, 0.015, 0.08, 0.015, 0.07)
        a[y:y + h, x:x + w] = rng.choice(tones)
    return a


def speckle(a, rng, n):
    """Stochastic-dither patches, like a failed decode."""
    for _ in range(n):
        x, y, w, h = rect(rng, 0.05, 0.25, 0.04, 0.2)
        blk = a[y:y + h, x:x + w]
        blk[:] = (rng.random(blk.shape) < blk).astype(np.float32) * 0.9 + 0.05
    return a


# ---------------------------------------------------------------- colour

def gradient_map(a, stops):
    """stops: list of (position, (r, g, b))."""
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float32) / 255
    out = np.stack([np.interp(a, pos, cols[:, i]) for i in range(3)], -1)
    return out


def finish(rgb, rng, grain, label):
    big = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize(
        (W * SCALE, H * SCALE), Image.NEAREST)
    arr = np.asarray(big, dtype=np.float32) / 255
    mono = rng.normal(0, grain, arr.shape[:2])[..., None]
    arr = np.clip(arr + mono + rng.normal(0, grain * 0.25, arr.shape), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 30)
    lum = arr[-120:-40, 80:900].mean()
    ink = (40, 40, 40) if lum > 0.55 else (150, 150, 150)
    d.text((80, H * SCALE - 100), label, font=font, fill=ink)
    return img


# ---------------------------------------------------------------- pieces

def spaced(s):
    return " ".join(s)


PIECES = [
    dict(
        num=13, seed=1313, title="SILENCE",
        pose=dict(x=0.52, tilt=-0.4, light=(-0.6, -0.4, 0.7), bg=0.86, bg_grad=-0.1),
        ops=dict(sort=26, smear=14, stutter=40, displace=12, mosaic=10, speckle=6),
        blocks=[0.02, 0.95], n_blocks=3, dither=0.22,
        stops=[(0, (8, 8, 8)), (0.5, (120, 120, 120)), (1, (246, 246, 244))],
        grain=0.07,
    ),
    dict(
        num=14, seed=1414, title="BRAISE",
        pose=dict(x=0.47, tilt=0.5, light=(0.7, -0.3, 0.6), bg=0.05, bg_grad=0.08),
        ops=dict(sort=34, smear=18, stutter=34, displace=16, mosaic=14, speckle=4),
        blocks=[0.0, 0.55, 0.75], n_blocks=5, dither=0.18,
        stops=[(0, (6, 2, 2)), (0.3, (70, 6, 8)), (0.6, (196, 22, 24)),
               (0.85, (236, 120, 96)), (1, (250, 228, 214))],
        grain=0.06,
    ),
    dict(
        num=15, seed=1515, title="PEAU",
        pose=dict(x=0.5, tilt=0.15, light=(-0.2, -0.6, 0.75), bg=0.93, bg_grad=-0.05),
        ops=dict(sort=30, smear=16, stutter=30, displace=14, mosaic=12, speckle=3),
        blocks=[0.42, 0.42, 0.98, 0.2], n_blocks=6, dither=0.15,
        stops=[(0, (40, 18, 22)), (0.25, (150, 60, 56)), (0.42, (214, 34, 38)),
               (0.6, (226, 150, 138)), (0.8, (240, 206, 194)), (1, (250, 244, 236))],
        grain=0.05,
    ),
]


def make(p):
    rng = np.random.default_rng(p["seed"])
    a = figure(rng, p["pose"])
    o = p["ops"]
    a = displace(a, rng, o["displace"])
    a = pixel_sort(a, rng, o["sort"])
    a = pixel_sort(a, rng, o["sort"] // 4, vertical=False)
    a = stutter(a, rng, o["stutter"])
    a = smear(a, rng, o["smear"])
    a = mosaic(a, rng, o["mosaic"])
    a = solid_blocks(a, rng, p["n_blocks"], p["blocks"])
    a = speckle(a, rng, o["speckle"])
    a = dither_grain(a, rng, p["dither"])
    rgb = gradient_map(a, p["stops"])
    label = f"{spaced('DROITS')}  {spaced('RÉSERVÉS')}  #{p['num']}   ·   {spaced(p['title'])}"
    return finish(rgb, rng, p["grain"], label)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for p in PIECES:
        path = os.path.join(OUT, f"droits-reserves-{p['num']}-{p['title'].lower()}.png")
        make(p).save(path, optimize=True)
        print(path)
