"""DROITS RÉSERVÉS #40 MERVEILLES — la piscine de Palm Springs revue à la Tim Burton:
crépuscule violet et grande lune, transats coiffés de hauts-de-forme (le thé du
Chapelier), pot rayé, fleurs à cœur spiralé, vitrail lumineux, lucioles.

    python3 burton.py      -> output/piscines/40-merveilles-*.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import la
import mix as m
import piscines as p

S = la.S
K = 2                                    # vector layer drawn at 2x, then downsampled
INK = (18, 12, 26, 255)
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)


def rot(pts, cx, cy, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return [((x - cx) * c - (y - cy) * s + cx, (x - cx) * s + (y - cy) * c + cy) for x, y in pts]


def ellipse_pts(cx, cy, rx, ry, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


def sc(pts):
    return [(x * K, y * K) for x, y in pts]


def spiral_pts(cx, cy, size, turns=2.5, flip=1):
    th = np.linspace(0, turns * 2 * np.pi, 160)
    r = size * th / th[-1]
    return list(zip(cx + flip * r * np.cos(th), cy - r * np.sin(th)))


def hat(d, cx, cy, w, h, tilt, band, stripes=False, flower=None):
    """A crooked Mad Hatter top hat standing on (cx, cy)."""
    bend = w * 0.18
    crown = [(cx - w * 0.42, cy), (cx - w * 0.55 + bend, cy - h), (cx + w * 0.55 + bend, cy - h),
             (cx + w * 0.42, cy)]
    brim = ellipse_pts(cx, cy, w * 0.78, w * 0.16)
    top = ellipse_pts(cx + bend, cy - h, w * 0.55, w * 0.12)
    band_poly = [(cx - w * 0.44, cy - h * 0.08), (cx - w * 0.46 + bend * 0.25, cy - h * 0.3),
                 (cx + w * 0.46 + bend * 0.25, cy - h * 0.3), (cx + w * 0.44, cy - h * 0.08)]
    R = lambda pts: sc(rot(pts, cx, cy, tilt))
    d.polygon(R(brim), fill=(28, 20, 40, 255), outline=INK, width=4 * K)
    d.polygon(R(crown), fill=(34, 26, 52, 255), outline=INK, width=4 * K)
    if stripes:
        for k in range(1, 6):
            f = 0.3 + k * 0.12
            y = cy - h * f
            seg = [(cx - w * (0.44 + 0.11 * f) + bend * f, y), (cx + w * (0.44 + 0.11 * f) + bend * f, y)]
            d.line(R(seg), fill=(225, 220, 235, 255), width=int(w * 0.05) * K)
    d.polygon(R(band_poly), fill=band + (255,), outline=INK, width=3 * K)
    d.polygon(R(top), fill=(48, 38, 70, 255), outline=INK, width=4 * K)
    if flower:
        fx, fy = rot([(cx + w * 0.38, cy - h * 0.22)], cx, cy, tilt)[0]
        bloom(d, fx, fy, w * 0.16, flower)


def bloom(d, cx, cy, r, colour, petals=7):
    """A whimsical flower: pointed petals and a spiral heart."""
    for i in range(petals):
        a = 2 * np.pi * i / petals
        tip = (cx + np.cos(a) * r * 1.9, cy + np.sin(a) * r * 1.9)
        l = (cx + np.cos(a + 0.45) * r * 0.9, cy + np.sin(a + 0.45) * r * 0.9)
        rr = (cx + np.cos(a - 0.45) * r * 0.9, cy + np.sin(a - 0.45) * r * 0.9)
        d.polygon(sc([(cx, cy), l, tip, rr]), fill=colour + (255,), outline=INK, width=2 * K)
    d.ellipse([(cx - r * 0.7) * K, (cy - r * 0.7) * K, (cx + r * 0.7) * K, (cy + r * 0.7) * K],
              fill=(250, 235, 160, 255), outline=INK, width=2 * K)
    d.line(sc(spiral_pts(cx, cy, r * 0.55, 2.2)), fill=INK, width=2 * K)


def stalk(d, x0, y0, height, lean, colour, r, rng):
    """A curly stem growing from (x0, y0), a leaf, and a flower on top."""
    t = np.linspace(0, 1, 40)
    xs = x0 + lean * height * t ** 1.5 + np.sin(t * 5 + rng.uniform(0, 6)) * height * 0.06
    ys = y0 - height * t
    d.line(sc(list(zip(xs, ys))), fill=(20, 60, 40, 255), width=4 * K, joint="curve")
    lx, ly = xs[15], ys[15]
    side = 1 if rng.random() < 0.5 else -1
    leaf = [(lx, ly), (lx + side * r * 1.6, ly - r * 0.6), (lx + side * r * 2.2, ly - r * 1.4),
            (lx + side * r * 0.8, ly - r * 0.6)]
    d.polygon(sc(leaf), fill=(70, 150, 90, 255), outline=INK, width=2 * K)
    d.line(sc(spiral_pts(lx + side * r * 2.2, ly - r * 1.4, r * 0.5, 1.6, side)), fill=INK, width=2 * K)
    bloom(d, xs[-1], ys[-1], r, colour)


def sky(rng):
    """Violet twilight, a huge pale moon, curly clouds, stars and a few bats."""
    g = np.clip(YY / 760, 0, 1)[..., None]
    a = np.array([22, 16, 52], np.float32) / 255 * (1 - g) + np.array([128, 86, 150], np.float32) / 255 * g
    mx, my, mr = 560, 250, 175
    dist = np.sqrt((XX - mx) ** 2 + (YY - my) ** 2)
    halo = np.clip(1 - (dist - mr) / 260, 0, 1)[..., None] ** 2 * 0.35
    a = a + halo * np.array([0.9, 0.85, 1.0])
    moon = np.clip((mr - dist) / 2.5, 0, 1)[..., None]
    shade = 0.88 + 0.12 * np.clip((XX - mx + YY - my) / (2 * mr), -1, 1)[..., None]
    crater = sum(np.clip((r - np.sqrt((XX - x) ** 2 + (YY - y) ** 2)) / 3, 0, 1)
                 for x, y, r in ((500, 200, 32), (620, 300, 22), (600, 180, 14), (480, 300, 18)))[..., None]
    lunar = np.array([0.96, 0.94, 0.88]) * shade * (1 - 0.1 * crater)
    a = a * (1 - moon) + lunar * moon
    for _ in range(140):                                     # stars
        x, y = int(rng.uniform(0, S)), int(rng.uniform(0, 520))
        if np.hypot(x - mx, y - my) > mr + 20:
            a[y:y + 2, x:x + 2] = rng.uniform(0.7, 1.0)
    return np.clip(a, 0, 1)


def merveilles(seed):
    rng = np.random.default_rng(seed)
    src = la.load("06.png", (0, 425, 923, 1343))
    graded = m.grade(src.copy(), 1.15, 1.3)
    labels = p.panes(graded)

    # 1. Burton night grade of the scene; the pool keeps its glow
    L = la.lum(src)[..., None]
    night = src * 0.45 + L * 0.35
    night = np.clip(night * np.array([0.72, 0.74, 1.0]) + np.array([0.04, 0.02, 0.09]), 0, 1)
    R, G, B = src[..., 0], src[..., 1], src[..., 2]
    pool = m.blur(((B - R > 0.25) & (la.lum(src) > 0.45) & (YY > 1000)).astype(np.float32), 3)[..., None]
    night = night * (1 - pool * 0.75) + np.clip(src * np.array([0.6, 1.0, 1.1]), 0, 1) * pool * 0.75

    # 2. the sky
    # sky = clear-blue (or cloud-white) pixels reachable from the top edge
    lumv = la.lum(src)
    cond = (((B > 0.62) & (B - R > 0.16)) | ((lumv > 0.75) & (YY < 330))) & (YY < 800)
    reach = np.zeros_like(cond)
    reach[0] = cond[0]
    while True:
        grow = reach.copy()
        grow[1:] |= reach[:-1]
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow &= cond
        if (grow == reach).all():
            break
        reach = grow
    # swallow the thin halo where sky meets roofs and ridge
    near = m.blur(reach.astype(np.float32), 3) > 0.35
    skym = (reach | (near & (B - R > 0.06) & (YY < 800)))
    skym &= ~(((XX < 225) & (YY > 476)) | ((XX < 470) & (YY > 532)))              # keep the building on the left
    skym = skym.astype(np.float32)
    skym = m.blur(skym, 2)[..., None]
    a = night * (1 - skym) + sky(rng) * skym

    # 3. the stained glass glows, its light falls on the terrace
    lit = p.lumiere(a, labels, np.random.default_rng(seed + 1), light=0.9, water=0.5)
    col, cov = p.pane_layer(labels, np.random.default_rng(seed + 1))
    glow = la.blur_rgb(col * cov[..., None], 22)
    a = np.clip(lit + glow * 0.45, 0, 1)

    # 4. the pot becomes a striped character
    pot = ((np.abs(XX - 1100) < 108 - 14 * np.clip((YY - 1230) / 330, 0, 1)) &
           (YY > 1225) & (YY < 1555)).astype(np.float32)
    curve = (YY - 1225) + ((XX - 1100) / 108) ** 2 * 26
    stripe = (np.sin(curve / 22) > 0).astype(np.float32)
    shade = 0.55 + 0.45 * np.clip(1 - np.abs(XX - 1070) / 120, 0, 1)
    potc = (stripe * 0.92 + (1 - stripe) * 0.08)[..., None] * shade[..., None] * np.array([0.95, 0.93, 1.0])
    pot = m.blur(pot, 1.2)[..., None]
    a = a * (1 - pot) + potc * pot

    # 5. vector layer: hats, flowers, curly clouds
    layer = Image.new("RGBA", (S * K, S * K), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for cx, cy, r, flip in ((1120, 470, 70, 1), (1500, 380, 55, -1), (250, 330, 60, 1)):
        d.line(sc(spiral_pts(cx, cy, r, 2.3, flip) + [(cx - flip * r * 2.5, cy + r * 0.2)]),
               fill=(235, 225, 245, 200), width=5 * K, joint="curve")
    palette = [(235, 80, 150), (150, 90, 220), (40, 190, 175), (255, 140, 40), (230, 60, 60), (250, 200, 70)]
    for x0, y0, hgt, lean in ((60, 1240, 210, 0.2), (130, 1235, 150, -0.1), (600, 1610, 170, 0.15),
                              (860, 1680, 220, -0.2), (940, 1690, 140, 0.25), (1280, 1600, 200, 0.1),
                              (1440, 1660, 150, -0.15), (1700, 1560, 190, -0.1), (470, 1520, 130, 0.2),
                              (1560, 1110, 120, 0.1)):
        stalk(d, x0, y0, hgt, lean, palette[int(rng.integers(len(palette)))], rng.uniform(16, 26), rng)
    hat(d, 1112, 1232, 150, 190, -8, (150, 90, 220), stripes=True, flower=(235, 80, 150))
    hat(d, 395, 1168, 88, 105, -14, (235, 80, 150), flower=(250, 200, 70))
    hat(d, 742, 1210, 80, 120, 10, (40, 190, 175))
    hat(d, 1290, 1182, 96, 118, -6, (255, 140, 40), stripes=True, flower=(150, 90, 220))
    vec = layer.resize((S, S), Image.LANCZOS)
    v = np.asarray(vec, np.float32) / 255
    a = a * (1 - v[..., 3:]) + v[..., :3] * v[..., 3:]

    # 6. fireflies and a soft vignette
    core = Image.new("L", (S, S), 0)
    dc = ImageDraw.Draw(core)
    for _ in range(70):
        x, y, r = rng.uniform(0, S), rng.uniform(600, 1750), rng.uniform(2.5, 6)
        dc.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(170, 255)))
    halo = np.asarray(core.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255 * 3.5
    dot = np.asarray(core.filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
    a = a + np.clip(halo * 0.6 + dot, 0, 1)[..., None] * np.array([1.0, 0.9, 0.55]) * 0.8
    vig = 1 - 0.35 * (((XX / S - 0.5) ** 2 + (YY / S - 0.5) ** 2) * 2.2)
    return la.grain(np.clip(a * vig[..., None], 0, 1), rng, 0.02)


if __name__ == "__main__":
    out_dir = os.path.join(la.ROOT, "output", "piscines")
    for k, seed in enumerate((11, 12, 13)):
        a = merveilles(seed)
        stem = os.path.join(out_dir, f"40-merveilles-{k + 1}")
        Image.fromarray((a * 255).astype(np.uint8)).save(stem + ".png", optimize=True)
        la.poster(a, "40", "MERVEILLES", "PALM SPRINGS", "33.82° N  116.54° W",
                  "CONTE / VITRAIL").save(stem + "-affiche.png", optimize=True)
        print(stem)
