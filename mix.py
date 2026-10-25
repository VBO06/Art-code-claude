"""Tours et paysages: filtres glitch sur les photos de source/villes.

    python3 mix.py            -> output/villes/*.png
    python3 mix.py trame      -> une seule pièce
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "source", "villes")
OUT = os.path.join(ROOT, "output", "villes")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W, H = 1500, 2000
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ---------------------------------------------------------------- io

def load(name, box=(0, 0, 1, 1)):
    """Load a photo, crop by fractional box (x0, y0, x1, y1), fit to W x H."""
    img = Image.open(os.path.join(SRC, name)).convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = box
    img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    # centre-crop to 3:4, then resize
    w, h = img.size
    if w / h > W / H:
        nw = int(h * W / H)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w * H / W)
        img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return np.asarray(img.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255


def lum(a):
    return a @ LUMA


def blur(m, r):
    img = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255


def save(a, name, label, ink=(235, 235, 235)):
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    ImageDraw.Draw(img).text((50, H - 60), label, font=ImageFont.truetype(FONT, 20), fill=ink)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    img.save(path, optimize=True)
    print(path)


# ---------------------------------------------------------------- effects (RGB)

def rect(rng, wmin, wmax, hmin, hmax, area=(0, 0, 1, 1)):
    ax0, ay0, ax1, ay1 = (int(area[0] * W), int(area[1] * H), int(area[2] * W), int(area[3] * H))
    w = max(2, int(rng.uniform(wmin, wmax) * W))
    h = max(2, int(rng.uniform(hmin, hmax) * H))
    x = int(rng.integers(ax0, max(ax0 + 1, ax1 - w)))
    y = int(rng.integers(ay0, max(ay0 + 1, ay1 - h)))
    return x, y, w, h


def sort_runs(a, x, y, w, h, lo, hi, vertical=True, min_run=8):
    """Asendorf pixel sort: sort runs whose brightness is within [lo, hi]."""
    reg = a[y:y + h, x:x + w] if vertical else a[y:y + h, x:x + w].transpose(1, 0, 2)
    L = lum(reg)
    for c in range(reg.shape[1]):
        m = (L[:, c] > lo) & (L[:, c] < hi)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], m.astype(np.int8), [0]))))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_run:
                idx = np.argsort(L[s:e, c])
                reg[s:e, c] = reg[s:e, c][idx]


def pixel_sort(a, rng, n, lo, hi, vertical=True, area=(0, 0, 1, 1)):
    for _ in range(n):
        if vertical:
            x, y, w, h = rect(rng, 0.03, 0.2, 0.15, 0.5, area)
        else:
            x, y, w, h = rect(rng, 0.2, 0.7, 0.01, 0.08, area)
        sort_runs(a, x, y, w, h, lo, hi, vertical)


def smear(a, rng, n, vertical=True, area=(0, 0, 1, 1)):
    for _ in range(n):
        if vertical:
            x, y, w, h = rect(rng, 0.01, 0.08, 0.05, 0.3, area)
            a[y:y + h, x:x + w] = a[y, x:x + w][None]
        else:
            x, y, w, h = rect(rng, 0.05, 0.4, 0.004, 0.03, area)
            a[y:y + h, x:x + w] = a[y:y + h, x][:, None]


def slices(a, rng, n, amount, area=(0, 0, 1, 1)):
    for _ in range(n):
        x, y, w, h = rect(rng, 1, 1, 0.003, 0.03, area)
        a[y:y + h] = np.roll(a[y:y + h], int(rng.integers(-amount, amount)), 1)


def rgb_split(a, dx):
    a[..., 0] = np.roll(a[..., 0], dx, 1)
    a[..., 2] = np.roll(a[..., 2], -dx, 1)


def mosaic(a, rng, n, area=(0, 0, 1, 1)):
    for _ in range(n):
        b = int(rng.choice([10, 16, 24]))
        x, y, w, h = rect(rng, 0.04, 0.15, 0.02, 0.08, area)
        w, h = w // b * b, h // b * b
        blk = a[y:y + h, x:x + w]
        m = blk.reshape(h // b, b, w // b, b, 3).mean((1, 3))
        blk[:] = np.repeat(np.repeat(m, b, 0), b, 1)


def grain(a, rng, s):
    return np.clip(a + rng.normal(0, s, (H, W, 1)).astype(np.float32), 0, 1)


def grade(a, contrast=1.1, sat=1.0, lift=0.0):
    a = (a - 0.5) * contrast + 0.5 + lift
    g = lum(a)[..., None]
    return np.clip(g + (a - g) * sat, 0, 1)


def mono(a, stops):
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float32) / 255
    L = lum(a)
    return np.stack([np.interp(L, pos, cols[:, i]) for i in range(3)], -1)


def sky_mask(a, r=3):
    """Blue, fairly bright pixels: the sky (and its clouds)."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = ((B - R > 0.12) & (lum(a) > 0.3)) | ((lum(a) > 0.78) & (B >= R))
    return blur(m.astype(np.float32), r)


# ---------------------------------------------------------------- pieces

def bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


def trame():
    """Filtre seul: trame 1-bit ordonnée, bleu nuit et blanc."""
    rng = np.random.default_rng(606)
    a = grade(load("07.jpg", (0, 0, 0.92, 1)), 1.25)
    pixel_sort(a, rng, 8, 0.4, 0.9, area=(0.1, 0.1, 0.9, 0.9))
    L = blur(lum(a), 0.6)
    t = np.tile(bayer(8), (H // 8 + 1, W // 8 + 1))[:H, :W]
    on = (L > t).astype(np.float32)[..., None]
    ink, paper = np.array([12, 22, 60]) / 255, np.array([244, 242, 236]) / 255
    a = ink * (1 - on) + paper * on
    save(a, "26-trame", "DROITS RÉSERVÉS #26  ·  TRAME", ink=(12, 22, 60))


def neige():
    """Filtre seul: palette noir / rouge / crème de la série, la neige coule."""
    rng = np.random.default_rng(707)
    a = grade(load("11.jpg"), 1.15)
    pixel_sort(a, rng, 10, 0.6, 0.98, area=(0.15, 0.25, 0.9, 0.5))
    smear(a, rng, 6, area=(0, 0.5, 1, 1))
    a = mono(a, [(0, (6, 2, 2)), (0.3, (70, 6, 8)), (0.55, (190, 24, 26)),
                 (0.8, (236, 150, 128)), (1, (250, 236, 224))])
    a = grain(a, rng, 0.04)
    save(a, "27-neige", "DROITS RÉSERVÉS #27  ·  NEIGE")


def grille():
    """Effet seul, noir et blanc: la façade fond en pixels."""
    rng = np.random.default_rng(404)
    a = load("05.jpg", (0, 0, 1, 0.8))
    pixel_sort(a, rng, 22, 0.3, 0.88)
    smear(a, rng, 16)
    slices(a, rng, 8, 60)
    mosaic(a, rng, 6)
    a = mono(a, [(0, (6, 6, 6)), (0.35, (55, 55, 55)), (0.7, (200, 200, 198)), (1, (250, 250, 248))])
    d = (rng.random((H, W)) < lum(a)).astype(np.float32)[..., None]
    a = a * 0.82 + d * 0.18
    a = grain(a, rng, 0.05)
    save(a, "22-grille", "DROITS RÉSERVÉS #22  ·  GRILLE")


def arc():
    """Effet seul, couleur: l'arc-en-ciel coule sur le parking."""
    rng = np.random.default_rng(505)
    a = grade(load("02.jpg"), 1.2, 1.6)
    pixel_sort(a, rng, 34, 0.5, 0.85, area=(0, 0, 1, 0.62))
    smear(a, rng, 18, area=(0.05, 0.2, 0.95, 1))
    rgb_split(a, 5)
    smear(a, rng, 8, vertical=False, area=(0, 0.6, 1, 1))
    slices(a, rng, 7, 50, area=(0, 0.55, 1, 1))
    a = grain(a, rng, 0.035)
    save(a, "23-arc", "DROITS RÉSERVÉS #23  ·  ARC")


def censor(a, box, b=22):
    """Pixellise une zone précise (enseignes, logos): la marque disparaît dans le glitch."""
    x0, y0, x1, y1 = int(box[0] * W) // b * b, int(box[1] * H) // b * b, int(box[2] * W), int(box[3] * H)
    w, h = max(b, (x1 - x0) // b * b + b), max(b, (y1 - y0) // b * b + b)
    blk = a[y0:y0 + h, x0:x0 + w]
    m = blk.reshape(h // b, b, w // b, b, 3).mean((1, 3))
    blk[:] = np.repeat(np.repeat(m, b, 0), b, 1)


def arve():
    """La rivière au crépuscule: les nuages coulent sur la montagne, l'eau file en stries."""
    rng = np.random.default_rng(808)
    a = grade(load("14.jpg"), 1.15, 1.1)
    pixel_sort(a, rng, 7, 0.65, 0.95, area=(0, 0.15, 1, 0.42))
    pixel_sort(a, rng, 18, 0.35, 0.8, vertical=False, area=(0.15, 0.56, 1, 1))
    smear(a, rng, 10, vertical=False, area=(0.15, 0.6, 1, 1))
    slices(a, rng, 5, 40, area=(0, 0.55, 1, 1))
    a = grain(a, rng, 0.03)
    save(a, "28-arve", "DROITS RÉSERVÉS #28  ·  ARVE")


def chantier():
    """Plainpalais: les enseignes sont effacées, la lumière des lampadaires coule."""
    rng = np.random.default_rng(909)
    a = grade(load("15.jpg"), 1.12, 1.15)
    for box in ((0.12, 0.405, 0.24, 0.45), (0.16, 0.282, 0.225, 0.302), (0.02, 0.425, 0.095, 0.45)):
        censor(a, box)
    pixel_sort(a, rng, 8, 0.5, 0.85, area=(0, 0.1, 1, 0.42))
    smear(a, rng, 8, area=(0.4, 0.18, 0.7, 0.7))
    pixel_sort(a, rng, 16, 0.25, 0.65, vertical=False, area=(0, 0.75, 1, 1))
    rgb_split(a, 3)
    a = grain(a, rng, 0.03)
    save(a, "29-chantier", "DROITS RÉSERVÉS #29  ·  CHANTIER")


def facades():
    """Les immeubles: chaque étage glisse, le sol rouge s'étire."""
    rng = np.random.default_rng(1010)
    a = grade(load("16.jpg", (0, 0, 0.9, 1)), 1.12, 1.1)
    censor(a, (0.5, 0.6, 0.78, 0.635), 16)
    slices(a, rng, 10, 90, area=(0, 0.25, 1, 0.62))
    pixel_sort(a, rng, 10, 0.5, 0.92, area=(0, 0, 1, 0.26))
    pixel_sort(a, rng, 22, 0.15, 0.5, vertical=False, area=(0, 0.68, 1, 1))
    smear(a, rng, 6, area=(0.05, 0.3, 0.95, 0.68))
    a = grain(a, rng, 0.03)
    save(a, "30-facades", "DROITS RÉSERVÉS #30  ·  FAÇADES")


PIECES = dict(grille=grille, arc=arc, trame=trame, neige=neige,
              arve=arve, chantier=chantier, facades=facades)

if __name__ == "__main__":
    for name in sys.argv[1:] or PIECES:
        PIECES[name]()
