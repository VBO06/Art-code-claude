"""DROITS RÉSERVÉS — série PISCINES: le langage de #40 (couleurs d'origine saturées,
coulures, voiles pastel translucides) appliqué à d'autres photos.

    python3 piscines.py          -> output/piscines/*.png (+ -affiche) + planches
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import la
import mix as m

S = la.S
OUT = os.path.join(la.ROOT, "output", "piscines")
PASTELS = [(255, 150, 190), (255, 210, 80), (80, 200, 220), (255, 255, 255)]


def load_jpg(path, box=(0, 0, 1, 1)):
    """A full-resolution photo from source/, cropped (fractional box) to a centred square."""
    img = Image.open(os.path.join(la.ROOT, "source", path)).convert("RGB")
    w, h = img.size
    img = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((S, S), Image.LANCZOS), np.float32) / 255


def censor(a, boxes, b=16):
    """Pixelate shop signs (fractional boxes) on a square S x S image."""
    for x0, y0, x1, y1 in boxes:
        x0, y0 = int(x0 * S) // b * b, int(y0 * S) // b * b
        w, h = (int(x1 * S) - x0) // b * b + b, (int(y1 * S) - y0) // b * b + b
        blk = a[y0:y0 + h, x0:x0 + w]
        blk[:] = np.repeat(np.repeat(blk.reshape(h // b, b, w // b, b, 3).mean((1, 3)), b, 0), b, 1)
    return a


def piscine(a, rng, sort_area, smear_area, keep=None):
    """The #40 recipe."""
    a = m.grade(a, 1.15, 1.3)
    clean = a.copy()
    m.pixel_sort(a, rng, 12, 0.55, 0.95, vertical=False, area=sort_area)
    m.smear(a, rng, 6, area=smear_area)
    a = la.veils(a, rng, PASTELS)
    if keep:
        ys, xs = slice(int(keep[1] * S), int(keep[3] * S)), slice(int(keep[0] * S), int(keep[2] * S))
        a[ys, xs] = clean[ys, xs]
    return la.grain(a, rng, 0.025)


# num, stem, title, place, coords, loader, sort area, smear area, area kept clean
PIECES = [
    (60, "monte-carlo", "PISCINE II", "MONTE-CARLO", "43.74° N  7.42° E",
     lambda: la.load("10.png", (0, 358, 923, 1280)), (0, 0.5, 1, 0.85), (0, 0.05, 1, 0.5), None),
    (61, "eze", "JARDIN", "ÈZE", "43.73° N  7.36° E",
     lambda: la.load("09.png", (0, 507, 923, 1428), patch=[(812, 530, 900, 600, "down")]),
     (0, 0.55, 1, 0.75), (0, 0, 1, 0.5), None),
    (62, "downtown", "MODERNE", "DOWNTOWN LOS ANGELES", "34.05° N  118.25° W",
     lambda: la.load("01.png", (0, 700, 923, 1525),
                     hide=[(740, 700, 830, 790), (120, 895, 175, 925), (180, 1385, 250, 1480)]),
     (0, 0.75, 1, 0.95), (0.3, 0.05, 0.65, 0.7), None),
    (63, "ascension", "ASCENSION II", "DOWNTOWN LOS ANGELES", "34.05° N  118.26° W",
     lambda: la.load("15.png", (0, 725, 923, 1648)), (0, 0.7, 1, 1), (0.1, 0, 0.7, 0.5),
     (0.5, 0.52, 0.72, 1)),
    (64, "houston", "VERRE", "HOUSTON", "29.76° N  95.37° W",
     lambda: load_jpg("villes/17.jpg", (0, 0, 1, 0.82)), (0, 0.6, 1, 0.95), (0.05, 0.1, 0.95, 0.8), None),
    (65, "geneve", "FAÇADES II", "GENÈVE", "46.20° N  6.14° E",
     lambda: censor(load_jpg("villes/16.jpg", (0, 0.2, 0.9, 0.85)),
                    [(0.0, 0.62, 0.3, 0.68), (0.5, 0.64, 0.84, 0.7), (0.6, 0.66, 0.68, 0.74)]), (0, 0.7, 1, 1), (0, 0.15, 1, 0.6), None),
]


def sheet(items, name, cols):
    t = 500
    rows = (len(items) + cols - 1) // cols
    img = Image.new("RGB", (cols * t, rows * (t + 50)), (242, 239, 232))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(la.FONTS + "Bold.ttf", 26)
    for i, (label, a) in enumerate(items):
        x, y = (i % cols) * t, (i // cols) * (t + 50)
        d.text((x + 16, y + 12), label, font=font, fill=(24, 24, 24))
        img.paste(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((t, t)), (x, y + 50))
    img.save(os.path.join(OUT, name), quality=90)


def save(a, stem, num, title, place, coords):
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(OUT, stem + ".png"), optimize=True)
    la.poster(a, str(num), title, place, coords, "PIXEL SORT / VOILES").save(
        os.path.join(OUT, stem + "-affiche.png"), optimize=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    # 1. six compositions of #40 (same photo and effects, rectangles placed differently)
    src = la.load("06.png", (0, 425, 923, 1343))
    variants = []
    for k in range(6):
        a = piscine(src.copy(), np.random.default_rng(4001 + 100 * k), (0.5, 0.55, 1, 0.85), (0, 0.2, 1, 0.7))
        save(a, f"40-piscine-composition-{k + 1}", 40, "PISCINE", "PALM SPRINGS", "33.82° N  116.54° W")
        variants.append((f"COMPOSITION {k + 1}", a))
    sheet(variants, "planche-compositions-40.jpg", 3)

    # 2. the series
    series = []
    for num, stem, title, place, coords, load, sort_area, smear_area, keep in PIECES:
        a = piscine(load(), np.random.default_rng(num * 101), sort_area, smear_area, keep)
        save(a, f"{num}-{stem}", num, title, place, coords)
        series.append((f"#{num} {title}", a))
    sheet(series, "planche-serie.jpg", 3)
    print(OUT)
