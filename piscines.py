"""DROITS RÉSERVÉS — série PISCINES: le langage de #40 (couleurs d'origine saturées,
coulures, voiles pastel translucides) appliqué à d'autres photos.

    python3 piscines.py          -> output/piscines/*.png (+ -affiche) + planches
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import la
import mix as m

S = la.S
OUT = os.path.join(la.ROOT, "output", "piscines")
PASTELS = [(255, 150, 190), (255, 210, 80), (80, 200, 220), (255, 255, 255)]
RAINBOW = [(255, 150, 190), (255, 210, 80), (80, 200, 220), (255, 255, 255), (255, 90, 60),
           (255, 140, 30), (120, 220, 120), (20, 150, 90), (60, 110, 240), (150, 110, 220),
           (230, 60, 160), (200, 185, 235), (40, 190, 175), (214, 32, 38)]


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


def grid(a, rng, tones, cell=150, share=0.4, op=(0.3, 0.6)):
    """Cut the image into squares; a share of them gets a translucent colour."""
    a = a.copy()
    for y in range(0, S, cell):
        for x in range(0, S, cell):
            if rng.random() < share:
                c = np.array(tones[int(rng.integers(len(tones)))], np.float32) / 255
                o = rng.uniform(*op)
                a[y:y + cell, x:x + cell] = a[y:y + cell, x:x + cell] * (1 - o) + c * o
    a[::cell] *= 0.85                                   # thin seams between squares
    a[:, ::cell] *= 0.85
    return a


# the fence between the terrace and the pool in #40 (full-frame pixel polygon)
FENCE = [(915, 800), (1800, 828), (1800, 1082), (1530, 1076), (918, 1232)]
PANES = [(255, 90, 60), (255, 210, 80), (40, 190, 175), (230, 60, 160), (120, 220, 120),
         (60, 110, 240), (255, 140, 30), (150, 110, 220), (255, 150, 190), (20, 150, 90),
         (214, 32, 38), (80, 200, 220)]


def panes(clean, f=3, dark=0.24, min_area=40):
    """Label the panes of the fence: light regions inside FENCE split by the black frame.

    Returns an S x S int map (0 = not a pane, k = pane k)."""
    poly = Image.new("L", (S, S), 0)
    ImageDraw.Draw(poly).polygon(FENCE, fill=1)
    inside = np.asarray(poly, bool)
    light = (la.lum(clean) > dark) & inside
    g = light[::f, ::f]
    lab = np.zeros(g.shape, np.int32)
    n = 0
    for y0, x0 in zip(*np.nonzero(g)):
        if lab[y0, x0]:
            continue
        n += 1
        stack, pix = [(y0, x0)], []
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < g.shape[0] and 0 <= xx < g.shape[1] and g[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        if len(pix) < min_area:
            for y, x in pix:
                lab[y, x] = -1
    lab[lab < 0] = 0
    full = np.repeat(np.repeat(lab, f, 0), f, 1)[:S, :S]
    return np.where(light, full, 0)


def colour_panes(a, labels, rng, op=0.62):
    a = a.copy()
    ids = [k for k in np.unique(labels) if k]
    order = rng.permutation(len(PANES))
    L = la.lum(a)[..., None]
    for i, k in enumerate(ids):
        m_ = m.blur((labels == k).astype(np.float32), 1.0)[..., None] * op
        c = np.array(PANES[order[i % len(PANES)]], np.float32) / 255
        a = a * (1 - m_) + np.clip(c * (0.45 + 0.8 * L), 0, 1) * m_
    return a


def pane_layer(labels, rng):
    """Colour layer (S x S x 3) and coverage (S x S) of the coloured panes."""
    col = np.zeros((S, S, 3), np.float32)
    order = rng.permutation(len(PANES))
    for i, k in enumerate(k for k in np.unique(labels) if k):
        col[labels == k] = np.array(PANES[order[i % len(PANES)]], np.float32) / 255
    return col, (labels > 0).astype(np.float32)


def lumiere(photo, labels, rng, axis=1150, light=0.75, water=0.45):
    """The panes as stained glass: their colours fall as light on the terrace
    and ripple as a reflection in the pool."""
    col, cov = pane_layer(labels, rng)
    a = colour_panes(photo, labels, np.random.default_rng(int(rng.integers(1 << 30))))
    L = la.lum(photo)
    rows = np.clip(2 * axis - np.arange(S), 0, S - 1)
    # light on the floor: mirrored, sheared toward the camera, stretched and soft
    yy = np.arange(S)[:, None]
    xx = np.arange(S)[None, :]
    src_y = np.clip(axis - (yy - axis) / 1.6, 0, S - 1).astype(int)
    src_x = np.clip(xx + ((yy - axis) * 0.35).astype(int), 0, S - 1)
    below = (yy > axis).astype(np.float32)
    pc, pa = col[src_y, src_x], cov[src_y, src_x] * below
    pc = la.blur_rgb(pc * pa[..., None], 14)
    pa = m.blur(pa, 14)
    floor = np.clip((L - 0.55) * 3, 0, 1) * (1 - cov)
    tint = np.where(pa[..., None] > 1e-3, pc / np.maximum(pa[..., None], 1e-3), 0)
    k = (pa * floor * light)[..., None]
    a = a * (1 - k) + a * np.clip(tint * 1.25, 0, 1.2) * k
    # reflection in the pool, with ripples
    R, G, B = photo[..., 0], photo[..., 1], photo[..., 2]
    pool = m.blur(((B - R > 0.25) & (L > 0.45) & (yy > axis - 80)).astype(np.float32), 3)
    rc, ra = col[rows], cov[rows]
    ripple = (np.sin(np.arange(S) / 6.0) * 9 + np.sin(np.arange(S) / 17.0) * 14).astype(int)
    rc = np.stack([np.roll(rc[y], ripple[y], 0) for y in range(S)])
    ra = np.stack([np.roll(ra[y], ripple[y], 0) for y in range(S)])
    w = (m.blur(ra, 3) * pool * water)[..., None]
    a = a * (1 - w) + np.clip(rc * (0.5 + 0.7 * L[..., None]), 0, 1) * w
    return np.clip(a, 0, 1)


# the glass balustrade around the terrace (full-frame pixel polygon)
GLASS = [(165, 927), (1635, 952), (1650, 960), (1800, 962), (1800, 1800), (1635, 1800),
         (900, 1665), (457, 1492), (172, 1410)]


def spiral(d, cx, cy, size, turns, flip, width):
    """A Burton-style curl: an Archimedean spiral ending in a long tendril."""
    th = np.linspace(0, turns * 2 * np.pi, 400)
    r = size * th / th[-1]
    pts = [(cx + flip * r_ * np.cos(t), cy - r_ * np.sin(t)) for r_, t in zip(r, th)]
    end = pts[-1]
    tail = [(end[0] + flip * k * size * 0.08, end[1] + size * 0.9 * (k / 12) ** 1.6) for k in range(13)]
    d.line(pts + tail, fill=255, width=width, joint="curve")


def feerie(a, rng, tint=0.34):
    """Smoked glass with etched curls and stars, and a few fireflies."""
    k = 2                                                 # draw at 2x for smooth lines
    poly = Image.new("L", (S, S), 0)
    ImageDraw.Draw(poly).polygon(GLASS, fill=255)
    glass = m.blur(np.asarray(poly, np.float32) / 255, 2)[..., None]
    smoke = np.array([0.06, 0.04, 0.12], np.float32)       # black with a hint of violet
    a = a * (1 - glass * tint) + smoke * glass * tint

    lines = Image.new("L", (S * k, S * k), 0)
    d = ImageDraw.Draw(lines)
    for cx, cy, size, turns, flip in ((330, 1250, 85, 2.6, 1), (690, 1390, 110, 3.0, -1),
                                      (1080, 1360, 70, 2.4, 1), (1420, 1500, 120, 3.2, -1),
                                      (1720, 1180, 75, 2.6, 1)):
        spiral(d, cx * k, cy * k, size * k, turns, flip, 3 * k)
    # a curly hill line along the foot of the glass
    xs = np.linspace(172, 1800, 300)
    base = np.interp(xs, [172, 457, 900, 1635, 1800], [1395, 1477, 1650, 1785, 1790])
    hill = [(x * k, (y - 18 - 10 * np.sin(x / 55)) * k) for x, y in zip(xs, base)]
    d.line(hill, fill=180, width=2 * k)
    for _ in range(26):                                    # tiny four-point stars
        x, y = rng.uniform(200, 1780), rng.uniform(960, 1350)
        r = rng.uniform(5, 13)
        d.line([(x * k - r * k, y * k), (x * k + r * k, y * k)], fill=230, width=k)
        d.line([(x * k, y * k - r * k), (x * k, y * k + r * k)], fill=230, width=k)
    etch = np.asarray(lines.resize((S, S), Image.LANCZOS), np.float32) / 255
    etch = (m.blur(etch, 0.8) * glass[..., 0])[..., None]
    a = a * (1 - etch * 0.55) + np.array([0.95, 0.96, 1.0]) * etch * 0.55

    # fireflies: soft warm glows drifting over the terrace and the pool
    core = Image.new("L", (S, S), 0)
    dc = ImageDraw.Draw(core)
    for _ in range(55):
        x, y, r = rng.uniform(150, 1790), rng.uniform(900, 1700), rng.uniform(2.5, 6)
        dc.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(170, 255)))
    halo = np.asarray(core.filter(ImageFilter.GaussianBlur(14)), np.float32) / 255 * 3.5
    dot = np.asarray(core.filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
    glow = np.clip(halo * 0.6 + dot, 0, 1)[..., None]
    return np.clip(a + glow * np.array([1.0, 0.9, 0.55]) * 0.85, 0, 1)


def piscine(a, rng, sort_area, smear_area, keep=None, tones=PASTELS, n_big=3, n_small=6, mode="veils"):
    """The #40 recipe."""
    a = m.grade(a, 1.15, 1.3)
    clean = a.copy()
    m.pixel_sort(a, rng, 12, 0.55, 0.95, vertical=False, area=sort_area)
    m.smear(a, rng, 6, area=smear_area)
    a = grid(a, rng, tones) if mode == "grid" else la.veils(a, rng, tones, n_big, n_small)
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

    # 1b. multicolour versions of #40
    multi = [("MULTICOLORE", dict(tones=RAINBOW, n_big=4, n_small=10)),
             ("MULTICOLORE DENSE", dict(tones=RAINBOW, n_big=7, n_small=22)),
             ("CARRÉS", dict(tones=RAINBOW, n_big=0, n_small=40)),
             ("GRILLE", dict(tones=RAINBOW, mode="grid"))]
    items = []
    for k, (label, kw) in enumerate(multi):
        a = piscine(src.copy(), np.random.default_rng(4001), (0.5, 0.55, 1, 0.85), (0, 0.2, 1, 0.7), **kw)
        slug = label.lower().replace(" ", "-").replace("é", "e")
        save(a, f"40-piscine-{slug}", 40, "PISCINE", "PALM SPRINGS", "33.82° N  116.54° W")
        items.append((label, a))
    sheet(items, "planche-multicolore-40.jpg", 4)

    # 1c. the fence panes, each in its own colour
    clean = m.grade(src.copy(), 1.15, 1.3)
    labels = panes(clean)
    print("panes:", len(np.unique(labels)) - 1)
    base = piscine(src.copy(), np.random.default_rng(4001), (0.5, 0.55, 1, 0.85), (0, 0.2, 1, 0.7))
    plain = m.grade(src.copy(), 1.15, 1.3)
    m.pixel_sort(plain, np.random.default_rng(4001), 12, 0.55, 0.95, vertical=False, area=(0.5, 0.55, 1, 0.85))
    vit = []
    for label, img, seed in (("VITRAIL + VOILES", base, 7), ("VITRAIL", plain, 7), ("VITRAIL II", plain, 8)):
        out = la.grain(colour_panes(img, labels, np.random.default_rng(seed)), np.random.default_rng(seed), 0.02)
        slug = label.lower().replace(" + ", "-").replace(" ", "-")
        save(out, f"40-piscine-{slug}", 40, "PISCINE", "PALM SPRINGS", "33.82° N  116.54° W")
        vit.append((label, out))
    sheet(vit, "planche-vitrail-40.jpg", 3)

    # 1d. the untouched photo, only the panes coloured
    orig = []
    for k, seed in enumerate((7, 8, 9)):
        out = colour_panes(src.copy(), labels, np.random.default_rng(seed))
        save(out, f"40-piscine-originale-{k + 1}", 40, "PISCINE", "PALM SPRINGS", "33.82° N  116.54° W")
        orig.append((f"ORIGINALE {k + 1}", out))
    sheet(orig, "planche-originale-40.jpg", 3)

    # 1e. LUMIÈRE: the stained glass lets the sun through
    lum_items = []
    for k, seed in enumerate((7, 8, 9)):
        out = lumiere(src.copy(), labels, np.random.default_rng(seed))
        save(out, f"40-piscine-lumiere-{k + 1}", 40, "LUMIÈRE", "PALM SPRINGS", "33.82° N  116.54° W")
        lum_items.append((f"LUMIÈRE {k + 1}", out))
    sheet(lum_items, "planche-lumiere-40.jpg", 3)

    # 1f. FÉERIE: LUMIÈRE + smoked glass, etched curls and fireflies
    fee = []
    for k, (_, img) in enumerate(lum_items):
        out = feerie(img, np.random.default_rng(70 + k))
        save(out, f"40-piscine-feerie-{k + 1}", 40, "FÉERIE", "PALM SPRINGS", "33.82° N  116.54° W")
        fee.append((f"FÉERIE {k + 1}", out))
    sheet(fee, "planche-feerie-40.jpg", 3)

    # 2. the series
    series, multis = [], []
    for num, stem, title, place, coords, load, sort_area, smear_area, keep in PIECES:
        a = piscine(load(), np.random.default_rng(num * 101), sort_area, smear_area, keep)
        save(a, f"{num}-{stem}", num, title, place, coords)
        series.append((f"#{num} {title}", a))
        b = piscine(load(), np.random.default_rng(num * 101), sort_area, smear_area, keep,
                    tones=RAINBOW, n_big=4, n_small=10)
        save(b, f"{num}-{stem}-multicolore", num, title, place, coords)
        multis.append((f"#{num} {title}", b))
    sheet(series, "planche-serie.jpg", 3)
    sheet(multis, "planche-serie-multicolore.jpg", 3)
    print(OUT)
