"""DROITS RÉSERVÉS #66 GENÈVE (façade au couchant) et #67 MUSTANG (Palm Springs).

Chaque photo dans trois des traitements préférés de la série:
    vitrail / une couleur sur noir et blanc / voiles pastel de #40.

    python3 duo.py      -> output/duo/*.png (+ -affiche) + planche.jpg
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import la
import mix as m
import piscines as p

S = la.S
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)
OUT = os.path.join(la.ROOT, "output", "duo")


def label(cond, f=3, min_area=25):
    """Connected regions of a boolean map (4-neighbour flood fill on a 1/f grid)."""
    g = cond[::f, ::f]
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
                lab[y, x] = 0
    full = np.repeat(np.repeat(lab, f, 0), f, 1)[:S, :S]
    return np.where(cond, full, 0)


def one_colour(src, keep, contrast=1.15):
    """Black and white, except where `keep` (0..1 map) holds the original colour, deepened."""
    L = la.lum(m.grade(src.copy(), contrast))
    bw = np.repeat(L[..., None], 3, -1)
    rich = m.grade(src.copy(), 1.15, 1.45)
    k = keep[..., None]
    return np.clip(bw * (1 - k) + rich * k, 0, 1)


def veils40(src, rng, sort_area, smear_area):
    return p.piscine(src.copy(), rng, sort_area, smear_area)


# ---------------------------------------------------------------- the façade

def facade():
    src = p.load_jpg("villes/22.jpg")
    R, G, B = src[..., 0], src[..., 1], src[..., 2]
    L = la.lum(src)
    zone = (XX > 60) & (XX < 1780) & (YY > 585) & (YY < 1045) & ~((XX > 1180) & (YY > 808))
    gold = (R - B > 0.12) & (L > 0.35)
    # window grid: column and row boundaries read off the façade (full-frame pixels)
    blocks = [
        ([78, 160, 245, 330, 410], [590, 645, 705, 765, 815, 875, 935, 995, 1045]),
        ([500, 590, 675, 760, 855], [590, 645, 705, 765, 815, 875, 935, 995, 1045]),
        ([860, 950, 1045, 1140, 1235], [590, 645, 705, 765, 815, 875, 935, 995, 1045]),
        ([1335, 1440, 1545, 1650, 1750], [590, 645, 705, 765, 805]),
    ]
    panes = np.zeros((S, S), np.int32)
    glass = (L < 0.42) | gold
    n = 0
    for xs, ys in blocks:
        for x0, x1 in zip(xs[:-1], xs[1:]):
            for y0, y1 in zip(ys[:-1], ys[1:]):
                n += 1
                cell = np.zeros((S, S), bool)
                cell[y0 + 4:y1 - 4, x0 + 5:x1 - 5] = True
                panes[cell & glass] = n
    panes[(XX > 1080) & (XX < 1240) & (YY > 850)] = 0           # the small block below the roof
    panes[(XX > 1170) & (YY > 808)] = 0
    out = {}
    out["vitrail"] = p.colour_panes(src, panes, np.random.default_rng(66), op=0.6)
    out["or"] = one_colour(src, m.blur((gold & zone).astype(np.float32), 1.5))
    out["voiles"] = veils40(src, np.random.default_rng(6601), (0, 0.6, 1, 0.75), (0.05, 0.3, 0.95, 0.6))
    return out, ("66", "COUCHANT", "GENÈVE", "46.20° N  6.14° E")


# ---------------------------------------------------------------- the Mustang

def mustang():
    src = la.load("16.png", (0, 426, 923, 1580))
    R, G, B = src[..., 0], src[..., 1], src[..., 2]
    L = la.lum(src)
    flowers = m.blur(((R - G > 0.1) & (B - G > -0.05) & (R > 0.4)).astype(np.float32), 3) * 1.8
    flowers = np.clip(flowers, 0, 1)
    car = ((XX > 170) & (XX < 1545) & (YY > 880) & (YY < 1340) & (B - R > 0.06) & (L < 0.5)) | \
          ((XX > 170) & (XX < 1545) & (YY > 960) & (YY < 1300) & (L < 0.18))
    car = m.blur(car.astype(np.float32), 2)
    out = {}
    out["fleurs"] = one_colour(src, flowers)
    out["bleu"] = one_colour(src, np.clip(flowers + car, 0, 1))
    out["voiles"] = veils40(src, np.random.default_rng(6701), (0, 0.72, 1, 1), (0.05, 0, 0.95, 0.45))
    return out, ("67", "MUSTANG", "PALM SPRINGS", "33.82° N  116.54° W")


NOTES = dict(vitrail="VITRAIL", **{"or": "NOIR / OR"}, voiles="PIXEL SORT / VOILES",
             fleurs="NOIR / FLEURS", bleu="NOIR / BLEU NUIT")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    items = []
    for fn in (facade, mustang):
        versions, (num, title, place, coords) = fn()
        for key, a in versions.items():
            a = la.grain(a, np.random.default_rng(int(num)), 0.02)
            stem = os.path.join(OUT, f"{num}-{title.lower()}-{key}")
            Image.fromarray((a * 255).astype(np.uint8)).save(stem + ".png", optimize=True)
            la.poster(a, num, title, place, coords, NOTES[key]).save(stem + "-affiche.png", optimize=True)
            items.append((f"#{num} {NOTES[key]}", a))
    t = 600
    sheet = Image.new("RGB", (3 * t, 2 * (t + 50)), (242, 239, 232))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(la.FONTS + "Bold.ttf", 26)
    for i, (lab, a) in enumerate(items):
        x, y = (i % 3) * t, (i // 3) * (t + 50)
        d.text((x + 16, y + 12), lab, font=font, fill=(24, 24, 24))
        sheet.paste(Image.fromarray((a * 255).astype(np.uint8)).resize((t, t)), (x, y + 50))
    sheet.save(os.path.join(OUT, "planche.jpg"), quality=90)
    print(OUT)
