"""#40, trois directions « Burton » sans dessins collés: tout vient de la photo.

1. NOIR    — palette Burton: froid, désaturé, noirs profonds, ciel lavande; le vitrail
             et la piscine brillent, le pot devient rayé.
2. ENCRE   — la photo redessinée à la plume (hachures croisées sur papier),
             seuls le vitrail et la piscine sont en couleur, comme une aquarelle.
3. BONBON  — conte de jour: pastels acidulés, ciel rose-lavande, pot rayé rose.

    python3 burton2.py     -> output/piscines/40-burton-*.png + planche
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import la
import mix as m
import piscines as p

S = la.S
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)
OUT = os.path.join(la.ROOT, "output", "piscines")


def masks(src):
    R, G, B = src[..., 0], src[..., 1], src[..., 2]
    L = la.lum(src)
    pool = m.blur(((B - R > 0.25) & (L > 0.45) & (YY > 1000)).astype(np.float32), 3)
    cond = (((B > 0.62) & (B - R > 0.16)) | ((L > 0.75) & (YY < 330))) & (YY < 800)
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
    near = m.blur(reach.astype(np.float32), 3) > 0.35
    sky = reach | (near & (B - R > 0.06) & (YY < 800))
    sky &= ~(((XX < 225) & (YY > 476)) | ((XX < 470) & (YY > 532)))
    pot = ((np.abs(XX - 1100) < 108 - 14 * np.clip((YY - 1230) / 330, 0, 1)) &
           (YY > 1225) & (YY < 1555)).astype(np.float32)
    return m.blur(sky.astype(np.float32), 2), pool, m.blur(pot, 1.2)


def stripes(a, pot, light, dark):
    curve = (YY - 1225) + ((XX - 1100) / 108) ** 2 * 26
    st = (np.sin(curve / 22) > 0).astype(np.float32)[..., None]
    shade = (0.6 + 0.4 * np.clip(1 - np.abs(XX - 1070) / 120, 0, 1))[..., None]
    c = (st * np.array(light) + (1 - st) * np.array(dark)) * shade
    return a * (1 - pot[..., None]) + c * pot[..., None]


def noir(src, labels):
    sky, pool, pot = masks(src)
    L = la.lum(src)[..., None]
    a = src * 0.3 + L * 0.7                                          # desaturate
    a = np.clip((a - 0.5) * 1.35 + 0.42, 0, 1) ** 1.25                # deep blacks
    a = a * np.array([0.82, 0.84, 1.0]) + np.array([0.02, 0.0, 0.06])  # cold violet cast
    g = np.clip(YY / 700, 0, 1)[..., None]
    skyc = np.array([0.42, 0.38, 0.55]) * (1 - g) + np.array([0.86, 0.82, 0.9]) * g
    a = a * (1 - sky[..., None]) + skyc * sky[..., None]
    a = a * (1 - pool[..., None] * 0.8) + np.clip(src * np.array([0.55, 1.0, 1.1]), 0, 1) * pool[..., None] * 0.8
    a = p.lumiere(np.clip(a, 0, 1), labels, np.random.default_rng(7), light=0.9, water=0.45)
    col, cov = p.pane_layer(labels, np.random.default_rng(7))
    a = np.clip(a + la.blur_rgb(col * cov[..., None], 24) * 0.4, 0, 1)
    a = stripes(a, pot, [0.93, 0.92, 0.96], [0.06, 0.05, 0.08])
    vig = 1 - 0.45 * (((XX / S - 0.5) ** 2 + (YY / S - 0.5) ** 2) * 2.2)
    return la.grain(np.clip(a * vig[..., None], 0, 1), np.random.default_rng(1), 0.03)


def encre(src, labels):
    sky, pool, pot = masks(src)
    L = m.blur(la.lum(src), 1.0)
    paper = np.array([0.95, 0.93, 0.88])
    ink = np.zeros((S, S), np.float32)
    for t, ang, per in ((0.78, 45, 9), (0.58, -45, 8), (0.4, 0, 7), (0.24, 90, 6)):
        th = np.deg2rad(ang)
        u = XX * np.cos(th) + YY * np.sin(th)
        line = (np.sin(u * 2 * np.pi / per) > 0.55).astype(np.float32)
        ink = np.maximum(ink, line * (L < t) * (1 - sky * 0.85))
    edges = np.abs(m.blur(L, 1.2) - m.blur(L, 3.5))
    ink = np.maximum(ink, np.clip((edges - 0.025) * 18, 0, 1))
    ink = np.maximum(ink, (L < 0.12).astype(np.float32))
    a = paper * (1 - ink[..., None] * 0.92) + np.array([0.07, 0.06, 0.09]) * ink[..., None] * 0.92
    wash = np.clip(src * np.array([0.5, 1.0, 1.1]), 0, 1)
    a = a * (1 - pool[..., None] * 0.55) + a * wash * pool[..., None] * 0.55 * 1.1
    col, cov = p.pane_layer(labels, np.random.default_rng(8))
    k = (m.blur(cov, 1.0) * 0.7)[..., None]
    a = a * (1 - k) + a * col * 1.15 * k
    return la.grain(np.clip(a, 0, 1), np.random.default_rng(2), 0.015)


def bonbon(src, labels):
    sky, pool, pot = masks(src)
    L = la.lum(src)[..., None]
    a = src * 0.55 + L * 0.45
    a = np.clip(a * 0.85 + 0.18, 0, 1)                               # lifted, milky
    a = a * np.array([1.0, 0.92, 0.98]) + np.array([0.03, 0.0, 0.04])  # pink-lavender cast
    g = np.clip(YY / 700, 0, 1)[..., None]
    skyc = np.array([0.78, 0.72, 0.95]) * (1 - g) + np.array([1.0, 0.8, 0.86]) * g
    a = a * (1 - sky[..., None]) + skyc * sky[..., None]
    a = a * (1 - pool[..., None] * 0.7) + np.clip(src * np.array([0.7, 1.05, 1.05]) + 0.05, 0, 1) * pool[..., None] * 0.7
    a = p.lumiere(np.clip(a, 0, 1), labels, np.random.default_rng(9), light=0.7, water=0.4)
    a = stripes(a, pot, [1.0, 0.95, 0.97], [0.95, 0.55, 0.7])
    return la.grain(a, np.random.default_rng(3), 0.02)


if __name__ == "__main__":
    src = la.load("06.png", (0, 425, 923, 1343))
    labels = p.panes(m.grade(src.copy(), 1.15, 1.3))
    items = []
    for name, title, fn in (("noir", "NOIR", noir), ("encre", "ENCRE", encre), ("bonbon", "BONBON", bonbon)):
        a = fn(src.copy(), labels)
        stem = os.path.join(OUT, f"40-burton-{name}")
        Image.fromarray((a * 255).astype(np.uint8)).save(stem + ".png", optimize=True)
        la.poster(a, "40", title, "PALM SPRINGS", "33.82° N  116.54° W", "CONTE / VITRAIL").save(
            stem + "-affiche.png", optimize=True)
        items.append((title, a))
    t = 700
    sheet = Image.new("RGB", (3 * t, t + 60), (242, 239, 232))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(la.FONTS + "Bold.ttf", 30)
    for i, (title, a) in enumerate(items):
        d.text((i * t + 20, 15), f"{i + 1} · {title}", font=font, fill=(24, 24, 24))
        sheet.paste(Image.fromarray((a * 255).astype(np.uint8)).resize((t, t)), (i * t, 60))
    sheet.save(os.path.join(OUT, "planche-burton.jpg"), quality=90)
    print(OUT)
