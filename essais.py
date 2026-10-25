"""Les deux photos de Ouagadougou dans les 4 styles validés au fil de la série.

    python3 essais.py      -> output/la/essais/*.png + planche.jpg
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import la
import mix as m
from voyages import trame_rouge

S = la.S
OUT = os.path.join(la.ROOT, "output", "la", "essais")
PHOTOS = dict(
    porteuse=lambda: la.load("13.png", (0, 358, 923, 1280), patch=[(812, 382, 900, 452)]),
    camion=lambda: la.load("11.png", (0, 358, 923, 1280), hide=[(435, 1035, 535, 1070)],
                           patch=[(812, 382, 900, 452)]),
)
RED = dict(porteuse=(0.38, 0.14, 0.62, 0.96), camion=(0.38, 0.07, 0.66, 0.48))


def geneve(a, rng, key):
    """Comme #28-#30: couleurs d'origine, coulures, rectangles translucides."""
    a = m.grade(a, 1.15, 1.15)
    m.pixel_sort(a, rng, 8, 0.5, 0.92, area=(0, 0, 1, 0.45))
    m.pixel_sort(a, rng, 16, 0.25, 0.75, vertical=False, area=(0, 0.6, 1, 1))
    m.slices(a, rng, 5, 50, area=(0, 0.5, 1, 1))
    tones = [(196, 22, 24), (40, 190, 170), (226, 140, 40), "mono"]
    return la.grain(m.veils(a, rng, tones, 3, 6, (0.25, 0.55)), rng, 0.03)


def series_la(a, rng, key):
    """Comme #36-#42: un traitement fort (thermique / infrarouge de nuit)."""
    if key == "porteuse":
        a = m.grade(a, 1.3)
        return la.gradient(m.blur(la.lum(a), 1.2), [
            (0, (10, 0, 30)), (0.25, (90, 10, 120)), (0.45, (220, 40, 60)),
            (0.65, (255, 140, 20)), (0.85, (255, 230, 90)), (1, (255, 255, 240))])
    a = la.bloom(m.grade(a, 1.2, 1.2), 0.6, 12, 0.8)
    la.trail(a, int(0.55 * S), int(0.8 * S), 40)
    m.rgb_split(a, 7)
    return la.scanlines(la.grain(a, rng, 0.03))


def sacre(a, rng, key):
    """Comme #46: trame noire sur papier, un aplat rouge."""
    if key == "camion":
        a = np.clip(a, 0, 1) ** 0.6
    return trame_rouge(a, rng, RED[key], 1.3)


def origine(a, rng, key):
    """Comme la collection d'origine (#12): noir et blanc, glitch lourd, trame."""
    a = m.grade(a, 1.3)
    m.pixel_sort(a, rng, 26, 0.3, 0.88)
    m.smear(a, rng, 18)
    m.slices(a, rng, 10, 80)
    m.mosaic(a, rng, 6)
    a = m.mono(a, [(0, (6, 6, 6)), (0.35, (55, 55, 55)), (0.7, (200, 200, 198)), (1, (250, 250, 248))])
    d = (rng.random((S, S)) < la.lum(a)).astype(np.float32)[..., None]
    return la.grain(a * 0.8 + d * 0.2, rng, 0.05)


STYLES = dict(geneve=geneve, serie_la=series_la, sacre=sacre, origine=origine)
NAMES = dict(geneve="1 · GENÈVE", serie_la="2 · SÉRIE LA", sacre="3 · SACRÉ", origine="4 · ORIGINE")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t = 600
    sheet = Image.new("RGB", (4 * t, 2 * t + 60), (242, 239, 232))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(la.FONTS + "Bold.ttf", 30)
    for j, (style, fn) in enumerate(STYLES.items()):
        d.text((j * t + 20, 15), NAMES[style], font=font, fill=(24, 24, 24))
        for i, (key, load) in enumerate(PHOTOS.items()):
            a = fn(load(), np.random.default_rng(6000 + 10 * j + i), key)
            img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
            img.save(os.path.join(OUT, f"{key}-{style}.png"), optimize=True)
            sheet.paste(img.resize((t, t)), (j * t, 60 + i * t))
    sheet.save(os.path.join(OUT, "planche.jpg"), quality=90)
    print(OUT)
