"""DROITS RÉSERVÉS — série Voyages (Èze, Monaco, Burkina Faso, Los Angeles).

Un traitement différent par photo, comme la série LA.
Même mise en page que la.py (affiche + plein cadre).

    python3 voyages.py           -> output/la/43-*.png ... 49-*.png
    python3 voyages.py eze       -> une seule pièce
"""
import sys

import numpy as np
from PIL import Image

import la
import mix as m

S = la.S


def restore(a, clean, x0, y0, x1, y1):
    ys, xs = slice(int(y0 * S), int(y1 * S)), slice(int(x0 * S), int(x1 * S))
    a[ys, xs] = clean[ys, xs]


def trame_rouge(a, rng, block, contrast=1.35, lift=0.0):
    """Le langage de #46: trame noire 45° sur papier, un seul aplat rouge sur le sujet."""
    a = m.grade(a, contrast, 1.0, lift)
    dots = la.halftone(np.clip((1 - la.lum(a)) * 1.1, 0, 1), 45, 6)[..., None]
    out = la.PAPER * (1 - dots) + np.array([0.08, 0.08, 0.08]) * dots
    x0, y0, x1, y1 = (int(v * S) for v in block)
    out[y0:y1, x0:x1] *= np.array([214, 32, 38], np.float32) / 255 * 0.85 + 0.15
    return la.grain(out, rng, 0.015)


def eze():
    """Infrarouge Aerochrome: les cactus virent au rouge corail."""
    rng = np.random.default_rng(4301)
    a = la.load("09.png", (0, 507, 923, 1428), patch=[(812, 530, 900, 600, "down")])
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    veg = m.blur(np.clip((G - R) * 10 + 0.2, 0, 1) * np.clip((G - B) * 10 + 0.5, 0, 1), 2)[..., None]
    ir = np.stack([np.clip(G * 1.4 + 0.12, 0, 1), R * 0.35, B * 0.5], -1)   # dull cactus greens -> red
    sky = np.stack([R * 0.75, G * 0.95, np.clip(B * 1.1, 0, 1)], -1)
    a = m.grade(sky * (1 - veg) + ir * veg, 1.15, 1.3)
    m.pixel_sort(a, rng, 8, 0.5, 0.92, area=(0.1, 0, 0.9, 0.45))
    m.slices(a, rng, 4, 40, area=(0, 0.6, 1, 1))
    la.save(la.grain(a, rng, 0.03), "43-eze", "43", "INFRAROUGE", "JARDIN EXOTIQUE D'ÈZE",
            "43.73° N  7.36° E", "AEROCHROME / PIXEL SORT")


def bleu():
    """Risographie deux encres, rose fluo et bleu."""
    rng = np.random.default_rng(4401)
    a = m.grade(la.load("10.png", (0, 358, 923, 1280)), 1.6, 1.0, -0.12)
    m.pixel_sort(a, rng, 8, 0.45, 0.9, vertical=False, area=(0, 0.5, 1, 0.85))
    a = la.riso(a, (255, 72, 176), (0, 120, 191))
    la.save(la.grain(a, rng, 0.02), "44-bleu", "44", "PISCINE", "MONTE-CARLO",
            "43.74° N  7.42° E", "RISOGRAPHIE 2 ENCRES")


def ouaga():
    """Slit-scan et chroma: le camion file dans la nuit."""
    rng = np.random.default_rng(4501)
    a = la.load("11.png", (0, 358, 923, 1280), hide=[(435, 1035, 535, 1070)],
                patch=[(812, 382, 900, 452)])
    a = la.bloom(m.grade(a, 1.15, 1.2), 0.6, 12, 0.8)
    la.trail(a, int(0.55 * S), int(0.8 * S), 40)
    m.rgb_split(a, 7)
    m.slices(a, rng, 7, 90, area=(0, 0.45, 1, 0.9))
    la.save(la.scanlines(la.grain(a, rng, 0.03)), "45-ouaga", "45", "VITESSE", "OUAGADOUGOU",
            "12.37° N  1.52° W", "SLIT-SCAN / CHROMA")


def crocodile():
    """Trame noire sur papier, un seul aplat rouge sur le crocodile."""
    rng = np.random.default_rng(4601)
    a = m.grade(la.load("12.png", (0, 358, 923, 1280)), 1.35)
    L = la.lum(a)
    dots = la.halftone(np.clip((1 - L) * 1.1, 0, 1), 45, 6)[..., None]
    out = la.PAPER * (1 - dots) + np.array([0.08, 0.08, 0.08]) * dots
    x0, y0, x1, y1 = (int(v * S) for v in (0.3, 0.6, 0.72, 0.93))
    out[y0:y1, x0:x1] *= np.array([214, 32, 38], np.float32) / 255 * 0.85 + 0.15
    la.save(la.grain(out, rng, 0.015), "46-crocodile", "46", "SACRÉ", "BAZOULÉ, BURKINA FASO",
            "12.21° N  1.55° W", "TRAME / UN ROUGE")


def porteuse():
    """Image thermique: la chaleur de la rue."""
    rng = np.random.default_rng(4701)
    a = m.grade(la.load("13.png", (0, 358, 923, 1280), patch=[(812, 382, 900, 452)]), 1.3)
    m.pixel_sort(a, rng, 8, 0.3, 0.85, vertical=False, area=(0, 0, 1, 0.35))
    a = la.gradient(m.blur(la.lum(a), 1.2), [
        (0, (10, 0, 30)), (0.25, (90, 10, 120)), (0.45, (220, 40, 60)),
        (0.65, (255, 140, 20)), (0.85, (255, 230, 90)), (1, (255, 255, 240))])
    la.save(la.scanlines(la.grain(a, rng, 0.025), 3, 0.1), "47-porteuse", "47", "CHALEUR",
            "OUAGADOUGOU", "12.37° N  1.52° W", "IMAGE THERMIQUE")


def hollywood():
    """Couleurs saturées, la colline coule, voiles translucides."""
    rng = np.random.default_rng(4801)
    a = m.grade(la.load("14.png", (0, 358, 923, 1280), hide=[(355, 440, 480, 482)]), 1.15, 1.3)
    clean = a.copy()
    m.pixel_sort(a, rng, 12, 0.4, 0.85, area=(0, 0, 1, 0.6))
    m.smear(a, rng, 6, area=(0, 0.3, 1, 0.9))
    a = la.veils(a, rng, [(255, 150, 190), (255, 210, 80), (80, 200, 220), (255, 255, 255)])
    restore(a, clean, 0.5, 0.1, 0.76, 0.4)                    # face stays clean
    la.save(la.grain(a, rng, 0.025), "48-hollywood", "48", "STAR", "HOLLYWOOD HILLS",
            "34.13° N  118.32° W", "PIXEL SORT / VOILES")


def ascension():
    """Le soleil explose, la tour coule vers le ciel."""
    rng = np.random.default_rng(4901)
    clean = m.grade(la.load("15.png", (0, 725, 923, 1648)), 1.2, 1.3)
    a = la.bloom(clean, 0.6, 14, 0.9)
    for _ in range(70):                                       # the tower melts in thin columns
        x = int(rng.integers(int(0.12 * S), int(0.62 * S)))
        m.sort_runs(a, x, 0, int(rng.integers(6, 40)), int(rng.uniform(0.3, 0.62) * S), 0.1, 0.95)
    m.slices(a, rng, 6, 60, area=(0, 0, 1, 0.5))
    m.rgb_split(a, 6)
    restore(a, la.bloom(clean, 0.6, 14, 0.9), 0.5, 0.52, 0.72, 1)
    la.save(la.scanlines(la.grain(a, rng, 0.03)), "49-ascension", "49", "ASCENSION",
            "DOWNTOWN LOS ANGELES", "34.05° N  118.26° W", "BLOOM / PIXEL SORT")


PIECES = dict(eze=eze, bleu=bleu, ouaga=ouaga, crocodile=crocodile, porteuse=porteuse,
              hollywood=hollywood, ascension=ascension)

if __name__ == "__main__":
    for name in sys.argv[1:] or PIECES:
        PIECES[name]()
