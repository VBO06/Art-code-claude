"""DROITS RÉSERVÉS — série Voyages (Èze, Monaco, Burkina Faso, Los Angeles).

Trame noire sur papier et un seul aplat rouge par image (langage de #46).
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
    a = la.load("09.png", (0, 507, 923, 1428), patch=[(812, 530, 900, 600, "down")])
    a = trame_rouge(a, np.random.default_rng(4301), (0.68, 0.04, 0.8, 0.96))
    la.save(a, "43-eze", "43", "ÈZE", "JARDIN EXOTIQUE D'ÈZE",
            "43.73° N  7.36° E", "TRAME / UN ROUGE")


def bleu():
    a = la.load("10.png", (0, 358, 923, 1280))
    a = trame_rouge(a, np.random.default_rng(4401), (0.1, 0.74, 0.4, 0.98), 1.7, -0.08)
    la.save(a, "44-bleu", "44", "BLEU", "MONTE-CARLO",
            "43.74° N  7.42° E", "TRAME / UN ROUGE")


def ouaga():
    a = la.load("11.png", (0, 358, 923, 1280), hide=[(435, 1035, 535, 1070)],
                patch=[(812, 382, 900, 452)])
    a = np.clip(a, 0, 1) ** 0.6                              # open up the night
    a = trame_rouge(a, np.random.default_rng(4501), (0.38, 0.07, 0.66, 0.48), 1.3)
    la.save(a, "45-ouaga", "45", "OUAGA", "OUAGADOUGOU",
            "12.37° N  1.52° W", "TRAME / UN ROUGE")


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
    a = la.load("13.png", (0, 358, 923, 1280), patch=[(812, 382, 900, 452)])
    a = trame_rouge(a, np.random.default_rng(4701), (0.38, 0.14, 0.62, 0.96), 1.3)
    la.save(a, "47-porteuse", "47", "PORTEUSE", "OUAGADOUGOU",
            "12.37° N  1.52° W", "TRAME / UN ROUGE")


def hollywood():
    a = la.load("14.png", (0, 358, 923, 1280), hide=[(355, 440, 480, 482)])
    a = trame_rouge(a, np.random.default_rng(4801), (0.52, 0.11, 0.74, 0.37), 1.3)
    la.save(a, "48-hollywood", "48", "STAR", "HOLLYWOOD HILLS",
            "34.13° N  118.32° W", "TRAME / UN ROUGE")


def ascension():
    a = la.load("15.png", (0, 725, 923, 1648))
    a = trame_rouge(a, np.random.default_rng(4901), (0.2, 0.0, 0.56, 0.46), 1.3)
    la.save(a, "49-ascension", "49", "ASCENSION", "DOWNTOWN LOS ANGELES",
            "34.05° N  118.26° W", "TRAME / UN ROUGE")


PIECES = dict(eze=eze, bleu=bleu, ouaga=ouaga, crocodile=crocodile, porteuse=porteuse,
              hollywood=hollywood, ascension=ascension)

if __name__ == "__main__":
    for name in sys.argv[1:] or PIECES:
        PIECES[name]()
