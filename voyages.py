"""DROITS RÉSERVÉS — série Voyages (Èze, Monaco, Burkina Faso, Los Angeles).

Même mise en page que la.py (affiche + plein cadre), un traitement par photo.

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


def eze():
    """Les cactus coulent dans un dégradé de coucher de soleil méditerranéen."""
    rng = np.random.default_rng(4301)
    a = m.grade(la.load("09.png", (0, 507, 923, 1428), patch=[(812, 530, 900, 600, "down")]), 1.15, 1.1)
    m.pixel_sort(a, rng, 18, 0.5, 0.95, area=(0, 0, 1, 0.6))
    g = la.gradient(la.lum(a), [(0, (8, 20, 60)), (0.35, (30, 90, 110)), (0.6, (230, 90, 70)),
                                (0.8, (250, 180, 120)), (1, (252, 240, 220))])
    a = la.grain(g * 0.78 + a * 0.22, rng, 0.025)
    la.save(a, "43-eze", "43", "ÈZE", "JARDIN EXOTIQUE D'ÈZE",
            "43.73° N  7.36° E", "PIXEL SORT / DÉGRADÉ")


def bleu():
    """La piscine en risographie turquoise et corail, façon Hockney."""
    rng = np.random.default_rng(4401)
    a = m.grade(la.load("10.png", (0, 358, 923, 1280)), 1.2, 1.2)
    m.pixel_sort(a, rng, 10, 0.5, 0.95, vertical=False, area=(0, 0.5, 1, 0.85))
    a = la.riso(a, (0, 160, 205), (255, 90, 120), shift=(7, -5))
    la.save(la.grain(a, rng, 0.015), "44-bleu", "44", "BLEU", "MONTE-CARLO",
            "43.74° N  7.42° E", "RISOGRAPHIE 2 ENCRES")


def ouaga():
    """La nuit à Ouagadougou: les lampadaires explosent, la route file."""
    rng = np.random.default_rng(4501)
    a = m.grade(la.load("11.png", (0, 358, 923, 1280), hide=[(435, 1035, 535, 1070)],
                         patch=[(812, 382, 900, 452)]), 1.2, 1.25)
    a = la.bloom(a, 0.6, 14, 1.1)
    la.trail(a, int(0.05 * S), int(0.45 * S), 30)
    m.rgb_split(a, 6)
    m.slices(a, rng, 8, 70, area=(0, 0.5, 1, 1))
    a = la.scanlines(la.grain(a, rng, 0.035))
    la.save(a, "45-ouaga", "45", "OUAGA", "OUAGADOUGOU",
            "12.37° N  1.52° W", "BLOOM / SLIT-SCAN")


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
    """Ocre et poussière; seul le foulard turquoise garde sa couleur."""
    rng = np.random.default_rng(4701)
    src = m.grade(la.load("13.png", (0, 358, 923, 1280), patch=[(812, 382, 900, 452)]), 1.2, 1.2)
    a = src.copy()
    m.pixel_sort(a, rng, 10, 0.35, 0.9, area=(0, 0, 0.38, 1))
    m.pixel_sort(a, rng, 10, 0.35, 0.9, area=(0.62, 0, 1, 1))
    g = la.gradient(la.lum(a), [(0, (40, 15, 10)), (0.5, (190, 90, 40)), (1, (250, 232, 205))])
    R, G, B = src[..., 0], src[..., 1], src[..., 2]
    scarf = m.blur(((G - R > 0.06) & (B - R > 0.04)).astype(np.float32), 2)[..., None]
    scarf[:, :int(0.36 * S)] = scarf[:, int(0.64 * S):] = 0
    a = g * (1 - scarf) + src * scarf
    la.save(la.grain(a, rng, 0.03), "47-porteuse", "47", "PORTEUSE", "OUAGADOUGOU",
            "12.37° N  1.52° W", "DUOTONE / UNE COULEUR")


def hollywood():
    """Grille pop 2x2: quatre palettes, l'enseigne effacée."""
    rng = np.random.default_rng(4801)
    a = m.grade(la.load("14.png", (0, 358, 923, 1280), hide=[(355, 440, 480, 482)]), 1.2)
    h = S // 2
    small = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((h, h), Image.LANCZOS),
                       np.float32) / 255
    L = np.round(la.lum(small) * 4) / 4                      # posterize to 5 levels
    palettes = [
        [(0, (20, 10, 60)), (0.5, (255, 60, 140)), (1, (255, 240, 120))],
        [(0, (0, 40, 40)), (0.5, (0, 190, 170)), (1, (255, 250, 230))],
        [(0, (60, 0, 0)), (0.5, (255, 120, 0)), (1, (200, 230, 255))],
        [(0, (10, 10, 10)), (0.5, (120, 120, 120)), (1, (214, 32, 38))],
    ]
    out = np.zeros((S, S, 3), np.float32)
    for i, p in enumerate(palettes):
        y, x = (i // 2) * h, (i % 2) * h
        out[y:y + h, x:x + h] = la.gradient(L, p)
    m.slices(out, rng, 6, 40)
    la.save(la.grain(out, rng, 0.02), "48-hollywood", "48", "STAR", "HOLLYWOOD HILLS",
            "34.13° N  118.32° W", "SÉRIGRAPHIE POP 2×2")


def ascension():
    """La tour fond vers le ciel; l'homme qui la regarde reste intact."""
    rng = np.random.default_rng(4901)
    clean = m.grade(la.load("15.png", (0, 725, 923, 1648)), 1.15, 1.15)
    a = clean.copy()
    m.pixel_sort(a, rng, 22, 0.25, 0.85, area=(0.1, 0, 0.75, 0.6))
    m.smear(a, rng, 8, area=(0, 0, 1, 0.5))
    m.rgb_split(a, 4)
    a = la.veils(a, rng, [(40, 80, 255), (255, 255, 255), "mono"], n_big=2, n_small=4)
    restore(a, clean, 0.5, 0.52, 0.72, 1)
    la.save(la.grain(a, rng, 0.025), "49-ascension", "49", "ASCENSION", "DOWNTOWN LOS ANGELES",
            "34.05° N  118.26° W", "PIXEL SORT / VOILES")


PIECES = dict(eze=eze, bleu=bleu, ouaga=ouaga, crocodile=crocodile, porteuse=porteuse,
              hollywood=hollywood, ascension=ascension)

if __name__ == "__main__":
    for name in sys.argv[1:] or PIECES:
        PIECES[name]()
