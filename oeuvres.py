"""DROITS RÉSERVÉS #50 et #51 — deux œuvres composées à partir des photos de Ouagadougou.

    python3 oeuvres.py           -> output/la/50-*.png, 51-*.png
"""
import sys

import numpy as np
from PIL import Image

import la
import mix as m

S = la.S
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32) / S


def hblur(a, r):
    """Horizontal box blur (anamorphic streaks)."""
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r), (0, 0)), mode="edge"), axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def scaled(a, s, cx, cy):
    """Zoom the image by s around (cx, cy) in fractions."""
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    w = int(S * s)
    small = img.resize((w, w), Image.LANCZOS)
    out = Image.new("RGB", (S, S))
    out.paste(small, (int(cx * S - cx * w), int(cy * S - cy * w)))
    return np.asarray(out, np.float32) / 255


def soleil():
    """La porteuse: silhouette noire sous un soleil terre cuite, la rue tramée en deux encres."""
    rng = np.random.default_rng(5001)
    src = m.grade(la.load("13.png", (0, 358, 923, 1280), patch=[(812, 382, 900, 452)]), 1.25, 1.2)
    L = la.lum(src)
    R, G, B = src[..., 0], src[..., 1], src[..., 2]

    # the woman: dark pixels inside her column, plus the basin
    col = (XX > 0.37) & (XX < 0.63) & (YY > 0.15)
    torso = (XX > 0.38) & (XX < 0.62) & (YY > 0.3)               # skip the dark street behind her head
    body = m.blur((torso & (L < 0.33)).astype(np.float32), 1.5)
    bowl = (((XX - 0.51) / 0.105) ** 2 + ((YY - 0.185) / 0.075) ** 2 < 1) & (YY > 0.18)
    rim = (((XX - 0.51) / 0.11) ** 2 + ((YY - 0.183) / 0.012) ** 2 < 1)
    neck = (XX > 0.49) & (XX < 0.535) & (YY > 0.24) & (YY < 0.3)
    basin = m.blur((bowl | rim | neck).astype(np.float32), 1.5)
    fig = np.clip(body + basin, 0, 1)[..., None]
    scarf = m.blur(((G - R > 0.06) & (B - R > 0.04) & col).astype(np.float32), 1.5)[..., None]
    veil = m.blur(((L > 0.75) & (XX > 0.44) & (XX < 0.52) & (YY > 0.3) & (YY < 0.65)).astype(np.float32), 1.5)[..., None]

    # street: dust drifting sideways, then a two-ink risograph (ochre + indigo)
    street = src.copy()
    m.pixel_sort(street, rng, 14, 0.3, 0.9, vertical=False, area=(0, 0.55, 1, 1))
    m.pixel_sort(street, rng, 8, 0.3, 0.9, area=(0, 0, 0.35, 0.6))
    m.pixel_sort(street, rng, 8, 0.3, 0.9, area=(0.65, 0, 1, 0.6))
    a = la.riso(street, (30, 40, 120), (226, 140, 40), shift=(8, -6))

    # the sun behind her head
    d = np.sqrt((XX - 0.5) ** 2 + (YY - 0.24) ** 2)
    sun = np.clip((0.2 - d) * 200, 0, 1)[..., None]
    a = a * (1 - sun * 0.85) + np.array([196, 70, 40], np.float32) / 255 * sun * 0.85

    ink = np.array([0.07, 0.06, 0.06])
    a = a * (1 - fig) + ink * fig
    a = a * (1 - veil * 0.9) + la.PAPER * veil * 0.9
    a = a * (1 - scarf) + np.array([40, 190, 170], np.float32) / 255 * scarf
    la.save(la.grain(a, rng, 0.02), "50-soleil", "50", "SOLEIL", "OUAGADOUGOU",
            "12.37° N  1.52° W", "RISOGRAPHIE / SILHOUETTE")


def echo():
    """Le camion de nuit: l'image se répète vers le point de fuite, les lumières s'étirent."""
    rng = np.random.default_rng(5101)
    src = la.load("11.png", (0, 358, 923, 1280), hide=[(435, 1035, 535, 1070)],
                  patch=[(812, 382, 900, 452)])
    src = m.grade(src, 1.25, 1.15)
    # night grade: shadows to deep blue, lights stay orange
    L = la.lum(src)[..., None]
    a = src * np.array([0.85, 0.95, 1.15]) * (0.6 + L * 0.6)
    a = np.clip(a + (1 - L) * np.array([0.0, 0.02, 0.08]), 0, 1)

    # echoes receding toward the vanishing point
    cx, cy = 0.42, 0.34
    out = a * 0.9
    for k in range(1, 7):
        ghost = scaled(a, 1 - 0.11 * k, cx, cy)
        out = np.maximum(out, ghost * (0.75 - 0.1 * k))
    # anamorphic flares on the lights
    bright = np.clip((la.lum(out) - 0.75) / 0.25, 0, 1)[..., None] * out
    flare = hblur(bright, 140) * np.array([0.6, 0.8, 1.3])
    out = la.bloom(np.clip(out + flare * 1.4, 0, 1), 0.7, 10, 0.5)
    m.rgb_split(out, 5)
    m.slices(out, rng, 6, 70, area=(0, 0.55, 1, 1))
    la.save(la.scanlines(la.grain(out, rng, 0.03)), "51-echo", "51", "ÉCHO", "OUAGADOUGOU",
            "12.37° N  1.52° W", "ÉCHOS / FLARES")


PIECES = dict(soleil=soleil, echo=echo)

if __name__ == "__main__":
    for name in sys.argv[1:] or PIECES:
        PIECES[name]()
