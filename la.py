"""DROITS RÉSERVÉS — série LA / Palm Springs.

Chaque photo reçoit un traitement différent (infrarouge, risographie, thermique,
slit-scan...) et est présentée en affiche: cadre papier, typographie suisse,
coordonnées. Une version plein cadre (sans affiche) est aussi exportée.

    python3 la.py              -> output/la/*.png
    python3 la.py griffith     -> une seule pièce
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import mix as m

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "source", "la")
OUT = os.path.join(ROOT, "output", "la")
FONTS = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-"
S = 1800                         # square artwork
m.W = m.H = S                    # mix.py effects work on module-level W, H
SCREEN = 1290 / 923              # screenshot pixels per displayed point
PAPER = np.array([242, 239, 232], np.float32) / 255
INK = (24, 24, 24)


# ---------------------------------------------------------------- io

def load(name, box, hide=()):
    """Crop a post out of an Instagram screenshot (box/hide in displayed points)."""
    img = Image.open(os.path.join(SRC, name)).convert("RGB")
    a = np.asarray(img, dtype=np.float32) / 255
    a = a.copy()
    for x0, y0, x1, y1 in hide:                  # pixelate logos and posters
        b = 14
        x0, y0 = int(x0 * SCREEN) // b * b, int(y0 * SCREEN) // b * b
        w = (int(x1 * SCREEN) - x0) // b * b + b
        h = (int(y1 * SCREEN) - y0) // b * b + b
        blk = a[y0:y0 + h, x0:x0 + w]
        mm = blk.reshape(h // b, b, w // b, b, 3).mean((1, 3))
        blk[:] = np.repeat(np.repeat(mm, b, 0), b, 1)
    x0, y0, x1, y1 = (int(v * SCREEN) for v in box)
    a = a[y0:y1, x0:x1]
    h, w = a.shape[:2]
    s = min(h, w)
    a = a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = Image.fromarray((a * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255


def lum(a):
    return m.lum(a)


def blur_rgb(a, r):
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255


def gradient(L, stops):
    pos = np.array([p for p, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float32) / 255
    return np.stack([np.interp(L, pos, cols[:, i]) for i in range(3)], -1)


def grain(a, rng, s):
    return np.clip(a + rng.normal(0, s, (S, S, 1)).astype(np.float32), 0, 1)


# ---------------------------------------------------------------- treatments

def aerochrome(a):
    """Film infrarouge couleur simulé: la végétation devient rose-rouge."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    veg = np.clip((G - R) * 5, 0, 1) * np.clip((G - B) * 8 + 0.3, 0, 1)   # foliage, not sky
    veg = m.blur(veg, 2)[..., None]
    ir = np.stack([np.clip(G * 1.25 + 0.15, 0, 1), R * 0.35, B * 0.55], -1)  # foliage -> red
    sky = np.stack([R * 0.75, G * 0.95, np.clip(B * 1.1, 0, 1)], -1)         # deeper cyan
    out = sky * (1 - veg) + ir * veg
    return m.grade(out, 1.15, 1.3)


def halftone(L, angle, cell):
    """Dot screen: returns ink coverage 0/1 for a darkness map L (1 = full ink)."""
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    t = np.deg2rad(angle)
    u = (xx * np.cos(t) + yy * np.sin(t)) * np.pi / cell
    v = (-xx * np.sin(t) + yy * np.cos(t)) * np.pi / cell
    screen = (np.cos(u) * np.cos(v) + 1) / 2                  # 0..1
    return (L > screen).astype(np.float32)


def riso(a, c1, c2, shift=(9, -6)):
    """Risographie deux encres, trame et léger défaut de repérage."""
    L = lum(a)
    d1 = halftone(np.clip((1 - L) * 1.15, 0, 1), 15, 7)
    warm = np.clip((a[..., 0] - a[..., 2]) * 2.2 + (1 - L) * 0.4, 0, 1)
    d2 = halftone(warm, 75, 7)
    d2 = np.roll(d2, shift, (0, 1))
    k1, k2 = np.array(c1, np.float32) / 255, np.array(c2, np.float32) / 255
    out = PAPER * (1 - d1[..., None] * (1 - k1))
    return out * (1 - d2[..., None] * (1 - k2))


def trail(a, y0, y1, length, decay=0.93):
    """Slit-scan: la bande laisse une traînée vers la gauche."""
    band = a[y0:y1].copy()
    acc = band.copy()
    for _ in range(length):
        band = np.roll(band, 2, 1) * decay + band * (1 - decay)
        acc = np.maximum(acc, band) * 0.5 + acc * 0.5
    a[y0:y1] = acc


def bloom(a, thresh, r, k):
    bright = np.clip((lum(a) - thresh) / (1 - thresh), 0, 1)[..., None] * a
    return np.clip(a + blur_rgb(bright, r) * k, 0, 1)


def scanlines(a, period=4, depth=0.12):
    rows = (np.arange(S) % period == 0).astype(np.float32)[:, None, None]
    return a * (1 - rows * depth)


def veils(a, rng, tones, n_big=3, n_small=6, op=(0.3, 0.55)):
    return m.veils(a, rng, tones, n_big, n_small, op)


# ---------------------------------------------------------------- poster

def poster(art, num, title, place, coords, note):
    """Affiche 2000 x 2300: image carrée sur papier, cartouche typographique."""
    PW, PH, M = 2000, 2300, 100
    page = Image.new("RGB", (PW, PH), tuple(int(c * 255) for c in PAPER))
    page.paste(Image.fromarray((np.clip(art, 0, 1) * 255).astype(np.uint8)), (M, M))
    d = ImageDraw.Draw(page)
    big = ImageFont.truetype(FONTS + "ExtraBold.ttf", 120)
    mid = ImageFont.truetype(FONTS + "Bold.ttf", 34)
    small = ImageFont.truetype(FONTS + "Regular.ttf", 26)
    y = M + S + 60
    d.line([(M, y - 25), (PW - M, y - 25)], fill=INK, width=3)
    d.text((M, y), f"#{num}", font=big, fill=INK)
    d.text((M + 300, y + 8), title, font=big, fill=INK)
    d.text((M, y + 170), "DROITS RÉSERVÉS", font=mid, fill=INK)
    d.text((M + 420, y + 170), place, font=mid, fill=INK)
    for i, line in enumerate((coords, note)):
        w = d.textlength(line, font=small)
        d.text((PW - M - w, y + 12 + i * 40), line, font=small, fill=INK)
    d.rectangle([PW - M - 60, y + 160, PW - M, y + 220], fill=(214, 32, 38))
    return page


def save(art, stem, *meta):
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray((np.clip(art, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(OUT, stem + "-plein-cadre.png"), optimize=True)
    path = os.path.join(OUT, stem + ".png")
    poster(art, *meta).save(path, optimize=True)
    print(path)


# ---------------------------------------------------------------- pieces

def downtown():
    rng = np.random.default_rng(3601)
    a = load("01.png", (0, 700, 923, 1525),
             hide=[(740, 700, 830, 790), (120, 895, 175, 925), (180, 1385, 250, 1480)])
    a = aerochrome(a)
    m.pixel_sort(a, rng, 10, 0.45, 0.92, area=(0.3, 0.1, 0.65, 0.75))
    m.slices(a, rng, 5, 50, area=(0, 0.6, 1, 1))
    a = grain(a, rng, 0.03)
    save(a, "36-downtown", "36", "INFRAROUGE", "DOWNTOWN LOS ANGELES",
         "34.05° N  118.25° W", "AEROCHROME / PIXEL SORT")


def griffith():
    rng = np.random.default_rng(3701)
    a = m.grade(load("02.png", (0, 630, 923, 1455)), 1.2)
    m.pixel_sort(a, rng, 8, 0.35, 0.8, area=(0, 0, 1, 0.45))
    a = riso(a, (255, 72, 176), (0, 120, 191))
    a = grain(a, rng, 0.02)
    save(a, "37-griffith", "37", "OBSERVATOIRE", "GRIFFITH PARK",
         "34.12° N  118.30° W", "RISOGRAPHIE 2 ENCRES")


def tomcat():
    rng = np.random.default_rng(3801)
    a = m.grade(load("04.png", (0, 600, 923, 1418)), 1.15, 0.9)
    trail(a, int(0.38 * S), int(0.62 * S), 40)
    m.rgb_split(a, 7)
    m.slices(a, rng, 7, 90, area=(0, 0.35, 1, 0.7))
    a = scanlines(grain(a, rng, 0.03))
    save(a, "38-tomcat", "38", "VITESSE", "PALM SPRINGS AIR MUSEUM",
         "33.83° N  116.51° W", "SLIT-SCAN / CHROMA")


def desert():
    rng = np.random.default_rng(3901)
    a = m.grade(load("05.png", (0, 603, 923, 1522)), 1.3)
    m.pixel_sort(a, rng, 12, 0.3, 0.85, vertical=False, area=(0, 0.25, 1, 0.6))
    L = m.blur(lum(a), 1.2)
    a = gradient(L, [(0, (10, 0, 30)), (0.25, (90, 10, 120)), (0.45, (220, 40, 60)),
                     (0.65, (255, 140, 20)), (0.85, (255, 230, 90)), (1, (255, 255, 240))])
    a = scanlines(grain(a, rng, 0.025), 3, 0.1)
    save(a, "39-desert", "39", "CHALEUR", "PALM SPRINGS",
         "33.83° N  116.55° W", "IMAGE THERMIQUE")


def piscine():
    rng = np.random.default_rng(4001)
    a = m.grade(load("06.png", (0, 425, 923, 1343)), 1.15, 1.3)
    m.pixel_sort(a, rng, 12, 0.55, 0.95, vertical=False, area=(0.5, 0.55, 1, 0.85))
    m.smear(a, rng, 6, area=(0, 0.2, 1, 0.7))
    a = veils(a, rng, [(255, 150, 190), (255, 210, 80), (80, 200, 220), (255, 255, 255)])
    a = grain(a, rng, 0.025)
    save(a, "40-piscine", "40", "PISCINE", "PALM SPRINGS",
         "33.82° N  116.54° W", "PIXEL SORT / VOILES")


def skyline():
    rng = np.random.default_rng(4101)
    a = load("07.png", (0, 557, 923, 1477), hide=[(745, 880, 800, 900)])
    m.pixel_sort(a, rng, 14, 0.25, 0.75, area=(0.1, 0.15, 0.95, 0.7))
    L = lum(m.grade(a, 1.45))
    a = np.repeat(L[..., None], 3, -1)
    x0, y0 = int(0.42 * S), int(0.18 * S)
    a[y0:y0 + int(0.42 * S), x0:x0 + int(0.2 * S)] *= np.array([1.0, 0.25, 0.25])
    a = grain(a, rng, 0.045)
    save(a, "41-skyline", "41", "SKYLINE", "LOS ANGELES",
         "34.04° N  118.24° W", "NOIR / UN ROUGE")


def mulholland():
    rng = np.random.default_rng(4201)
    a = m.grade(load("08.png", (0, 632, 923, 1550)), 1.2, 1.3)
    a = bloom(a, 0.55, 12, 0.9)
    lights = (0, int(0.45 * S), S, int(0.4 * S))
    for _ in range(60):                                       # lights stretched upward
        x = int(rng.integers(0, S - 8))
        w = int(rng.integers(2, 10))
        m.sort_runs(a, x, int(0.3 * S), w, int(0.45 * S), 0.12, 1.0, min_run=4)
    m.rgb_split(a, 4)
    a = scanlines(grain(a, rng, 0.03))
    save(a, "42-mulholland", "42", "NUIT", "MULHOLLAND DRIVE",
         "34.13° N  118.35° W", "BLOOM / PIXEL SORT")


PIECES = dict(downtown=downtown, griffith=griffith, tomcat=tomcat, desert=desert,
              piscine=piscine, skyline=skyline, mulholland=mulholland)

if __name__ == "__main__":
    for name in sys.argv[1:] or PIECES:
        PIECES[name]()
