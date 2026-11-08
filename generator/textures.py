"""Procedural textures (PIL).  All images are written as PNG."""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SKIN = (255, 241, 234)
SKIN_SHADOW = (242, 200, 190)
HAIR = (246, 132, 104)
HAIR_DARK = (221, 94, 74)
HAIR_LIGHT = (255, 186, 160)


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _save(img, d, name):
    os.makedirs(d, exist_ok=True)
    img.save(os.path.join(d, name), optimize=True)
    return name


def _ss(size, ss=4, mode="RGBA", color=(0, 0, 0, 0)):
    return Image.new(mode, (size[0] * ss, size[1] * ss), color)


def _down(img, size):
    return img.resize(size, Image.LANCZOS)


def gradient_v(size, stops):
    """Vertical gradient image from list of (t, rgb)."""
    w, h = size
    arr = np.zeros((h, w, 3))
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    y = np.linspace(0, 1, h)
    for k in range(3):
        arr[:, :, k] = np.interp(y, ts, cs[:, k])[:, None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
def toon(d, name, shadow, edge=0.48, soft=0.06):
    """MMD toon ramp: top = lit, bottom = shadow."""
    h = 64
    y = np.linspace(0, 1, h)
    t = np.clip((y - (edge - soft)) / (2 * soft), 0, 1)
    t = t * t * (3 - 2 * t)
    arr = np.zeros((h, 16, 3))
    for k in range(3):
        arr[:, :, k] = (255 * (1 - t) + shadow[k] * t)[:, None]
    return _save(Image.fromarray(arr.astype(np.uint8), "RGB"), d, name)


def solid(d, name, rgb, size=8):
    return _save(Image.new("RGB", (size, size), rgb), d, name)


def face(d, name="face.png", size=1024):
    """Planar-projected face skin.  u = 0.5 + x/2.4, v = (19.7 - y)/3.4."""
    img = Image.new("RGB", (size, size), SKIN)
    ov = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)

    def uv(x, y):
        return ((0.5 + x / 2.4) * size, (19.7 - y) / 3.4 * size)

    # soft cheek warmth
    for sx in (-1, 1):
        cx, cy = uv(0.50 * sx, 17.42)
        for r, a in ((80, 5), (62, 6), (44, 7)):
            dr.ellipse([cx - r * 1.6, cy - r * 0.8, cx + r * 1.6, cy + r * 0.8], fill=(255, 150, 140, a))
    # nose: tiny shadow stroke on the side + highlight
    nx, ny = uv(0.025, 17.33)
    dr.line([nx, ny - 9, nx + 3, ny + 5], fill=(232, 178, 168, 110), width=3)
    ov = ov.filter(ImageFilter.GaussianBlur(14))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    return _save(img, d, name)


def eye_white(d, name="eye_white.png"):
    img = gradient_v((64, 128), [(0.0, (196, 178, 196)), (0.25, (224, 214, 226)),
                                 (0.5, (250, 248, 250)), (1.0, (255, 255, 255))])
    return _save(img, d, name)


def iris(d, name="iris.png", size=256):
    ss = 4
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S] / (S - 1)
    # elliptical coordinates over the whole texture (patch = iris bbox)
    ex = (xx - 0.5) / 0.5
    ey = (yy - 0.5) / 0.5
    r = np.sqrt(ex ** 2 + ey ** 2)
    col = np.zeros((S, S, 3))
    top = np.array(_hex("#6a2412"), dtype=float)
    mid = np.array(_hex("#d5652a"), dtype=float)
    bot = np.array(_hex("#ffb35c"), dtype=float)
    t = yy
    for k in range(3):
        col[..., k] = np.where(t < 0.5, top[k] + (mid[k] - top[k]) * (t / 0.5),
                               mid[k] + (bot[k] - mid[k]) * ((t - 0.5) / 0.5))
    # dark outer rim
    rim = np.clip((r - 0.80) / 0.2, 0, 1)
    rimc = np.array(_hex("#4a160a"), dtype=float)
    col = col * (1 - rim[..., None]) + rimc * rim[..., None]
    # radial streaks
    ang = np.arctan2(ey, ex)
    streak = 0.08 * np.sin(ang * 23) * np.clip((r - 0.35) / 0.4, 0, 1) * (1 - rim)
    col *= (1 + streak[..., None])
    # pupil
    px = (xx - 0.5) / 0.17
    py = (yy - 0.47) / 0.27
    pr = np.sqrt(px ** 2 + py ** 2)
    pupil = np.clip((1.15 - pr) / 0.3, 0, 1)
    pc = np.array(_hex("#3a1007"), dtype=float)
    col = col * (1 - 0.92 * pupil[..., None]) + pc * 0.92 * pupil[..., None]
    # glow ring under the pupil
    gx = (xx - 0.5) / 0.32
    gy = (yy - 0.78) / 0.16
    glow = np.clip(1 - np.sqrt(gx ** 2 + gy ** 2), 0, 1) ** 1.5
    col = col + (np.array(_hex("#ffe08a"), float) - col) * (0.75 * glow)[..., None]
    # upper lash shadow
    sh = np.clip((0.36 - yy) / 0.36, 0, 1) ** 1.2
    col *= (1 - 0.55 * sh)[..., None]
    alpha = np.clip((1.0 - r) * S * 0.25, 0, 1) * 255
    img = np.dstack([col.clip(0, 255), alpha]).astype(np.uint8)
    im = Image.fromarray(img, "RGBA")
    dr = ImageDraw.Draw(im)
    # highlights
    def ell(cx, cy, rx, ry, fill):
        dr.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=fill)
    ell(0.5, 0.86, 0.10, 0.025, (255, 248, 220, 140))
    return _save(_down(im, (size, size)), d, name)


def highlight(d, name="highlight.png", size=64):
    ss = 4
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S] / (S - 1)
    r = np.sqrt(((xx - 0.5) / 0.5) ** 2 + ((yy - 0.5) / 0.5) ** 2)
    a = np.clip((1 - r) * 12, 0, 1) * 255
    img = np.dstack([np.full((S, S), 255), np.full((S, S), 255), np.full((S, S), 252), a]).astype(np.uint8)
    return _save(_down(Image.fromarray(img, "RGBA"), (size, size)), d, name)


PALETTE = [
    ["#3b1a14", "#7a3a2c", "#c4553c", "#9a3b36"],
    ["#a0503e", "#fbf8f4", "#2a2228", "#e8833e"],
    ["#d9a441", "#f6f2ee", "#ef7a3f", "#5a2a1c"],
    ["#ffffff", "#000000", "#888888", "#cccccc"],
]


def palette(d, name="lines.png", size=64):
    """4x4 colour swatches; sample at ((i + .5)/4, (j + .5)/4)."""
    im = Image.new("RGB", (size, size))
    dr = ImageDraw.Draw(im)
    c = size // 4
    for j, row in enumerate(PALETTE):
        for i, col in enumerate(row):
            dr.rectangle([i * c, j * c, (i + 1) * c - 1, (j + 1) * c - 1], fill=_hex(col))
    return _save(im, d, name)


def mouth_inside(d, name="mouth.png", size=128):
    img = gradient_v((size, size), [(0.0, (120, 40, 44)), (0.55, (150, 52, 56)), (0.75, (214, 104, 110)),
                                    (1.0, (226, 120, 124))])
    return _save(img, d, name)


def blush(d, name="blush.png", size=256):
    ss = 2
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S] / (S - 1)
    r = np.sqrt(((xx - 0.5) / 0.5) ** 2 + ((yy - 0.5) / 0.5) ** 2)
    a = np.clip(1 - r, 0, 1) ** 1.3 * 150
    col = np.zeros((S, S, 4))
    col[..., 0], col[..., 1], col[..., 2] = 255, 120, 125
    col[..., 3] = a
    im = Image.fromarray(col.astype(np.uint8), "RGBA")
    dr = ImageDraw.Draw(im)
    # diagonal hatch strokes
    for i in range(5):
        x0 = (0.28 + i * 0.11) * S
        dr.line([x0, 0.36 * S, x0 - 0.06 * S, 0.62 * S], fill=(236, 90, 100, 200), width=int(0.018 * S))
    return _save(_down(im, (size, size)), d, name)


def hair(d, name="hair.png", w=256, h=512, seed=3):
    """Soft anime hair clumps.  Two clumps across u (period 0.5): a strand's outer
    face (u 0..0.5) shows one rounded clump with darker edges."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, w, endpoint=False)
    f = (2 * u) % 1.0
    clump = np.sin(np.pi * f) ** 0.7                      # 0 at the edges, 1 in the middle
    fine = np.zeros(w)
    for fr, a in ((6, 0.5), (14, 0.3), (26, 0.2)):
        fine += a * np.sin(2 * np.pi * (fr * u + rng.random()))
    fine /= np.abs(fine).max()
    v = np.linspace(0, 1, h)[:, None]
    base = np.array(HAIR, float)
    dark = np.array(HAIR_DARK, float)
    light = np.array(HAIR_LIGHT, float)
    t = (0.30 + 0.52 * clump + 0.18 * (0.5 + 0.5 * fine))[None, :]   # 0 dark .. 1 light
    col = np.zeros((h, w, 3))
    for k in range(3):
        c = dark[k] + (base[k] - dark[k]) * np.clip(t / 0.6, 0, 1)
        c = c + (light[k] - base[k]) * np.clip((t - 0.75) / 0.25, 0, 1) * 0.55
        c = c * (0.94 + 0.06 * np.clip(v / 0.25, 0, 1))                       # darker roots
        c = c + (light[k] - c) * 0.20 * np.clip((v - 0.65) / 0.35, 0, 1)      # lighter ends
        col[:, :, k] = c
    img = Image.fromarray(col.clip(0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.8))
    return _save(img, d, name)


def hair_sphere(d, name="hair_s.png", size=128):
    """Additive sphere map: soft horizontal highlight band."""
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    band = np.exp(-((yy - 0.32) / 0.07) ** 2) * np.clip(1 - np.abs(xx - 0.5) * 1.6, 0, 1)
    c = (band * 70)[..., None] * np.array([1.0, 0.85, 0.75])
    return _save(Image.fromarray(c.clip(0, 255).astype(np.uint8), "RGB"), d, name)


def satin_sphere(d, name="satin_s.png", size=128):
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    r = np.sqrt((xx - 0.42) ** 2 + (yy - 0.35) ** 2)
    c = np.clip(1 - r / 0.45, 0, 1) ** 2 * 55
    c = np.dstack([c, c, c * 1.05])
    return _save(Image.fromarray(c.clip(0, 255).astype(np.uint8), "RGB"), d, name)


def sock(d, name="sock.png", w=512, h=512):
    """u around the leg (0 = back seam, 0.5 = front), v from top (sock band) to ankle."""
    ss = 2
    im = Image.new("RGB", (w * ss, h * ss), (250, 248, 246))
    dr = ImageDraw.Draw(im)
    W, H = w * ss, h * ss
    # ribbed band at top
    dr.rectangle([0, 0, W, int(0.08 * H)], fill=(244, 241, 239))
    for i in range(0, W, 10 * ss):
        dr.line([i, 0, i, int(0.08 * H)], fill=(230, 226, 224), width=ss * 2)
    orange = (238, 128, 64)
    # vertical chain of diamonds on the outer side and front
    for cu in (0.5,):
        cx = cu * W
        y = 0.16 * H
        while y < 0.92 * H:
            dw, dh = 0.035 * W, 0.045 * H
            dr.polygon([(cx, y - dh), (cx + dw, y), (cx, y + dh), (cx - dw, y)], fill=orange)
            y += 0.14 * H
    for cu in (0.25, 0.75):
        cx = cu * W
        y = 0.23 * H
        while y < 0.92 * H:
            dw, dh = 0.02 * W, 0.03 * H
            dr.polygon([(cx, y - dh), (cx + dw, y), (cx, y + dh), (cx - dw, y)], fill=(244, 160, 100))
            y += 0.14 * H
    return _save(_down(im, (w, h)), d, name)


def petals(d, name="flower.png", size=128):
    img = gradient_v((size, size), [(0.0, (236, 92, 48)), (0.5, (246, 128, 64)), (1.0, (255, 176, 112))])
    return _save(img, d, name)


def flower_center(d, name="flower_c.png", size=128):
    ss = 2
    S = size * ss
    yy, xx = np.mgrid[0:S, 0:S] / (S - 1)
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 2
    col = np.zeros((S, S, 3))
    a = np.array([255, 222, 170.0])
    b = np.array([176, 92, 50.0])
    t = np.clip(r, 0, 1)[..., None]
    col = a * (1 - t) + b * t
    dots = (np.sin(xx * 60) * np.sin(yy * 60) > 0.6) & (r < 0.9)
    col[dots] *= 0.8
    return _save(Image.fromarray(col.clip(0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS), d, name)


def build_all(d):
    names = {}
    names["toon_skin"] = toon(d, "toon_skin.png", (246, 206, 198), edge=0.50)
    names["toon_hair"] = toon(d, "toon_hair.png", (234, 158, 150), edge=0.50)
    names["toon_cloth"] = toon(d, "toon_cloth.png", (214, 210, 226), edge=0.50)
    names["toon_dark"] = toon(d, "toon_dark.png", (170, 166, 180), edge=0.52)
    names["toon_flat"] = toon(d, "toon_flat.png", (250, 248, 248), edge=0.5)
    names["face"] = face(d)
    names["eye_white"] = eye_white(d)
    names["iris"] = iris(d)
    names["mouth"] = mouth_inside(d)
    names["highlight"] = highlight(d)
    names["lines"] = palette(d)
    names["blush"] = blush(d)
    names["hair"] = hair(d)
    names["hair_s"] = hair_sphere(d)
    names["satin_s"] = satin_sphere(d)
    names["sock"] = sock(d)
    names["flower"] = petals(d)
    names["flower_c"] = flower_center(d)
    return names
