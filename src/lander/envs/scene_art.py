"""Shared pygame art for the viewer's scenes: drawing helpers, cached glow sprites,
the lunar sky backdrop (stars, Earth, ridges) and the Apollo-LM-style lander sprite."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pygame
import pygame.gfxdraw


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def ipts(pts):
    return [(round(x), round(y)) for x, y in pts]


def aapoly(surf, color, pts) -> None:
    pts = ipts(pts)
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    pygame.gfxdraw.aapolygon(surf, pts, color)


def aaline(surf, color, p1, p2, width: float) -> None:
    """Thick anti-aliased line with round caps."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    nx, ny = -(y2 - y1) / length * width / 2, (x2 - x1) / length * width / 2
    aapoly(surf, color, [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)])
    r = int(width / 2)
    if r >= 1:
        for x, y in (p1, p2):
            pygame.gfxdraw.filled_circle(surf, int(x), int(y), r, color)
            pygame.gfxdraw.aacircle(surf, int(x), int(y), r, color)


# --- cached soft sprites -----------------------------------------------------

_GLOW: dict = {}
_SOFT: dict = {}


def glow(radius: int, color: tuple[int, int, int], squash: float = 1.0) -> pygame.Surface:
    """Radial gradient on black, for additive blending (BLEND_ADD)."""
    key = (radius, color, squash)
    if key not in _GLOW:
        ry = max(1, int(radius * squash))
        yy, xx = np.mgrid[-ry : ry + 1, -radius : radius + 1]
        f = np.clip(1 - np.sqrt((xx / radius) ** 2 + (yy / ry) ** 2), 0, 1) ** 2
        arr = (np.asarray(color, float)[None, None, :] * f[..., None]).astype(np.uint8)
        _GLOW[key] = pygame.surfarray.make_surface(np.ascontiguousarray(arr.transpose(1, 0, 2)))
    return _GLOW[key]


def soft(radius: int, color: tuple[int, int, int], alpha: int, squash: float = 1.0) -> pygame.Surface:
    """Soft-edged disc with per-pixel alpha, for normal blending (dust, shadows)."""
    key = (radius, color, alpha, squash)
    if key not in _SOFT:
        ry = max(1, int(radius * squash))
        surf = pygame.Surface((2 * radius + 1, 2 * ry + 1), pygame.SRCALPHA)
        surf.fill((*color, 0))
        yy, xx = np.mgrid[-ry : ry + 1, -radius : radius + 1]
        f = np.clip(1 - np.sqrt((xx / radius) ** 2 + (yy / ry) ** 2), 0, 1) ** 1.5
        a = pygame.surfarray.pixels_alpha(surf)
        a[:] = (f * alpha).T.astype(np.uint8)
        del a
        _SOFT[key] = surf
    return _SOFT[key]


def blit_add(surf, sprite, center) -> None:
    surf.blit(sprite, sprite.get_rect(center=(int(center[0]), int(center[1]))), special_flags=pygame.BLEND_ADD)


def blit(surf, sprite, center) -> None:
    surf.blit(sprite, sprite.get_rect(center=(int(center[0]), int(center[1]))))


# Fire colour over a particle's life (0 = just born), pre-darkened for additive blending.
FIRE_STOPS = [(0.0, (255, 245, 220)), (0.2, (255, 205, 120)), (0.5, (255, 125, 45)), (1.0, (60, 12, 4))]


def fire_color(t: float, intensity: float = 1.0) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in itertools.pairwise(FIRE_STOPS):
        if t <= t1:
            c = lerp(c0, c1, (t - t0) / (t1 - t0))
            break
    fade = (1 - t) ** 0.8 * intensity
    return tuple(int(v * fade) for v in c)


FIRE_RAMP = [fire_color(i / 11) for i in range(12)]
RCS_RAMP = [tuple(int(v * (1 - i / 8) ** 1.2) for v in (200, 220, 255)) for i in range(8)]


def draw_twinkle(canvas: pygame.Surface, twinkle: list, t: float) -> None:
    for x, y, b, ph, fr in twinkle:
        k = 0.55 + 0.45 * math.sin(t * fr + ph)
        v = int(b * k)
        blit_add(canvas, glow(3, (v, v, min(255, v + 20))), (x, y))


def make_sky(w: int = 1200, h: int = 800) -> tuple[pygame.Surface, list]:
    """Starfield with a Milky Way band, Earth, and two shaded ridge lines.
    Returns (surface, twinkle) where twinkle lists stars for draw_twinkle()."""
    rng = random.Random(42)
    # vertical gradient + a faint diagonal Milky Way haze
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    t = (yy / h)[..., None]
    top, bottom = np.array([3, 4, 12.0]), np.array([18, 20, 38.0])
    img = top * (1 - t) + bottom * t
    band_d = (yy - (260 - xx * 0.18)) / 75.0
    haze = np.exp(-band_d**2)[..., None] * np.array([16, 16, 26.0])
    img = np.clip(img + haze, 0, 255).astype(np.uint8)
    sky = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))

    ridge_far = ridge(rng, w, 350, [(30, 150), (17, 60), (8, 23)])
    ridge_near = ridge(rng, w, 430, [(26, 115), (13, 43), (6, 17)])

    # stars: a uniform field plus a denser band along the haze; only above the far ridge
    def star(x, y, b):
        if y >= ridge_far[int(x)] - 4 or y >= 395:
            return
        tint = rng.choice([(200, 215, 255), (255, 244, 225), (255, 255, 255), (220, 230, 255)])
        sky.set_at((int(x), int(y)), lerp((0, 0, 0), tint, b / 255))
        if b > 200:
            blit_add(sky, glow(3, tuple(int(c * 0.35) for c in tint)), (x, y))

    for _ in range(330):
        star(rng.uniform(0, w - 1), rng.uniform(0, 400), 40 + 215 * rng.random() ** 3)
    for _ in range(650):
        x = rng.uniform(0, w - 1)
        star(x, 260 - x * 0.18 + rng.gauss(0, 45), 25 + 90 * rng.random() ** 2)
    twinkle = []
    while len(twinkle) < 22:
        x, y = rng.uniform(20, w - 20), rng.uniform(10, 300)
        if y < ridge_far[int(x)] - 10 and math.hypot(x - 800, y - 112) > 50:
            twinkle.append((x, y, rng.uniform(120, 230), rng.uniform(0, 6.3), rng.uniform(1.5, 4.0)))

    earth = make_earth(30)
    sky.blit(earth, earth.get_rect(center=(800, 112)))

    fill_ridge(sky, ridge_far, (26, 27, 39), (52, 54, 70))
    fill_ridge(sky, ridge_near, (35, 36, 48), (68, 69, 84))
    return sky, twinkle


def ridge(rng, width: int, base: float, waves) -> np.ndarray:
    x = np.arange(width + 1, dtype=float)
    y = np.full_like(x, base)
    for amp, period in waves:
        y += amp * np.sin(x / period + rng.uniform(0, 2 * math.pi))
    return y


def fill_ridge(surf, ridge, fill, crest) -> None:
    """Mountain silhouette: lit crest fading into shadow below, with an AA edge."""
    w, h = surf.get_size()
    top = ridge[:w, None]
    yy = np.arange(h, dtype=float)[None, :]
    below = yy - top
    t = np.clip(below / 70.0, 0, 1) ** 0.6
    col = np.asarray(crest, float) * (1 - t[..., None]) + np.asarray(fill, float) * t[..., None]
    layer = pygame.Surface((w, h), pygame.SRCALPHA)
    pix = pygame.surfarray.pixels3d(layer)
    pix[:] = col.astype(np.uint8)
    del pix
    a = pygame.surfarray.pixels_alpha(layer)
    a[:] = (np.clip(below + 0.5, 0, 1) * 255).astype(np.uint8)
    del a
    surf.blit(layer, (0, 0))


def make_earth(r: int) -> pygame.Surface:
    pad = 8
    n = 2 * (r + pad) + 1
    yy, xx = np.mgrid[-(r + pad) : r + pad + 1, -(r + pad) : r + pad + 1].astype(float)
    nx, ny = xx / r, -yy / r
    d2 = nx**2 + ny**2
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    light = np.clip(nx * -0.72 + ny * 0.42 + nz * 0.55, 0, 1)
    # cheap "continents" and "clouds" from overlapping sinusoids on the sphere
    wx = nx + 0.35 * np.sin(ny * 3.1 + 0.6)  # domain warp keeps shapes from looking like a grid
    wy = ny + 0.35 * np.sin(nx * 2.7 + 1.9)
    land = np.sin(wx * 2.3 + 1.3) + np.sin(wy * 2.9 + nz * 1.7 + 0.7) + 0.6 * np.sin((wx - wy) * 4.7 + 2.2)
    cloud = np.sin(ny * 13 + 2.2 * np.sin(nx * 4.3 + 0.5)) + 0.6 * np.sin((nx + 0.5 * ny) * 7.1 + 1.0)
    col = np.zeros((n, n, 3))
    col[:] = (28, 74, 158)
    col[land > 1.05] = (66, 108, 64)
    col[land > 1.75] = (146, 128, 88)
    cmask = np.clip((cloud - 0.7) * 1.8, 0, 0.8)[..., None]
    col = col * (1 - cmask) + np.array([235, 240, 245.0]) * cmask
    col *= (0.03 + 0.97 * light)[..., None]
    # atmosphere: blue rim, brighter on the lit side
    d = np.sqrt(d2)
    rim = np.exp(-(((d - 1.0) / 0.07) ** 2)) * (0.25 + 0.75 * np.clip(nx * -0.72 + ny * 0.42 + 0.35, 0, 1))
    col += rim[..., None] * np.array([60, 120, 255.0])
    alpha = np.clip((1.0 - d) * r + 0.5, 0, 1) * 255
    alpha = np.maximum(alpha, rim * 200)
    surf = pygame.Surface((n, n), pygame.SRCALPHA)
    pix = pygame.surfarray.pixels3d(surf)
    pix[:] = np.clip(col, 0, 255).astype(np.uint8).transpose(1, 0, 2)
    del pix
    a = pygame.surfarray.pixels_alpha(surf)
    a[:] = alpha.astype(np.uint8).T
    del a
    return surf


def make_lander_sprite(px_per_u: float) -> pygame.Surface:
    """Apollo-LM-style body (descent + ascent stage, engine bell) in the LunarLander
    body frame (U units, ~34 U wide), drawn once at 4x and downsampled for anti-aliasing.
    , downsample
    for anti-aliasing. Rotated per frame; legs are drawn separately because
    they move on their joints."""
    ss = 4
    k = px_per_u * ss  # px per U at supersampled resolution
    ext_x, ext_y = 22, 24  # sprite covers x in [-22, 22] U, y in [-24, 24] U, centred on the body origin
    big = pygame.Surface((int(2 * ext_x * k), int(2 * ext_y * k)), pygame.SRCALPHA)

    def P(pts):
        return [((x + ext_x) * k, (ext_y - y) * k) for x, y in pts]

    def poly(color, pts, outline=None):
        pygame.draw.polygon(big, color, P(pts))
        if outline:
            pygame.draw.polygon(big, outline, P(pts), width=max(1, int(0.45 * k)))

    def line(color, a, b, wu):
        pygame.draw.line(big, color, P([a])[0], P([b])[0], max(1, int(wu * k)))

    outline = (22, 22, 26)
    # engine bell (stays above the footpads, which sit at about y = -13.5 U)
    poly((80, 78, 84), [(-3, -10), (3, -10), (5.5, -13), (-5.5, -13)], outline)
    poly((58, 56, 62), [(0, -10), (3, -10), (5.5, -13), (0, -13)])
    line((140, 138, 146), (-5.5, -13), (5.5, -13), 0.6)

    # descent stage: gold foil octagon
    oct_ = [(-17, -6), (-14, -10.5), (14, -10.5), (17, -6), (17, 1.5), (14, 3.5), (-14, 3.5), (-17, 1.5)]
    foil = pygame.Surface(big.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(foil, (204, 158, 58), P(oct_))
    rng = random.Random(3)
    for _ in range(260):  # crinkled foil: lots of small facets in slightly different golds
        cx, cy = rng.uniform(-17, 17), rng.uniform(-10.5, 3.5)
        sz = rng.uniform(1.2, 3.5)
        shade = lerp((168, 122, 38), (246, 208, 112), rng.random())
        tri = [(cx + rng.uniform(-sz, sz), cy + rng.uniform(-sz, sz)) for _ in range(3)]
        pygame.draw.polygon(foil, shade, P(tri))
    mask = pygame.Surface(big.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(mask, (255, 255, 255, 255), P(oct_))
    foil.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    big.blit(foil, (0, 0))
    shade = pygame.Surface(big.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(shade, (0, 0, 0, 70), P([(6, -10.5), (14, -10.5), (17, -6), (17, 1.5), (14, 3.5), (6, 3.5)]))
    pygame.draw.polygon(shade, (255, 255, 230, 40), P([(-17, -6), (-14, -10.5), (-8, -10.5), (-8, 3.5), (-14, 3.5), (-17, 1.5)]))
    big.blit(shade, (0, 0))
    for x in (-8, 6):
        line((130, 96, 30), (x, -10.2), (x, 1.5), 0.35)
    poly((44, 40, 36), [(-16, 1.5), (16, 1.5), (14, 3.5), (-14, 3.5)])
    pygame.draw.polygon(big, outline, P(oct_), width=max(1, int(0.45 * k)))

    # ascent stage: faceted silver cabin
    cabin = [(-13, 3.5), (-14.5, 9), (-10, 15.5), (-3, 17), (7, 17), (12.5, 12), (13, 3.5)]
    poly((192, 195, 203), cabin)
    poly((148, 152, 162), [(4, 3.5), (13, 3.5), (12.5, 12), (7, 17), (4, 17)])
    poly((218, 221, 228), [(-14.5, 9), (-10, 15.5), (-5, 16.6), (-8.5, 9)])
    poly((92, 92, 100), [(-13, 3.5), (13, 3.5), (13, 5.5), (-13.4, 5.5)])
    poly((168, 171, 180), [(-3, 5.5), (1.5, 5.5), (1.5, 10), (-3, 10)], (110, 112, 122))  # hatch
    poly((20, 30, 56), [(-11.3, 9.4), (-7.2, 14.2), (-4.4, 9.4)], (40, 44, 60))  # window
    line((130, 180, 240), (-9.6, 10.6), (-7.6, 13.0), 0.5)
    poly((118, 120, 130), [(-2.5, 17), (2.5, 17), (2.5, 19), (-2.5, 19)], outline)  # docking tunnel
    pygame.draw.polygon(big, outline, P(cabin), width=max(1, int(0.45 * k)))
    # RCS thruster quads
    for sx in (-1, 1):
        x0 = 14.2 * sx
        poly((112, 112, 120), [(x0 - 1.4, 9.5), (x0 + 1.4, 9.5), (x0 + 1.4, 12.5), (x0 - 1.4, 12.5)], outline)
        line((70, 70, 76), (x0 + sx * 1.4, 11), (x0 + sx * 2.6, 11), 0.6)
    # antennas
    line((176, 176, 184), (7.5, 17), (10, 21.2), 0.5)
    pygame.draw.circle(big, (226, 228, 234), P([(10.3, 21.6)])[0], int(1.9 * k))
    pygame.draw.circle(big, (150, 152, 160), P([(10.3, 21.6)])[0], int(1.9 * k), width=max(1, int(0.4 * k)))
    line((160, 160, 168), (-9, 15.6), (-11.5, 19.5), 0.35)

    return pygame.transform.smoothscale(big, (int(big.get_width() / ss), int(big.get_height() / ss)))
