"""Stylised renderer ("scene") for LunarLander, used by the viewer.

It reads the Box2D state (terrain, pad, lander body and legs) and draws it with
pygame. It is purely cosmetic and never touches the simulation, so you see
exactly what the policy experiences. Engine flames come from the action the
policy chose this step, and the particles (exhaust, dust, explosion) are a
visual-only system.

Coordinates: the env's world is 20 x 13.3 units with y up. The canvas is
1200 x 800 px with y down, 60 px per unit. Body shapes (LANDER_POLY etc.) are
written in the env's 1/SCALE units, called U below.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pygame
import pygame.gfxdraw
from gymnasium.envs.box2d.lunar_lander import LEG_H, SCALE, VIEWPORT_H, VIEWPORT_W

from lander.hud import Gauge

CANVAS_W, CANVAS_H = 1200, 800
WORLD_W, WORLD_H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
PPU = CANVAS_W / WORLD_W  # px per world unit
U = 1 / SCALE  # body-shape unit, in world units
PX_U = PPU * U  # px per U

GRAVITY = -6.0  # for dust/debris; a bit floatier than the sim's -10 reads better
SUN = (-0.55, 0.83)  # light comes from the upper left

# particle kinds
FIRE, RCS, DUST, DEBRIS, BLAST, SMOKE = range(6)


def to_screen(x: float, y: float) -> tuple[float, float]:
    return x * PPU, CANVAS_H - y * PPU


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def _ipts(pts):
    return [(round(x), round(y)) for x, y in pts]


def _aapoly(surf, color, pts) -> None:
    pts = _ipts(pts)
    pygame.gfxdraw.filled_polygon(surf, pts, color)
    pygame.gfxdraw.aapolygon(surf, pts, color)


def _aaline(surf, color, p1, p2, width: float) -> None:
    """Thick anti-aliased line with round caps."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    nx, ny = -(y2 - y1) / length * width / 2, (x2 - x1) / length * width / 2
    _aapoly(surf, color, [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)])
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


def _blit_add(surf, sprite, center) -> None:
    surf.blit(sprite, sprite.get_rect(center=(int(center[0]), int(center[1]))), special_flags=pygame.BLEND_ADD)


def _blit(surf, sprite, center) -> None:
    surf.blit(sprite, sprite.get_rect(center=(int(center[0]), int(center[1]))))


# Fire colour over a particle's life (0 = just born), pre-darkened for additive blending.
_FIRE_STOPS = [(0.0, (255, 245, 220)), (0.2, (255, 205, 120)), (0.5, (255, 125, 45)), (1.0, (60, 12, 4))]


def _fire_color(t: float, intensity: float = 1.0) -> tuple[int, int, int]:
    for (t0, c0), (t1, c1) in itertools.pairwise(_FIRE_STOPS):
        if t <= t1:
            c = _lerp(c0, c1, (t - t0) / (t1 - t0))
            break
    fade = (1 - t) ** 0.8 * intensity
    return tuple(int(v * fade) for v in c)


FIRE_RAMP = [_fire_color(i / 11) for i in range(12)]
RCS_RAMP = [tuple(int(v * (1 - i / 8) ** 1.2) for v in (200, 220, 255)) for i in range(8)]


class LunarScene:
    size = (CANVAS_W, CANVAS_H)
    render_mode = None  # we draw from physics state; the env needn't render

    def __init__(self) -> None:
        self.rng = random.Random(7)
        self.sky = self._make_sky()
        self.sprite = self._make_lander_sprite()
        self.sprite_dead = self.sprite.copy()
        self.sprite_dead.fill((70, 60, 55, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.background = self.sky
        self.particles: list[list] = []
        self.terrain_x: list[float] = []
        self.terrain_y: list[float] = []
        self.t = 0.0
        self.action = (0.0, 0.0)
        self.dead = False
        self.pad_ok = False
        self.flash = 0.0

    # --- protocol used by the viewer ----------------------------------------

    def reset(self, env) -> None:
        u = env.unwrapped
        pts = [p[0] for p in u.sky_polys] + [u.sky_polys[-1][1]]
        self.terrain_x = [p[0] for p in pts]
        self.terrain_y = [p[1] for p in pts]
        self.pad = (u.helipad_x1, u.helipad_x2, u.helipad_y)
        self.background = self._make_background()
        self.particles.clear()
        self.t, self.action, self.dead, self.pad_ok, self.flash = 0.0, (0.0, 0.0), False, False, 0.0

    def update(self, env, action, dt: float) -> None:
        """Advance visual effects by dt. ``action`` is None between episodes."""
        self.t += dt
        self.action = (0.0, 0.0) if action is None else (float(action[0]), float(action[1]))
        if action is not None and not self.dead:
            self._spawn_thrust(env.unwrapped)
        self.flash = max(0.0, self.flash - dt * 1.4)
        alive = []
        for p in self.particles:
            p[4] += dt
            if p[4] >= p[5]:
                continue
            if p[4] < 0:  # not started yet (staggered smoke)
                alive.append(p)
                continue
            kind = p[6]
            if kind in (DUST, DEBRIS):
                p[3] += GRAVITY * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            g = self.ground(p[0])
            if p[1] < g:
                p[1] = g
                if kind in (FIRE, RCS, BLAST):  # exhaust hitting the ground fans out sideways
                    p[2] = math.copysign(max(abs(p[2]), abs(p[3]) * 0.8), p[2] or self.rng.uniform(-1, 1))
                    p[3] = abs(p[3]) * 0.08
                else:
                    p[2] *= 0.5
                    p[3] = -p[3] * 0.25
            alive.append(p)
        self.particles = alive

    def on_outcome(self, env, label: str) -> None:
        u = env.unwrapped
        if label == "crashed":
            self.dead = True
            self._explode(u.lander.position.x, u.lander.position.y, u.lander.linearVelocity)
        elif label == "landed on pad":
            self.pad_ok = True

    def banner(self, label: str) -> tuple[str, str, str]:
        return {
            "landed on pad": ("TOUCHDOWN", "on the pad", "good"),
            "landed off pad": ("TOUCHDOWN", "outside the pad", "warn"),
            "crashed": ("CRASH", "hull contact with the surface", "bad"),
            "flew off-screen": ("LOST", "flew out of the landing zone", "bad"),
            "timeout (hovering)": ("TIMEOUT", "still airborne after 1000 steps", "warn"),
        }.get(label, (label.upper(), "", "normal"))

    def outcome_tone(self, label: str) -> str:
        return self.banner(label)[2]

    def outcome_sound(self, label: str) -> str | None:
        return {"landed on pad": "success", "landed off pad": "touchdown", "crashed": "explosion",
                "flew off-screen": "fail", "timeout (hovering)": "fail"}.get(label)

    def sound_levels(self, env) -> tuple[dict[str, float], float]:
        """Loop volumes for lander.sound (engine follows throttle, rcs the side jets),
        and the stereo pan from the lander's horizontal position."""
        a0, a1 = self.action
        pan = (env.unwrapped.lander.position.x / WORLD_W) * 2 - 1
        if self.dead:
            return {}, pan
        main = 0.0 if a0 <= 0 else (min(a0, 1.0) + 1) / 2
        side = 0.0 if abs(a1) <= 0.5 else min(abs(a1), 1.0)
        return {"engine": main, "rcs": 0.6 * side}, pan

    def gauges(self, env) -> list[Gauge]:
        u = env.unwrapped
        lander = u.lander
        x, y = lander.position
        vx, vy = lander.linearVelocity
        alt = max(0.0, y - self.ground(x) - 18 * U)  # 18 U: body centre height when standing on its legs
        tilt = math.degrees(lander.angle)
        a0, a1 = self.action
        main = 0.0 if a0 <= 0 else (min(a0, 1.0) + 1) / 2
        side = "off" if abs(a1) <= 0.5 else f"{'LEFT' if a1 > 0 else 'RIGHT'} {min(abs(a1), 1.0) * 100:3.0f}%"
        low = alt < 2.5
        right, left = (leg.ground_contact for leg in u.legs)  # legs[0] is the right leg
        legs = {(True, True): "both down", (True, False): "left down", (False, True): "right down"}.get((left, right), "up")
        return [
            Gauge("altitude", f"{alt:6.2f}"),
            Gauge("vert speed", f"{vy:+6.2f}", tone=("good" if abs(vy) < 0.6 else "warn" if abs(vy) < 1.5 else "bad") if low else "normal"),
            Gauge("horiz speed", f"{vx:+6.2f}", tone=("good" if abs(vx) < 0.4 else "warn") if low else "normal"),
            Gauge("tilt", f"{tilt:+6.1f}°", tone=("good" if abs(tilt) < 8 else "warn" if abs(tilt) < 20 else "bad") if low else "normal"),
            Gauge("main engine", f"{main * 100:4.0f}%", frac=main, tone="warn" if main else "dim"),
            Gauge("side rcs", side, tone="accent" if abs(a1) > 0.5 else "dim"),
            Gauge("legs", legs, tone="good" if left or right else "dim"),
        ]

    def draw(self, canvas: pygame.Surface, env) -> None:
        canvas.blit(self.background, (0, 0))
        self._draw_twinkle(canvas)
        u = env.unwrapped
        self._draw_beacons(canvas)
        self._draw_shadow(canvas, u.lander)
        self._draw_engine_light(canvas, u.lander)
        self._draw_particles(canvas, behind=True)
        self._draw_legs(canvas, u)
        self._draw_body(canvas, u.lander)
        self._draw_flames(canvas, u.lander)
        self._draw_particles(canvas, behind=False)
        if self.flash > 0:
            x, y = to_screen(*self.flash_pos)
            k = self.flash
            k = round(k * 12) / 12  # quantised so the glow cache stays small
            _blit_add(canvas, glow(260, (int(255 * k), int(160 * k), int(80 * k))), (x, y))

    # --- helpers --------------------------------------------------------------

    def ground(self, x: float) -> float:
        xs, ys = self.terrain_x, self.terrain_y
        if not xs:
            return 0.0
        dx = xs[1] - xs[0]
        i = min(max(int((x - xs[0]) / dx), 0), len(xs) - 2)
        t = (x - xs[i]) / dx
        return ys[i] + (ys[i + 1] - ys[i]) * t

    @staticmethod
    def _local(body, lx: float, ly: float) -> tuple[float, float]:
        """Body-local point in U -> screen px."""
        c, s = math.cos(body.angle), math.sin(body.angle)
        x, y = body.position
        return to_screen(x + (c * lx - s * ly) * U, y + (s * lx + c * ly) * U)

    # --- effects ----------------------------------------------------------------

    def _spawn_thrust(self, u) -> None:
        rng, lander = self.rng, u.lander
        a0, a1 = self.action
        c, s = math.cos(lander.angle), math.sin(lander.angle)
        down, right = (s, -c), (c, s)
        vx, vy = lander.linearVelocity
        main = 0.0 if a0 <= 0 else (min(a0, 1.0) + 1) / 2
        if main > 0:
            nx = lander.position.x + s * 13 * U
            ny = lander.position.y - c * 13 * U
            for _ in range(int(3 + 6 * main)):
                spd = rng.uniform(5, 9) * main
                spread = rng.gauss(0, 0.8)
                off = rng.uniform(-3, 3) * U
                self.particles.append([
                    nx + right[0] * off, ny + right[1] * off,
                    vx + down[0] * spd + right[0] * spread, vy + down[1] * spd + right[1] * spread,
                    0.0, rng.uniform(0.18, 0.42), FIRE, rng.choice((2, 3, 3, 4)),
                ])
            # Dust kicked up where the plume meets the ground.
            h = ny - self.ground(nx)
            if h < 4.0:
                k = (1 - h / 4.0) * main
                gx = nx + down[0] * max(h, 0)
                gy = self.ground(gx)
                for _ in range(int(12 * k + rng.random())):
                    side = rng.choice((-1, 1))
                    self.particles.append([
                        gx + rng.uniform(-0.25, 0.25), gy + 0.02,
                        side * rng.uniform(2, 7) * (0.5 + k), rng.uniform(0.3, 1.8) * (0.4 + k),
                        0.0, rng.uniform(0.7, 1.6), DUST, rng.choice((3, 4, 5, 6)),
                    ])
        if abs(a1) > 0.5:
            d = 1 if a1 > 0 else -1  # a1 > 0 fires the left-side thruster, pushing the lander right
            px, py = self._local_world(lander, -d * 16, 11)
            ex, ey = -d * c, -d * s  # exhaust direction: outward from that side
            for _ in range(2):
                spd = rng.uniform(3, 5)
                self.particles.append([
                    px, py, vx + ex * spd + rng.gauss(0, 0.4), vy + ey * spd + rng.gauss(0, 0.4),
                    0.0, rng.uniform(0.1, 0.22), RCS, 2,
                ])

    @staticmethod
    def _local_world(body, lx, ly) -> tuple[float, float]:
        c, s = math.cos(body.angle), math.sin(body.angle)
        return body.position.x + (c * lx - s * ly) * U, body.position.y + (s * lx + c * ly) * U

    def _explode(self, x: float, y: float, vel) -> None:
        rng = self.rng
        self.flash, self.flash_pos = 1.0, (x, y)
        for _ in range(160):  # fireball
            a, spd = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 6.0) * rng.random() ** 0.5
            self.particles.append([x, y, math.cos(a) * spd + vel.x * 0.3, abs(math.sin(a)) * spd + 0.8,
                                   0.0, rng.uniform(0.5, 1.6), BLAST, rng.choice((4, 6, 8, 10))])
        for _ in range(40):  # long-lived embers
            a, spd = rng.uniform(0.2, math.pi - 0.2), rng.uniform(2.0, 7.0)
            self.particles.append([x, y, math.cos(a) * spd, math.sin(a) * spd, 0.0, rng.uniform(1.2, 2.2), FIRE, 2])
        for _ in range(26):  # smoke column, rising slowly and spreading
            self.particles.append([x + rng.gauss(0, 0.25), y + rng.uniform(0, 0.4), rng.gauss(0, 0.25) + vel.x * 0.05,
                                   rng.uniform(0.25, 0.9), -rng.uniform(0, 0.6), rng.uniform(1.8, 2.8), SMOKE, rng.choice((7, 9, 11))])
        for _ in range(35):
            a, spd = rng.uniform(0.15, math.pi - 0.15), rng.uniform(2.0, 6.5)
            self.particles.append([x, y, math.cos(a) * spd, math.sin(a) * spd,
                                   0.0, rng.uniform(1.4, 2.6), DEBRIS, rng.choice((2, 2, 3))])
        for _ in range(50):
            side = rng.choice((-1, 1))
            self.particles.append([x + rng.uniform(-0.4, 0.4), self.ground(x) + 0.05, side * rng.uniform(1, 6),
                                   rng.uniform(0.5, 3.0), 0.0, rng.uniform(1.0, 2.2), DUST, rng.choice((4, 5, 6, 7))])

    # --- per-frame drawing ------------------------------------------------------

    def _draw_twinkle(self, canvas) -> None:
        for x, y, b, ph, fr in self.twinkle:
            k = 0.55 + 0.45 * math.sin(self.t * fr + ph)
            v = int(b * k)
            _blit_add(canvas, glow(3, (v, v, min(255, v + 20))), (x, y))

    def _draw_beacons(self, canvas) -> None:
        x1, x2, py = self.pad
        for i, x in enumerate((x1, x2)):
            sx, sy = to_screen(x, py)
            top = (sx, sy - 0.95 * PPU)
            if self.pad_ok:
                k = 0.8 + 0.2 * math.sin(self.t * 6)
                color = (int(60 * k), int(255 * k), int(130 * k))
            else:
                phase = (self.t + 0.5 * i) % 1.0
                k = 1.0 if phase < 0.12 else 0.12
                color = (int(255 * k), int(70 * k), int(50 * k))
            _blit_add(canvas, glow(18, color), top)
            pygame.gfxdraw.filled_circle(canvas, int(top[0]), int(top[1]), 2, _lerp((40, 40, 40), (255, 255, 255), k))

    def _draw_shadow(self, canvas, lander) -> None:
        x, y = lander.position
        alt = y - self.ground(x)
        if alt > 8:
            return
        k = 1 - alt / 8
        sx = x + alt * 0.35  # sun from upper left casts the shadow to the right
        w = int((44 + alt * 5) // 4 * 4)
        _blit(canvas, soft(w, (0, 0, 0), int(150 * k) // 10 * 10, squash=0.13), to_screen(sx, self.ground(sx)))

    def _draw_engine_light(self, canvas, lander) -> None:
        a0 = self.action[0]
        if self.dead or a0 <= 0:
            return
        main = (min(a0, 1.0) + 1) / 2
        nx, ny = self._local_world(lander, 0, -13)
        h = ny - self.ground(nx)
        if h < 5:
            k = round((1 - h / 5) * main * 10) / 10
            if k > 0:
                c = (int(150 * k), int(80 * k), int(30 * k))
                _blit_add(canvas, glow(120, c, squash=0.2), to_screen(nx, self.ground(nx)))

    def _draw_legs(self, canvas, u) -> None:
        lander = u.lander
        c, s = math.cos(lander.angle), math.sin(lander.angle)
        for leg in u.legs:
            side = 1 if lander.GetLocalPoint(leg.position).x > 0 else -1
            top = to_screen(*leg.GetWorldPoint((0, LEG_H * U)))
            mid = to_screen(*leg.GetWorldPoint((0, 0)))
            foot_w = leg.GetWorldPoint((0, -LEG_H * U))
            foot = to_screen(*foot_w)
            strut, strut2, tube, pad = (
                ((70, 62, 56), (58, 52, 48), (92, 84, 78), (100, 92, 86)) if self.dead
                else ((120, 112, 96), (96, 92, 84), (178, 176, 180), (200, 200, 205))
            )
            _aaline(canvas, strut, self._local(lander, side * 15, -1), top, 3)
            _aaline(canvas, strut2, self._local(lander, side * 11, -9), mid, 2)
            _aaline(canvas, tube, top, foot, 4)
            # footpad: a flat disc kept level with the body
            p1 = to_screen(foot_w[0] - c * 4.5 * U, foot_w[1] - s * 4.5 * U)
            p2 = to_screen(foot_w[0] + c * 4.5 * U, foot_w[1] + s * 4.5 * U)
            _aaline(canvas, pad, p1, p2, 3)
            if leg.ground_contact:
                _blit_add(canvas, glow(6, (40, 120, 70)), foot)

    def _draw_body(self, canvas, lander) -> None:
        img = pygame.transform.rotozoom(self.sprite_dead if self.dead else self.sprite, math.degrees(lander.angle), 1.0)
        canvas.blit(img, img.get_rect(center=_ipts([to_screen(*lander.position)])[0]))

    def _flame_poly(self, lander, base_y, length, half_w, profile, scale_w=1.0, scale_l=1.0):
        pts_l, pts_r = [], []
        for t, w in profile:
            ly = base_y - t * length * scale_l
            pts_l.append(self._local(lander, -w * half_w * scale_w, ly))
            pts_r.append(self._local(lander, w * half_w * scale_w, ly))
        return pts_l + pts_r[::-1]

    def _add_poly(self, canvas, color, pts) -> None:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = int(min(xs)) - 2, int(min(ys)) - 2
        tmp = pygame.Surface((int(max(xs)) - x0 + 4, int(max(ys)) - y0 + 4))
        _aapoly(tmp, color, [(x - x0, y - y0) for x, y in pts])
        canvas.blit(tmp, (x0, y0), special_flags=pygame.BLEND_ADD)

    def _draw_flames(self, canvas, lander) -> None:
        if self.dead:
            return
        rng = self.rng
        a0, a1 = self.action
        if a0 > 0:
            main = (min(a0, 1.0) + 1) / 2
            length = (16 + 30 * main) * rng.uniform(0.85, 1.15)
            # Don't draw the plume through the ground: clip it where the thrust axis meets the surface.
            nx, ny = self._local_world(lander, 0, -13)
            to_ground = (ny - self.ground(nx)) / max(math.cos(lander.angle), 0.25) / U
            length = max(0.0, min(length, to_ground))
            profile = [(0, 0.8), (0.12, 1.0), (0.4, 0.75), (0.7, 0.4), (1.0, 0.0)]
            hw = 5.5 * (0.8 + 0.4 * main)
            self._add_poly(canvas, (170, 62, 18), self._flame_poly(lander, -13, length, hw, profile))
            self._add_poly(canvas, (200, 120, 48), self._flame_poly(lander, -13, length, hw, profile, 0.62, 0.72))
            self._add_poly(canvas, (190, 180, 150), self._flame_poly(lander, -13, length, hw, profile, 0.32, 0.42))
            _blit_add(canvas, glow(30, (110, 64, 26)), self._local(lander, 0, -15))
        if abs(a1) > 0.5:
            d = 1 if a1 > 0 else -1
            p = min(abs(a1), 1.0)
            ln = 9 * p * rng.uniform(0.8, 1.2)
            cone = [self._local(lander, -d * 16, 11 + 1.6), self._local(lander, -d * (16 + ln), 11 + 0.5),
                    self._local(lander, -d * (16 + ln), 11 - 0.5), self._local(lander, -d * 16, 11 - 1.6)]
            self._add_poly(canvas, (150, 175, 220), cone)
            _blit_add(canvas, glow(10, (70, 90, 130)), self._local(lander, -d * 17, 11))

    def _draw_particles(self, canvas, behind: bool) -> None:
        for x, y, _vx, _vy, age, life, kind, r in self.particles:
            if (kind in (DUST, DEBRIS)) != behind:
                continue
            t = age / life
            pos = to_screen(x, y)
            if kind == FIRE or kind == BLAST:
                _blit_add(canvas, glow(r, FIRE_RAMP[min(int(t * 12), 11)]), pos)
            elif kind == RCS:
                _blit_add(canvas, glow(r + 1, RCS_RAMP[min(int(t * 8), 7)]), pos)
            elif kind == DUST:
                a = int(110 * (1 - t) ** 1.3) // 10 * 10
                if a > 0:
                    _blit(canvas, soft(int(r * (1 + t)), (176, 170, 160), a), pos)
            elif kind == SMOKE:
                if age < 0:
                    continue
                a = int(150 * min(1.0, t * 6) * (1 - t) ** 1.5) // 10 * 10
                if a > 0:
                    _blit(canvas, soft(int(r * (1 + 1.8 * t)), (92, 88, 86), a), pos)
            else:  # DEBRIS
                shade = int(90 - 40 * t)
                pygame.draw.rect(canvas, (shade, shade - 6, shade - 10), (int(pos[0]), int(pos[1]), r, r))

    # --- static art (built once / once per episode) -----------------------------

    def _make_sky(self) -> pygame.Surface:
        rng = random.Random(42)
        w, h = CANVAS_W, CANVAS_H
        # vertical gradient + a faint diagonal Milky Way haze
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        t = (yy / h)[..., None]
        top, bottom = np.array([3, 4, 12.0]), np.array([18, 20, 38.0])
        img = top * (1 - t) + bottom * t
        band_d = (yy - (260 - xx * 0.18)) / 75.0
        haze = np.exp(-band_d**2)[..., None] * np.array([16, 16, 26.0])
        img = np.clip(img + haze, 0, 255).astype(np.uint8)
        sky = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))

        ridge_far = self._ridge(rng, 350, [(30, 150), (17, 60), (8, 23)])
        ridge_near = self._ridge(rng, 430, [(26, 115), (13, 43), (6, 17)])

        # stars: a uniform field plus a denser band along the haze; only above the far ridge
        def star(x, y, b):
            if y >= ridge_far[int(x)] - 4 or y >= 395:
                return
            tint = rng.choice([(200, 215, 255), (255, 244, 225), (255, 255, 255), (220, 230, 255)])
            sky.set_at((int(x), int(y)), _lerp((0, 0, 0), tint, b / 255))
            if b > 200:
                _blit_add(sky, glow(3, tuple(int(c * 0.35) for c in tint)), (x, y))

        for _ in range(330):
            star(rng.uniform(0, w - 1), rng.uniform(0, 400), 40 + 215 * rng.random() ** 3)
        for _ in range(650):
            x = rng.uniform(0, w - 1)
            star(x, 260 - x * 0.18 + rng.gauss(0, 45), 25 + 90 * rng.random() ** 2)
        self.twinkle = []
        while len(self.twinkle) < 22:
            x, y = rng.uniform(20, w - 20), rng.uniform(10, 300)
            if y < ridge_far[int(x)] - 10 and math.hypot(x - 800, y - 112) > 50:
                self.twinkle.append((x, y, rng.uniform(120, 230), rng.uniform(0, 6.3), rng.uniform(1.5, 4.0)))

        earth = self._make_earth(30)
        sky.blit(earth, earth.get_rect(center=(800, 112)))

        self._fill_ridge(sky, ridge_far, (26, 27, 39), (52, 54, 70))
        self._fill_ridge(sky, ridge_near, (35, 36, 48), (68, 69, 84))
        return sky

    @staticmethod
    def _ridge(rng, base: float, waves) -> np.ndarray:
        x = np.arange(CANVAS_W + 1, dtype=float)
        y = np.full_like(x, base)
        for amp, period in waves:
            y += amp * np.sin(x / period + rng.uniform(0, 2 * math.pi))
        return y

    @staticmethod
    def _fill_ridge(surf, ridge, fill, crest) -> None:
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

    @staticmethod
    def _make_earth(r: int) -> pygame.Surface:
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

    def _make_background(self) -> pygame.Surface:
        rng = random.Random(hash(tuple(round(y, 4) for y in self.terrain_y)))
        bg = self.sky.copy()
        w, h = CANVAS_W, CANVAS_H
        surf_y = np.interp(np.arange(w) / PPU, self.terrain_x, self.terrain_y)
        surf_sy = h - surf_y * PPU  # screen y of the surface for each column

        # regolith: depth gradient + blotchy low-frequency noise + fine grain
        yy = np.arange(h, dtype=float)[None, :]
        depth = np.clip((yy - surf_sy[:, None]) / 280.0, 0, 1)
        top, deep = np.array([124, 121, 116.0]), np.array([50, 49, 52.0])
        col = top * (1 - depth[..., None]) + deep * depth[..., None]
        small = pygame.surfarray.make_surface(np.random.default_rng(rng.randrange(1 << 30)).integers(0, 255, (w // 30, h // 30, 3), dtype=np.uint8))
        blotch = pygame.surfarray.array3d(pygame.transform.smoothscale(small, (w, h)))[..., :1].astype(float)
        col += (blotch - 128) / 128 * 10
        col += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 4, (w, h, 1))
        # side-on craters: shallow dark lenses with a lit lower lip, just under the surface
        for _ in range(18):
            cx = rng.uniform(0, w)
            a, b = rng.uniform(14, 48), rng.uniform(3, 7)
            cy = surf_sy[int(cx)] + rng.uniform(b + 10, 110)
            x0, x1 = int(max(cx - a - 2, 0)), int(min(cx + a + 2, w))
            y0, y1 = int(max(cy - b - 3, 0)), int(min(cy + b + 4, h))
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.mgrid[x0:x1, y0:y1].astype(float)
            q = ((gx - cx) / a) ** 2 + ((gy - cy) / b) ** 2
            inner = np.clip((1 - q) * 4, 0, 1)  # soft edge
            shade = 0.74 + 0.16 * np.clip((gy - cy) / b, -1, 1)  # upper wall in shadow, lower brighter
            lip = np.clip(1 - abs(q - 1.25) * 4, 0, 1) * (gy > cy)
            col[x0:x1, y0:y1] *= (1 + inner * (shade - 1) + lip * 0.1)[..., None]
        alpha = np.clip(yy - surf_sy[:, None] + 0.5, 0, 1) * 255
        terrain = pygame.Surface((w, h), pygame.SRCALPHA)
        pix = pygame.surfarray.pixels3d(terrain)
        pix[:] = np.clip(col, 0, 255).astype(np.uint8)
        del pix
        a = pygame.surfarray.pixels_alpha(terrain)
        a[:] = alpha.astype(np.uint8)
        del a
        bg.blit(terrain, (0, 0))

        # sunlit rim along the surface, brightness by slope facing the sun
        pts = [to_screen(x, y) for x, y in zip(self.terrain_x, self.terrain_y)]
        for (xa, ya), (xb, yb) in itertools.pairwise(pts):
            dx, dy = xb - xa, -(yb - ya)
            ln = math.hypot(dx, dy)
            nx, ny = -dy / ln, dx / ln
            lit = max(0.0, nx * SUN[0] + ny * SUN[1])
            band = _lerp((70, 68, 66), (196, 191, 180), lit)
            _aapoly(bg, band, [(xa, ya), (xb, yb), (xb, yb + 6), (xa, ya + 6)])
            _aaline(bg, _lerp(band, (235, 232, 222), 0.5), (xa, ya), (xb, yb), 2)

        self._draw_rocks(bg, rng)
        self._draw_pad(bg)
        return bg

    def _draw_rocks(self, bg, rng) -> None:
        x1, x2, _ = self.pad
        placed = 0
        while placed < 16:
            x = rng.uniform(0.3, WORLD_W - 0.3)
            if x1 - 0.5 < x < x2 + 0.5:
                continue
            placed += 1
            r = rng.uniform(3, 10)
            sx, sy = to_screen(x, self.ground(x))
            sy += r * 0.25
            _blit(bg, soft(int(r * 1.8), (0, 0, 0), 90, squash=0.25), (sx + r * 0.9, sy))
            pts = []
            for i in range(9):
                ang = math.pi * i / 8
                rr = r * rng.uniform(0.75, 1.1)
                pts.append((sx - math.cos(ang) * rr * 1.25, sy - math.sin(ang) * rr * 0.85))
            _aapoly(bg, (78, 75, 72), pts)
            lit = [(px - (px - sx) * 0.25 - r * 0.15, py - (py - sy) * 0.2 - r * 0.1) for px, py in pts[:6]]
            lit.append((sx - r * 0.1, sy - r * 0.15))
            _aapoly(bg, (160, 156, 148), lit)

    def _draw_pad(self, bg) -> None:
        x1, x2, py = self.pad
        (sx1, sy), (sx2, _) = to_screen(x1, py), to_screen(x2, py)
        slab = pygame.Rect(int(sx1) - 6, int(sy) - 1, int(sx2 - sx1) + 12, 8)
        pygame.draw.rect(bg, (78, 82, 94), slab)
        pygame.draw.line(bg, (186, 192, 204), slab.topleft, (slab.right - 1, slab.top), 2)
        pygame.draw.line(bg, (40, 42, 50), (slab.left, slab.bottom), (slab.right - 1, slab.bottom), 2)
        # hazard stripes at both ends
        for end in (pygame.Rect(slab.x, slab.y + 2, 24, 6), pygame.Rect(slab.right - 24, slab.y + 2, 24, 6)):
            pygame.draw.rect(bg, (24, 24, 26), end)
            bg.set_clip(end)
            for k in range(-2, 8):
                x = end.x + k * 8
                pygame.draw.polygon(bg, (232, 190, 40), [(x, end.bottom), (x + 4, end.bottom), (x + 10, end.top), (x + 6, end.top)])
            bg.set_clip(None)
        # small marker lights along the pad edge
        for i in range(1, 6):
            lx = sx1 + (sx2 - sx1) * i / 6
            _blit_add(bg, glow(4, (110, 80, 20)), (lx, sy + 4))
            bg.set_at((int(lx), int(sy + 4)), (255, 210, 120))
        # beacon masts (the lamps themselves are drawn per frame)
        for sx in (sx1, sx2):
            top = sy - 0.95 * PPU
            _aaline(bg, (120, 124, 136), (sx, sy), (sx, top + 4), 3)
            pygame.draw.rect(bg, (90, 94, 104), (int(sx) - 4, int(sy) - 6, 9, 6))
            pygame.draw.rect(bg, (60, 62, 70), (int(sx) - 3, int(top) - 2, 7, 6), border_radius=2)

    def _make_lander_sprite(self) -> pygame.Surface:
        """Draw the body (descent + ascent stage, engine bell) once at 4x, downsample
        for anti-aliasing. Rotated per frame; legs are drawn separately because
        they move on their joints."""
        ss = 4
        k = PX_U * ss  # px per U at supersampled resolution
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
            shade = _lerp((168, 122, 38), (246, 208, 112), rng.random())
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
