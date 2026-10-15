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

from lander.envs.scene_art import FIRE_RAMP, RCS_RAMP, draw_twinkle, glow, make_lander_sprite, make_sky, soft
from lander.envs.scene_art import aaline as _aaline
from lander.envs.scene_art import aapoly as _aapoly
from lander.envs.scene_art import blit as _blit
from lander.envs.scene_art import blit_add as _blit_add
from lander.envs.scene_art import ipts as _ipts
from lander.envs.scene_art import lerp as _lerp
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




class LunarScene:
    size = (CANVAS_W, CANVAS_H)
    render_mode = None  # we draw from physics state; the env needn't render

    def __init__(self) -> None:
        self.rng = random.Random(7)
        self.sky, self.twinkle = make_sky(CANVAS_W, CANVAS_H)
        self.sprite = make_lander_sprite(PX_U)
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

    def update(self, env, action, dt: float, frac: float = 1.0) -> None:
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
        draw_twinkle(canvas, self.twinkle, self.t)

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
