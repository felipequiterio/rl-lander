"""Viewer scene for descent3d: a side view that follows and zooms, plus a top-down inset.

Side view: the vertical plane through the target along the approach direction (the
horizontal direction from the target to the start point, fixed per episode). The
horizontal axis is the along-track distance; motion across the track only shows in
the inset. The camera keeps the lander and the target in frame and zooms in as the
lander descends, from ~2 km at the start to tens of metres at touchdown. The lander
is drawn at a fixed size (not to scale), tilted to its thrust direction.

Top-down inset: the landing zone, the horizontal track, and the exact 3D glide-slope
limit at the current altitude (a ring the lander must stay inside). The side view's
cone is exact only without cross-track offset; the inset's ring always is.

The env decides at 5 Hz; the scene interpolates ``substeps`` frames between
decisions, so the viewer plays back smoothly at 50 fps in real time.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame
import pygame.gfxdraw

from lander.envs.scene_art import (
    FIRE_RAMP,
    aaline,
    aapoly,
    blit,
    blit_add,
    draw_twinkle,
    glow,
    lerp,
    make_lander_sprite,
    make_sky,
    soft,
)
from lander.hud import TONES, Gauge

CANVAS_W, CANVAS_H = 1200, 800
GROUND_Y = 690  # screen y of the ground (z = 0)
TOP_Y = 170  # highest screen y the lander is kept below (room for the HUD)
VIEW_X = (70, 880)  # screen x range the side view keeps the lander and target in (the inset sits right of it)
PX_U = 2.0  # lander sprite scale: px per LunarLander body unit
FOOT_PX = 13.5 * PX_U  # body centre height above the footpads
INSET = pygame.Rect(CANVAS_W - 16 - 260, 318, 260, 260)

FIRE, DUST, BLAST, DEBRIS = range(4)


BANNERS = {  # outcome -> (title, subtitle, tone)
    "landed on target": ("TOUCHDOWN", "on target", "good"),
    "landed off target": ("OFF TARGET", "soft landing outside the 10 m zone", "warn"),
    "crashed": ("CRASH", "touchdown too fast", "bad"),
    "tipped over": ("TIPPED OVER", "soft touchdown on ground steeper than 12\u00b0", "bad"),
    "glide-slope violation": ("GLIDE SLOPE", "dropped below the approach cone", "bad"),
    "out of propellant": ("OUT OF PROPELLANT", "tanks dry while airborne", "bad"),
    "timeout": ("TIMEOUT", "still airborne at the time limit", "warn"),
}


def _nice_step(target: float) -> float:
    """1, 2 or 5 x 10^k closest above ``target``."""
    k = 10 ** math.floor(math.log10(max(target, 1e-9)))
    for m in (1, 2, 5, 10):
        if m * k >= target:
            return m * k
    return 10 * k


class Descent3DScene:
    size = (CANVAS_W, CANVAS_H)
    render_mode = None
    substeps = 10  # frames drawn per env step (5 Hz control -> 50 fps)

    def __init__(self) -> None:
        self.sky, self.twinkle = make_sky(CANVAS_W, CANVAS_H)
        self.ground_band = self._make_ground_band()
        self.sprite = make_lander_sprite(PX_U)
        self.sprite_dead = self.sprite.copy()
        self.sprite_dead.fill((70, 60, 55, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.font = pygame.font.SysFont("ubuntumono,consolas,dejavusansmono,menlo", 13)
        self.rng = random.Random(5)
        self.t = 0.0

    # --- protocol used by the viewer ------------------------------------------

    def reset(self, env) -> None:
        u = env.unwrapped
        self.cfg = u.cfg
        rh = u.r[:2]
        d = float(np.hypot(*rh))
        self.axis = rh / d if d > 1.0 else np.array([1.0, 0.0])  # along-track unit vector
        self.prev = self.cur = self._snapshot(u)
        self.frac = 1.0
        self.trail: list[np.ndarray] = [u.r.copy()]
        self.particles: list[list] = []
        self.dead = self.pad_ok = False
        self.flash = 0.0
        # camera: snap to the start, then follow smoothly
        self.cam_x, self.scale = self._camera_target(self._state())
        self.inset_half = self._inset_target(self._state())

    def update(self, env, action, dt: float, frac: float = 1.0) -> None:
        u = env.unwrapped
        self.t += dt
        if u.t != self.cur["t"]:  # a new env step happened: interpolate from the previous one
            self.prev, self.cur = self.cur, self._snapshot(u)
            self.trail.append(u.r.copy())
        self.frac = frac
        st = self._state()
        cx, sc = self._camera_target(st)
        k = 1 - math.exp(-dt * 2.5)  # ~0.4 s time constant
        self.cam_x += (cx - self.cam_x) * k
        self.scale *= (sc / self.scale) ** k  # smooth in log space: zooms feel uniform
        self.inset_half *= (self._inset_target(st) / self.inset_half) ** k
        if action is not None and not self.dead:
            self._spawn_thrust(st)
        self._step_particles(dt)
        self.flash = max(0.0, self.flash - dt * 1.4)

    def on_outcome(self, env, label: str) -> None:
        if label.startswith("crashed"):
            self.dead = True
            st = self._state()
            self._explode(self._along(st["r"]), max(st["r"][2], 0.0))
        elif label == "landed on target":
            self.pad_ok = True


    def banner(self, label: str) -> tuple[str, str, str]:
        violated = label.endswith(" after glide-slope violation")
        base = label.removesuffix(" after glide-slope violation")
        title, sub, tone = BANNERS.get(base, (label.upper(), "", "normal"))
        if violated:
            sub, tone = f"{sub}, after a glide-slope violation".lstrip(", "), "bad"
        return title, sub, tone

    def outcome_tone(self, label: str) -> str:
        return self.banner(label)[2]

    def outcome_sound(self, label: str) -> str | None:
        base = label.removesuffix(" after glide-slope violation")
        return {"landed on target": "success", "landed off target": "touchdown", "crashed": "explosion",
                "tipped over": "touchdown"}.get(base, "fail")

    def sound_levels(self, env) -> tuple[dict[str, float], float]:
        st = self._state()
        pan = self._sx(self._along(st["r"])) / CANVAS_W * 2 - 1
        if self.dead:
            return {}, pan
        return {"engine": 0.35 + 0.65 * self._throttle(st)}, pan

    def gauges(self, env) -> list[Gauge]:
        u, st, c = env.unwrapped, self._state(), self.cfg
        r, v, T = st["r"], st["v"], st["thrust"]
        # on uneven terrain, what a radar altimeter reads: the height above the ground below
        alt, dist = max(r[2] - u.ground_height(r[0], r[1]), 0.0), float(np.hypot(r[0], r[1]))
        vz, vh = v[2], float(np.hypot(v[0], v[1]))
        tmag = float(np.linalg.norm(T))
        tilt = math.degrees(math.acos(T[2] / tmag)) if tmag > 0 else 0.0
        prop = max(st["m"] - c.m_min, 0.0)
        margin = u.glide_angle_margin_deg(r)
        low = alt < 60
        return [
            Gauge("radar alt" if u.site is not None else "altitude", f"{alt:7.1f} m"),
            Gauge("vert speed", f"{vz:+6.1f} m/s",
                  tone=("good" if -vz <= c.max_touchdown_vz else "warn" if -vz <= 2 * c.max_touchdown_vz else "bad") if low else "normal"),
            Gauge("horiz speed", f"{vh:5.1f} m/s", tone=("good" if vh <= c.max_touchdown_vh else "warn") if low else "normal"),
            Gauge("to target", f"{dist:7.1f} m", tone=("good" if dist <= c.landing_radius else "normal")),
            Gauge("thrust", f"{tmag / 1000:5.1f} kN", frac=tmag / c.thrust_max, tone="warn"),
            Gauge("tilt", f"{tilt:5.1f}°"),
            Gauge("propellant", f"{prop:5.0f} kg", frac=prop / c.propellant,
                  tone="bad" if prop < 0.1 * c.propellant else "warn" if prop < 0.25 * c.propellant else "good"),
            Gauge("glide margin", "in zone" if math.isinf(margin) else f"{margin:+5.1f}°",
                  tone="good" if margin > 10 else "warn" if margin >= 0 else "bad"),
        ]

    def draw(self, canvas: pygame.Surface, env) -> None:
        st = self._state()
        canvas.blit(self.sky, (0, 0))
        draw_twinkle(canvas, self.twinkle, self.t)
        self._draw_cone(canvas)
        canvas.blit(self.ground_band, (0, GROUND_Y))
        pygame.draw.line(canvas, (205, 200, 190), (0, GROUND_Y), (CANVAS_W, GROUND_Y), 2)
        self._draw_ruler(canvas)
        self._draw_pad(canvas)
        self._draw_trail(canvas)
        self._draw_shadow(canvas, st)
        self._draw_particles(canvas, behind=True)
        self._draw_lander(canvas, st)
        self._draw_particles(canvas, behind=False)
        if self.flash > 0:
            k = round(self.flash * 12) / 12
            blit_add(canvas, glow(240, (int(255 * k), int(160 * k), int(80 * k))), self.flash_pos)
        self._draw_inset(canvas, st)

    # --- state, interpolation and camera -----------------------------------------------

    @staticmethod
    def _snapshot(u) -> dict:
        return {"t": u.t, "r": u.r.copy(), "v": u.v.copy(), "m": u.m, "thrust": u.thrust.copy()}

    def _state(self) -> dict:
        a, b, f = self.prev, self.cur, self.frac
        return {
            "r": a["r"] + (b["r"] - a["r"]) * f,
            "v": a["v"] + (b["v"] - a["v"]) * f,
            "m": a["m"] + (b["m"] - a["m"]) * f,
            "thrust": b["thrust"],  # thrust is piecewise constant: the current command
        }

    def _along(self, r: np.ndarray) -> float:
        return float(r[:2] @ self.axis)

    def _throttle(self, st: dict) -> float:
        c = self.cfg
        return (float(np.linalg.norm(st["thrust"])) - c.thrust_min) / (c.thrust_max - c.thrust_min)

    def _camera_target(self, st: dict) -> tuple[float, float]:
        s, z = self._along(st["r"]), max(st["r"][2], 0.0)
        lo, hi = min(s, 0.0), max(s, 0.0)
        span_x = (hi - lo) * 1.25 + 30.0
        span_z = z * 1.15 + 25.0
        scale = min((VIEW_X[1] - VIEW_X[0]) / span_x, (GROUND_Y - TOP_Y) / span_z)
        return (lo + hi) / 2, min(scale, 40.0)

    def _inset_target(self, st: dict) -> float:
        return max(1.35 * float(np.hypot(*st["r"][:2])), 25.0)

    def _sx(self, s: float) -> float:
        return (VIEW_X[0] + VIEW_X[1]) / 2 + (s - self.cam_x) * self.scale

    def _sy(self, z: float) -> float:
        return GROUND_Y - z * self.scale

    # --- effects -----------------------------------------------------------------

    def _body(self, cx: float, cy: float, tilt: float, lx: float, ly: float) -> tuple[float, float]:
        """Lander body point (LunarLander U units, y up) -> screen, with the body leaning
        ``tilt`` radians to the right."""
        c, s = math.cos(tilt), math.sin(tilt)
        return cx + (lx * c + ly * s) * PX_U, cy - (-lx * s + ly * c) * PX_U

    def _pose(self, st: dict) -> tuple[float, float, float]:
        T = st["thrust"]
        tilt = math.atan2(float(T[:2] @ self.axis), float(T[2]))
        return self._sx(self._along(st["r"])), self._sy(max(st["r"][2], 0.0)) - FOOT_PX, tilt

    def _spawn_thrust(self, st: dict) -> None:
        rng = self.rng
        cx, cy, tilt = self._pose(st)
        thr = self._throttle(st)
        nx, ny = self._body(cx, cy, tilt, 0, -13)
        down = (-math.sin(tilt), math.cos(tilt))  # screen direction of the exhaust
        for _ in range(int(2 + 4 * thr)):
            spd = rng.uniform(260, 420) * (0.5 + 0.5 * thr)  # px/s
            spread = rng.gauss(0, 30)
            vx_px = down[0] * spd + down[1] * spread
            vy_px = down[1] * spd - down[0] * spread
            self.particles.append([nx, ny, vx_px, vy_px, 0.0, rng.uniform(0.15, 0.35), FIRE, rng.choice((2, 3, 3, 4))])
        h_px = GROUND_Y - ny
        if 0 < h_px < 140:  # plume reaching the ground kicks up dust
            k = (1 - h_px / 140) * (0.4 + 0.6 * thr)
            gx = nx + down[0] * h_px
            for _ in range(int(10 * k + rng.random())):
                side = rng.choice((-1, 1))
                self.particles.append([gx + rng.uniform(-6, 6), GROUND_Y - 1, side * rng.uniform(80, 300) * (0.5 + k),
                                       -rng.uniform(10, 70) * (0.4 + k), 0.0, rng.uniform(0.6, 1.4), DUST, rng.choice((3, 4, 5, 6))])

    def _explode(self, s: float, z: float) -> None:
        rng = self.rng
        x, y = self._sx(s), self._sy(z) - FOOT_PX
        self.flash, self.flash_pos = 1.0, (x, y)
        for _ in range(150):
            a, spd = rng.uniform(math.pi, 2 * math.pi), rng.uniform(30, 330) * rng.random() ** 0.5
            self.particles.append([x, y, math.cos(a) * spd, math.sin(a) * spd - 40, 0.0, rng.uniform(0.5, 1.5), BLAST, rng.choice((4, 6, 8, 10))])
        for _ in range(35):
            a, spd = rng.uniform(math.pi + 0.15, 2 * math.pi - 0.15), rng.uniform(120, 380)
            self.particles.append([x, y, math.cos(a) * spd, math.sin(a) * spd, 0.0, rng.uniform(1.4, 2.6), DEBRIS, rng.choice((2, 2, 3))])
        for _ in range(50):
            side = rng.choice((-1, 1))
            self.particles.append([x + rng.uniform(-10, 10), GROUND_Y - 1, side * rng.uniform(60, 330), -rng.uniform(30, 160),
                                   0.0, rng.uniform(1.0, 2.2), DUST, rng.choice((4, 5, 6, 7))])

    def _step_particles(self, dt: float) -> None:
        # Particles live in screen space (they're decoration; they don't need to zoom).
        alive = []
        for p in self.particles:
            p[4] += dt
            if p[4] >= p[5]:
                continue
            if p[6] in (DUST, DEBRIS):
                p[3] += 360 * dt  # px/s^2 "gravity"
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            if p[1] > GROUND_Y - 1:
                p[1] = GROUND_Y - 1
                if p[6] in (FIRE, BLAST):  # exhaust hitting the ground fans out sideways
                    p[2] = math.copysign(max(abs(p[2]), abs(p[3]) * 0.8), p[2] or self.rng.uniform(-1, 1))
                    p[3] = -abs(p[3]) * 0.08
                else:
                    p[2] *= 0.5
                    p[3] = -p[3] * 0.25
            alive.append(p)
        self.particles = alive

    # --- drawing -------------------------------------------------------------------

    def _draw_cone(self, canvas) -> None:
        """Glide-slope cone in the side view: shaded no-go wedges below lines rising at
        the glide-slope angle from the landing-zone edges."""
        c = self.cfg
        if c.glide_slope_deg is None:
            return
        tan_g = math.tan(math.radians(c.glide_slope_deg))
        far = 2 * CANVAS_W / self.scale  # far enough to leave the screen on either side
        layer = pygame.Surface((CANVAS_W, GROUND_Y), pygame.SRCALPHA)
        for side in (-1, 1):
            edge = side * c.landing_radius
            p0 = (self._sx(edge), self._sy(0))
            p1 = (self._sx(edge + side * far), self._sy(far * tan_g))
            wedge = [p0, p1, (self._sx(edge + side * far), GROUND_Y)]
            pygame.draw.polygon(layer, (255, 90, 70, 26), [(x, min(y, GROUND_Y)) for x, y in wedge])
            pygame.draw.aaline(layer, (255, 130, 110, 150), p0, p1)
        canvas.blit(layer, (0, 0))
        # label along the right-hand line, where it crosses x = 830 px
        s_lab = self.cam_x + (VIEW_X[1] - 50 - (VIEW_X[0] + VIEW_X[1]) / 2) / self.scale
        if s_lab > c.landing_radius:
            y = self._sy((s_lab - c.landing_radius) * tan_g)
            if TOP_Y < y < GROUND_Y - 20:
                img = self.font.render(f"{c.glide_slope_deg:g}° glide slope", True, (255, 150, 130))
                canvas.blit(img, (VIEW_X[1] - 50 - img.get_width(), y - 22))

    def _draw_ruler(self, canvas) -> None:
        step = _nice_step(90 / self.scale)  # ticks at least ~90 px apart
        s0 = self.cam_x - (CANVAS_W / 2 + 200) / self.scale
        s1 = self.cam_x + (CANVAS_W / 2 + 200) / self.scale
        for i in range(math.floor(s0 / step), math.ceil(s1 / step) + 1):
            s = i * step
            x = self._sx(s)
            if not 0 <= x <= CANVAS_W:
                continue
            pygame.draw.line(canvas, (150, 146, 140), (x, GROUND_Y + 2), (x, GROUND_Y + (9 if i else 14)), 1)
            label = "target" if i == 0 else f"{s:+g} m" if step < 1000 else f"{s / 1000:+g} km"
            img = self.font.render(label, True, (165, 160, 152) if i else (230, 225, 210))
            canvas.blit(img, (x - img.get_width() / 2, GROUND_Y + 16))

    def _draw_pad(self, canvas) -> None:
        R = self.cfg.landing_radius
        x0, x1 = self._sx(-R), self._sx(R)
        if x1 - x0 < 8:  # too far out to show the pad: a marker
            xc = (x0 + x1) / 2
            pygame.draw.polygon(canvas, (232, 190, 40), [(xc, GROUND_Y - 2), (xc - 6, GROUND_Y - 12), (xc + 6, GROUND_Y - 12)])
            return
        slab = pygame.Rect(int(x0), GROUND_Y - 2, int(x1 - x0), 7)
        pygame.draw.rect(canvas, (78, 82, 94), slab)
        pygame.draw.line(canvas, (186, 192, 204), slab.topleft, (slab.right - 1, slab.top), 2)
        w = min(24, slab.width // 4)
        for end in (pygame.Rect(slab.x, slab.y + 2, w, 5), pygame.Rect(slab.right - w, slab.y + 2, w, 5)):
            pygame.draw.rect(canvas, (24, 24, 26), end)
            canvas.set_clip(end)
            for k in range(-2, w // 8 + 3):
                xx = end.x + k * 8
                pygame.draw.polygon(canvas, (232, 190, 40), [(xx, end.bottom), (xx + 4, end.bottom), (xx + 9, end.top), (xx + 5, end.top)])
            canvas.set_clip(None)
        # beacons at the zone edges: blink red; steady green after a landing on target
        mast = min(60.0, max(14.0, 4.0 * self.scale))
        for i, x in enumerate((x0, x1)):
            aaline(canvas, (120, 124, 136), (x, GROUND_Y - 2), (x, GROUND_Y - mast), 2)
            if self.pad_ok:
                k = 0.8 + 0.2 * math.sin(self.t * 6)
                color = (int(60 * k), int(255 * k), int(130 * k))
            else:
                k = 1.0 if (self.t + 0.5 * i) % 1.0 < 0.12 else 0.12
                color = (int(255 * k), int(70 * k), int(50 * k))
            blit_add(canvas, glow(16, color), (x, GROUND_Y - mast))
            pygame.gfxdraw.filled_circle(canvas, int(x), int(GROUND_Y - mast), 2, lerp((40, 40, 40), (255, 255, 255), k))

    def _draw_trail(self, canvas) -> None:
        pts = [(self._sx(self._along(r)), self._sy(max(r[2], 0.0)) - FOOT_PX) for r in self.trail]
        st = self._state()
        pts.append((self._sx(self._along(st["r"])), self._sy(max(st["r"][2], 0.0)) - FOOT_PX))
        if len(pts) < 2:
            return
        layer = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
        n = len(pts)
        for i in range(1, n):
            a = int(40 + 150 * i / n)
            pygame.draw.line(layer, (120, 200, 255, a), pts[i - 1], pts[i], 2)
        canvas.blit(layer, (0, 0))

    def _draw_shadow(self, canvas, st: dict) -> None:
        z = st["r"][2]
        h_px = z * self.scale
        if h_px > 400:
            return
        k = 1 - h_px / 400
        x = self._sx(self._along(st["r"])) + h_px * 0.3
        w = int((40 + h_px * 0.15) // 4 * 4)
        blit(canvas, soft(w, (0, 0, 0), int(140 * k) // 10 * 10, squash=0.14), (x, GROUND_Y))

    def _draw_lander(self, canvas, st: dict) -> None:
        cx, cy, tilt = self._pose(st)
        body = lambda lx, ly: self._body(cx, cy, tilt, lx, ly)
        # legs (fixed geometry: the LunarLander legs at rest), behind the body
        strut, tube, pad = ((70, 62, 56), (92, 84, 78), (100, 92, 86)) if self.dead else ((120, 112, 96), (178, 176, 180), (200, 200, 205))
        for side in (-1, 1):
            top, foot = body(side * 22.4, 0.6), body(side * 29.9, -13.5)
            aaline(canvas, strut, body(side * 15, -1), top, 3)
            aaline(canvas, strut, body(side * 11, -9), body(side * 26, -6.5), 2)
            aaline(canvas, tube, top, foot, 4)
            aaline(canvas, pad, body(side * 25.4, -13.5), body(side * 34.4, -13.5), 3)
        img = pygame.transform.rotozoom(self.sprite_dead if self.dead else self.sprite, -math.degrees(tilt), 1.0)
        canvas.blit(img, img.get_rect(center=(int(cx), int(cy))))
        if not self.dead:
            self._draw_flame(canvas, st, cx, cy, tilt)

    def _draw_flame(self, canvas, st, cx, cy, tilt) -> None:
        thr = self._throttle(st)
        length = (12 + 34 * thr) * self.rng.uniform(0.85, 1.15)
        nx, ny = self._body(cx, cy, tilt, 0, -13)
        to_ground = (GROUND_Y - ny) / max(math.cos(tilt), 0.25) / PX_U
        length = max(0.0, min(length, to_ground))
        profile = [(0, 0.8), (0.12, 1.0), (0.4, 0.75), (0.7, 0.4), (1.0, 0.0)]
        hw = 5.5 * (0.8 + 0.4 * thr)
        for color, sw, sl in (((170, 62, 18), 1.0, 1.0), ((200, 120, 48), 0.62, 0.72), ((190, 180, 150), 0.32, 0.42)):
            left = [self._body(cx, cy, tilt, -w * hw * sw, -13 - t * length * sl) for t, w in profile]
            right = [self._body(cx, cy, tilt, w * hw * sw, -13 - t * length * sl) for t, w in profile]
            self._add_poly(canvas, color, left + right[::-1])
        blit_add(canvas, glow(30, (110, 64, 26)), self._body(cx, cy, tilt, 0, -15))
        h_px = GROUND_Y - ny
        if h_px < 160:  # engine glow on the ground
            k = round((1 - h_px / 160) * (0.5 + 0.5 * thr) * 10) / 10
            if k > 0:
                blit_add(canvas, glow(110, (int(150 * k), int(80 * k), int(30 * k)), squash=0.2), (nx, GROUND_Y))

    @staticmethod
    def _add_poly(canvas, color, pts) -> None:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0 = int(min(xs)) - 2, int(min(ys)) - 2
        tmp = pygame.Surface((int(max(xs)) - x0 + 4, int(max(ys)) - y0 + 4))
        aapoly(tmp, color, [(x - x0, y - y0) for x, y in pts])
        canvas.blit(tmp, (x0, y0), special_flags=pygame.BLEND_ADD)

    def _draw_particles(self, canvas, behind: bool) -> None:
        for x, y, _vx, _vy, age, life, kind, r in self.particles:
            if (kind in (DUST, DEBRIS)) != behind:
                continue
            t = age / life
            if kind in (FIRE, BLAST):
                blit_add(canvas, glow(r, FIRE_RAMP[min(int(t * 12), 11)]), (x, y))
            elif kind == DUST:
                a = int(110 * (1 - t) ** 1.3) // 10 * 10
                if a > 0:
                    blit(canvas, soft(int(r * (1 + t)), (176, 170, 160), a), (x, y))
            else:
                shade = int(90 - 40 * t)
                pygame.draw.rect(canvas, (shade, shade - 6, shade - 10), (int(x), int(y), r, r))

    def _draw_inset(self, canvas, st: dict) -> None:
        """Top-down view centred on the target: x right, y up (world axes)."""
        rect, c = INSET, self.cfg
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(box, (8, 10, 18, 185), box.get_rect(), border_radius=10)
        pygame.draw.rect(box, (255, 255, 255, 26), box.get_rect(), width=1, border_radius=10)
        canvas.blit(box, rect)
        k = (rect.width / 2 - 14) / self.inset_half  # px per metre
        ox, oy = rect.centerx, rect.centery + 8

        def P(x, y):
            return ox + x * k, oy - y * k

        canvas.set_clip(rect.inflate(-4, -4))
        # the side view's plane, through the target along the approach axis
        a = self.axis * self.inset_half * 2
        pygame.draw.line(canvas, (60, 70, 92), P(-a[0], -a[1]), P(a[0], a[1]), 1)
        # exact glide-slope limit at the current altitude: the lander must stay inside this ring
        if c.glide_slope_deg is not None:
            ring = c.landing_radius + max(st["r"][2], 0.0) / math.tan(math.radians(c.glide_slope_deg))
            if ring * k < rect.width:
                pygame.draw.circle(canvas, (255, 130, 110), (int(ox), int(oy)), max(int(ring * k), 1), 1)
        zone = max(int(c.landing_radius * k), 2)
        pygame.draw.circle(canvas, (232, 190, 40), (int(ox), int(oy)), zone, 1)
        pygame.draw.line(canvas, (232, 190, 40), (ox - 5, oy), (ox + 5, oy))
        pygame.draw.line(canvas, (232, 190, 40), (ox, oy - 5), (ox, oy + 5))
        pts = [P(r[0], r[1]) for r in self.trail] + [P(st["r"][0], st["r"][1])]
        if len(pts) > 1:
            pygame.draw.aalines(canvas, (120, 200, 255), False, pts)
        lx, ly = P(st["r"][0], st["r"][1])
        v = st["v"][:2]
        speed = float(np.hypot(*v))
        if speed > 0.05:  # velocity arrow: direction, length ~ speed (capped at 40 px)
            n = v / speed
            ln = 8 + min(speed, 30.0) / 30.0 * 32
            tip = (lx + n[0] * ln, ly - n[1] * ln)
            aaline(canvas, (250, 204, 90), (lx, ly), tip, 2)
            back = (tip[0] - n[0] * 7, tip[1] + n[1] * 7)
            side = (-n[1] * 4, -n[0] * 4)
            aapoly(canvas, (250, 204, 90), [tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1])])
        pygame.gfxdraw.filled_circle(canvas, int(lx), int(ly), 4, (255, 255, 255))
        pygame.gfxdraw.aacircle(canvas, int(lx), int(ly), 4, (255, 255, 255))
        canvas.set_clip(None)
        # title and scale bar
        canvas.blit(self.font.render("TOP VIEW", True, TONES["dim"]), (rect.x + 12, rect.y + 8))
        bar_m = _nice_step(self.inset_half / 3)
        bx, by = rect.x + 12, rect.bottom - 14
        pygame.draw.line(canvas, TONES["dim"], (bx, by), (bx + bar_m * k, by), 2)
        canvas.blit(self.font.render(f"{bar_m:g} m", True, TONES["dim"]), (bx + bar_m * k + 6, by - 8))

    # --- static art -------------------------------------------------------------------

    @staticmethod
    def _make_ground_band() -> pygame.Surface:
        h = CANVAS_H - GROUND_Y
        rng = np.random.default_rng(3)
        t = (np.arange(h, dtype=float) / h)[None, :, None]
        top, deep = np.array([124, 121, 116.0]), np.array([58, 56, 58.0])
        col = top * (1 - t) + deep * t + rng.normal(0, 4, (CANVAS_W, h, 1))
        return pygame.surfarray.make_surface(np.clip(col, 0, 255).astype(np.uint8))
