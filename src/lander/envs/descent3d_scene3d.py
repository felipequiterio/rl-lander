"""3D viewer scene for descent3d: the lander, terrain and trajectory in real 3D.

Rendered with OpenGL (see lander.viz3d.renderer) under the same pygame HUD as the
2D scene. Telemetry, outcome banners, sounds, the 5 Hz -> 50 fps interpolation and
the top-down inset are inherited from the 2D scene; drawing, the camera and the
effects are 3D.

The lander model is the Apollo LM from Bluedog Design Bureau, bundled in the repo
(src/lander/assets/apollo_lm, CC BY-NC-SA 4.0) or re-imported live from a KSP
install (``--bdb PATH`` / ``LANDER_BDB``); a simple built-in model is the fallback. Its attitude follows the thrust vector (a point-mass lander points its engine
where it thrusts).

Cameras (press C to cycle): chase (behind the lander along its approach), side,
ground (standing at the landing zone, zooming to keep the lander framed) and orbit.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pygame

from lander.envs.descent3d_scene import Descent3DScene
from lander.envs.scene_art import FIRE_RAMP
from lander.viz3d.glmath import model_matrix, normalize, rotation_z_to
from lander.viz3d.renderer import SUN_DIR, Frame, Renderer

CAMERAS = ("chase", "side", "ground", "orbit")
LUNAR_G = 1.62


def load_model():
    """The Apollo LM: re-imported live from a KSP install if one is configured
    (``--bdb`` / ``LANDER_BDB``), else the copy bundled in the repo (see
    lander.viz3d.import_bdb), else a simple built-in model."""
    from lander.viz3d.ksp_model import BUNDLE_DIR, load_bundle

    bdb = os.environ.get("LANDER_BDB")
    if bdb:
        try:
            from lander.viz3d.ksp_model import assemble_lm

            model = assemble_lm(bdb)
            print(f"3D model: Bluedog Design Bureau Apollo LM, live from {bdb}", flush=True)
            return model
        except Exception as e:  # noqa: BLE001 - any failure falls back below
            print(f"could not load the BDB model from {bdb} ({e!r})", flush=True)
    model = load_bundle()
    if model is not None:
        print(f"3D model: Bluedog Design Bureau Apollo LM (bundled, {BUNDLE_DIR.name})", flush=True)
        return model
    from lander.viz3d.procedural import procedural_lm

    print("3D model: built-in", flush=True)
    return procedural_lm()


class Descent3DScene3D(Descent3DScene):
    def __init__(self) -> None:
        super().__init__()
        self.model = load_model()
        self.renderer = Renderer(self.model, self.size)  # raises if OpenGL is unavailable (callers fall back to 2D)
        print(f"3D renderer: {self.renderer.gpu}", flush=True)
        self.camera = 0
        self.key_bindings = {pygame.K_c: self.next_camera}
        self.label_font = pygame.font.SysFont("ubuntu,segoeui,dejavusans,arial", 14)
        self.rng3 = random.Random(9)

    def next_camera(self) -> None:
        self.camera = (self.camera + 1) % len(CAMERAS)
        self.cam_eye = None  # snap to the new viewpoint

    # --- protocol ---------------------------------------------------------------------

    def reset(self, env) -> None:
        super().reset(env)
        c = env.unwrapped.cfg
        self.renderer.zone_radius = c.landing_radius
        self.renderer.tan_gs = math.tan(math.radians(c.glide_slope_deg)) if c.glide_slope_deg else None
        self.p3: list[list] = []  # particles: [pos(3), vel(3), age, life, kind, size]
        self.trail3 = [env.unwrapped.r.copy()]
        self.cam_eye = None
        self.flash3 = 0.0
        self.wreck = None
        a = -self.axis  # horizontal direction from the start point towards the target
        self.approach = np.array([a[0], a[1], 0.0])

    def update(self, env, action, dt: float, frac: float = 1.0) -> None:
        u = env.unwrapped
        self.t += dt
        if u.t != self.cur["t"]:
            self.prev, self.cur = self.cur, self._snapshot(u)
            self.trail3.append(u.r.copy())
            self.trail.append(u.r.copy())
        self.frac = frac
        st = self._state()
        k = 1 - math.exp(-dt * 2.5)
        self.inset_half *= (self._inset_target(st) / self.inset_half) ** k
        if action is not None and not self.dead:
            self._spawn3(st, dt)
        self._step3(dt)
        self.flash3 = max(0.0, self.flash3 - dt * 1.2)
        self._update_camera(st, dt)

    def on_outcome(self, env, label: str) -> None:
        if label.startswith("crashed"):
            self.dead = True
            st = self._state()
            # The wreck: tipped over and half-buried, falling towards its horizontal motion.
            v = st["v"]
            tip = normalize(np.array([v[0], v[1], 0.0])) if np.hypot(v[0], v[1]) > 0.5 else np.array([1.0, 0.0, 0.0])
            self.wreck = rotation_z_to(np.array([0, 0, 1.0]) + tip * math.tan(math.radians(28))), np.array([0, 0, -1.0])
            self._explode3(st["r"])
        elif label == "landed on target":
            self.pad_ok = True

    def sound_levels(self, env) -> tuple[dict[str, float], float]:
        if self.dead:
            return {}, 0.0
        return {"engine": 0.35 + 0.65 * self._throttle(self._state())}, 0.0  # the camera follows the lander

    def draw(self, canvas: pygame.Surface, env) -> None:
        st = self._state()
        canvas.blit(self.renderer.render(self._frame(st)), (0, 0))
        self._draw_inset(canvas, st)
        label = self.label_font.render(f"camera: {CAMERAS[self.camera]}   [C] to change", True, (190, 196, 210))
        canvas.blit(label, (16, self.size[1] - 30))

    # --- pose, camera ------------------------------------------------------------------

    def _rotation(self, st: dict) -> np.ndarray:
        if self.dead and self.wreck is not None:
            return self.wreck[0]
        T = st["thrust"]
        return rotation_z_to(T if np.linalg.norm(T) > 0 else np.array([0, 0, 1.0]))

    def _update_camera(self, st: dict, dt: float) -> None:
        r = st["r"]
        alt = max(r[2], 0.0)
        centre = r + np.array([0, 0, 3.0])
        mode = CAMERAS[self.camera]
        to_target = -np.array([r[0], r[1], 0.0])
        d = np.linalg.norm(to_target)
        fwd = to_target / d if d > 15 else self.approach  # near the target, keep the last approach direction
        side = np.cross([0, 0, 1.0], fwd)
        fov = 50.0
        if mode == "chase":
            dist, height = 24 + 0.05 * alt, 7 + 0.03 * alt
            eye = centre - fwd * dist + np.array([0, 0, height])
            look = centre + fwd * min(d, 40.0) * 0.25
        elif mode == "side":
            eye = centre + side * (30 + 0.06 * alt) + np.array([0, 0, 4.0])
            look = centre
        elif mode == "ground":
            # Stand on the sun side of the target, so the lander is front-lit as it comes down.
            sun_h = normalize(np.array([SUN_DIR[0], SUN_DIR[1], 0.0]))
            eye = sun_h * 55.0 + np.array([0, 0, 7.0])
            look = centre
            rng = np.linalg.norm(centre - eye)
            fov = float(np.clip(math.degrees(2 * math.atan(14.0 / rng)), 2.0, 55.0))  # zoom lens: keep the LM framed
        else:  # orbit
            ang = self.t * 0.12
            eye = centre + 30 * np.array([math.cos(ang), math.sin(ang), 0]) + np.array([0, 0, 9.0])
            look = centre
        eye[2] = max(eye[2], 1.2)  # never below the surface
        if self.cam_eye is None:
            self.cam_eye, self.cam_look, self.cam_fov = eye, look, fov
        else:
            k = 1 - math.exp(-dt * 3.0)
            self.cam_eye = self.cam_eye + (eye - self.cam_eye) * k
            self.cam_look = self.cam_look + (look - self.cam_look) * min(1.0, k * 2)
            self.cam_fov += (fov - self.cam_fov) * k

    # --- effects (world space, metres) ------------------------------------------------------

    def _nozzle(self, st: dict) -> tuple[np.ndarray, np.ndarray]:
        R = self._rotation(st)
        return st["r"] + R @ self.model.engine_exit, -R[:, 2]  # exit point, exhaust direction

    def _spawn3(self, st: dict, dt: float) -> None:
        rng = self.rng3
        thr = self._throttle(st)
        exit_pt, down = self._nozzle(st)
        v = st["v"]
        for _ in range(int(2 + 4 * thr)):
            spd = rng.uniform(50, 90) * (0.6 + 0.4 * thr)
            jitter = np.array([rng.gauss(0, 0.12), rng.gauss(0, 0.12), rng.gauss(0, 0.12)])
            self.p3.append([*(exit_pt + jitter * 0.5), *(v + (down + jitter) * spd), 0.0, rng.uniform(0.08, 0.2), 0, rng.uniform(0.1, 0.22)])
        # Dust where the exhaust meets the ground: in vacuum it flies off in long, flat ballistic arcs.
        h = exit_pt[2]
        if down[2] < -0.3 and h < 30:
            hit = exit_pt + down * (h / -down[2])
            k = (1 - h / 30) * (0.3 + 0.7 * thr)
            for _ in range(int(12 * k + rng.random())):
                a = rng.uniform(0, 2 * math.pi)
                spd = rng.uniform(8, 30) * (0.4 + k)
                self.p3.append([hit[0] + rng.gauss(0, 1), hit[1] + rng.gauss(0, 1), 0.05,
                                math.cos(a) * spd, math.sin(a) * spd, rng.uniform(0.3, 2.5) * k, 0.0, rng.uniform(1.0, 2.5), 1, rng.uniform(0.6, 1.6)])

    def _explode3(self, r: np.ndarray) -> None:
        rng = self.rng3
        c = r + np.array([0, 0, 2.5])
        self.flash3, self.flash_pos3 = 1.0, c
        for _ in range(220):
            d = normalize(np.array([rng.gauss(0, 1), rng.gauss(0, 1), abs(rng.gauss(0, 1))]))
            self.p3.append([*c, *(d * rng.uniform(3, 22) * rng.random() ** 0.5), 0.0, rng.uniform(0.5, 1.8), 2, rng.uniform(0.8, 2.6)])
        for _ in range(25):  # debris
            d = normalize(np.array([rng.gauss(0, 1), rng.gauss(0, 1), abs(rng.gauss(0, 1)) + 0.3]))
            self.p3.append([*c, *(d * rng.uniform(8, 24)), 0.0, rng.uniform(2.0, 4.0), 3, rng.uniform(0.08, 0.2)])
        for _ in range(90):  # dust ring
            a = rng.uniform(0, 2 * math.pi)
            spd = rng.uniform(5, 25)
            self.p3.append([c[0], c[1], 0.1, math.cos(a) * spd, math.sin(a) * spd, rng.uniform(0.5, 4.0), 0.0, rng.uniform(1.5, 3.0), 1, rng.uniform(1.0, 2.5)])

    def _step3(self, dt: float) -> None:
        alive = []
        for p in self.p3:
            p[6] += dt
            if p[6] >= p[7]:
                continue
            if p[8] in (1, 3):  # dust and debris fall at lunar gravity
                p[5] -= LUNAR_G * dt
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            if p[2] < 0.05:
                p[2] = 0.05
                if p[8] == 0:  # exhaust hitting the ground fans out
                    h = math.hypot(p[3], p[4]) or 1.0
                    spd = abs(p[5]) * 0.8 + h
                    p[3], p[4], p[5] = p[3] / h * spd, p[4] / h * spd, 0.0
                else:
                    p[3] *= 0.5
                    p[4] *= 0.5
                    p[5] = -p[5] * 0.2
            alive.append(p)
        self.p3 = alive[-3500:]

    # --- frame -------------------------------------------------------------------------

    def _frame(self, st: dict) -> Frame:
        R = self._rotation(st)
        M = model_matrix(st["r"] + (self.wreck[1] if self.dead and self.wreck is not None else 0.0), R)
        thr = self._throttle(st)
        exit_pt, down = self._nozzle(st)
        glow, soft = [], []
        for x, y, z, _vx, _vy, _vz, age, life, kind, size in self.p3:
            t = age / life
            if kind in (0, 2):
                c = FIRE_RAMP[min(int(t * 12), 11)]
                glow.append((x, y, z, size * (1 + t), c[0] / 255, c[1] / 255, c[2] / 255, 1.0))
            elif kind == 1:
                soft.append((x, y, z, size * (1 + 1.5 * t), 0.58, 0.56, 0.53, 0.45 * (1 - t) ** 1.3))
            else:
                soft.append((x, y, z, size, 0.15, 0.14, 0.13, 1.0 - t))
        plume_model, plume_k = None, 0.0
        ground_glow = (0.0, 0.0, 0.0, 0.0)
        if not self.dead:
            length = (3.0 + 9.0 * thr) * self.rng3.uniform(0.9, 1.1)
            if down[2] < -0.05:
                length = min(length, exit_pt[2] / -down[2])  # don't draw the plume into the ground
            if length > 0.2:
                plume_model = M @ model_matrix(self.model.engine_exit, np.eye(3)) @ np.diag([1.0 + 0.5 * thr, 1.0 + 0.5 * thr, length, 1.0])
                plume_k = 0.25 + 0.55 * thr
            glow.append((*exit_pt, 1.6 + 1.6 * thr, 1.0, 0.55, 0.22, 0.9))
            if exit_pt[2] < 40:
                ground_glow = (exit_pt[0], exit_pt[1], 0.0, (1 - exit_pt[2] / 40) * (0.25 + 0.5 * thr))
        # zone beacons: blink red; steady green after a landing on target
        R0 = self.cfg.landing_radius
        for i in range(4):
            a = math.pi / 4 + i * math.pi / 2
            if self.pad_ok:
                col, k = (0.3, 1.0, 0.5), 0.9
            else:
                col, k = (1.0, 0.28, 0.2), 1.0 if (self.t + 0.25 * i) % 1.0 < 0.15 else 0.12
            glow.append((R0 * math.cos(a), R0 * math.sin(a), 1.0, 0.7, *col, k))
        if self.flash3 > 0:
            glow.append((*self.flash_pos3, 14.0 * (0.5 + self.flash3), 1.0, 0.65, 0.35, 0.6 * self.flash3**2))
        alt = max(st["r"][2], 0.0)
        return Frame(
            eye=self.cam_eye, look=self.cam_look, fov=self.cam_fov, lander_model=M, charred=self.dead,
            plume_model=plume_model, plume_intensity=plume_k, glow=glow, soft=soft,
            trail=np.array(self.trail3 + [st["r"]]),
            cone_height=float(np.clip(alt * 1.2, 15.0, 150.0)), limit_ring=alt if alt > 0.5 else None,
            zone_color=(0.3, 1.0, 0.5) if self.pad_ok else (0.95, 0.75, 0.2), ground_glow=ground_glow,
        )
