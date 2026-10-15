"""3D point-mass lunar powered descent, posed as the G-FOLD problem.

See docs/phase2-plan.md for the design rationale. In short:

    r'' = g + T / m          (target-centred frame, z up, ground at z = 0)
    m'  = -|T| / (Isp g0)

- Point mass: the vehicle's attitude is assumed to follow the commanded thrust
  direction instantly (the standard 3DOF assumption of G-FOLD).
- Constraints:
  - Thrust magnitude must stay within [thrust_min, thrust_max] (the engine can't be
    switched off).
  - Thrust may tilt at most max_tilt_deg from vertical.
  - The vehicle must stay above a glide-slope cone around the landing zone.
  The first two can't be violated: actions are mapped into that set. Violating the
  glide slope ends the episode as a failure.
- Control at 1/control_dt Hz. Thrust is held constant over each control step while RK4
  integrates at physics_dt.

All parameters live in DescentConfig and can be overridden from a run config's
``env.kwargs``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar

import gymnasium as gym
import numpy as np

G0 = 9.80665

SUCCESS = frozenset({"landed on target"})


@dataclass(frozen=True)
class DescentConfig:
    # --- environment and vehicle (loosely Apollo LM descent stage near "high gate") ---
    gravity: float = 1.62  # m/s², Moon
    isp: float = 311.0  # s, LM descent engine
    m0: float = 8000.0  # kg at the start of the episode
    propellant: float = 600.0  # kg usable; the vehicle can't go below m0 - propellant
    thrust_min: float = 4500.0  # N; 10% of the LM descent engine's 45 kN
    thrust_max: float = 27000.0  # N; its 60% throttle limit
    max_tilt_deg: float = 45.0  # thrust pointing limit, from vertical
    glide_slope_deg: float | None = 15.0  # min elevation above the landing zone; None disables
    # "terminate": a violation ends the episode (the hard constraint, as in G-FOLD).
    # "penalty": training aid. A violation costs reward every step spent below the cone and
    # the episode continues, but the outcome still records it, so it never counts as a
    # success. Success rates are therefore identical in both modes: the dynamics are the
    # same until the first violation.
    glide_slope_mode: str = "terminate"
    glide_slope_penalty: float = 1.0  # per control step below the cone, x (1 + depth / 10 m)
    # --- success criteria (Apollo LM limits were 10 ft/s vertical, 4 ft/s horizontal) ---
    landing_radius: float = 10.0  # m
    max_touchdown_vz: float = 2.0  # m/s
    max_touchdown_vh: float = 1.0  # m/s
    # --- timing ---
    control_dt: float = 0.2  # s between actions (5 Hz)
    physics_dt: float = 0.05  # s RK4 step
    max_time: float = 150.0  # s, then the episode is truncated
    # --- start conditions, sampled per episode from the reset seed ---
    altitude_range: tuple[float, float] = (1000.0, 2000.0)
    offset_range: tuple[float, float] = (0.0, 1000.0)  # horizontal distance from target
    descent_rate_range: tuple[float, float] = (15.0, 40.0)
    horizontal_speed_range: tuple[float, float] = (0.0, 30.0)
    # Curriculum: at difficulty 0 starts are drawn from the "easy" ranges below; at 1
    # (the default) from the full ranges above; in between, linearly interpolated.
    # Training ramps it up via set_difficulty(); evaluation always uses 1.
    difficulty: float = 1.0
    easy_altitude_range: tuple[float, float] = (100.0, 300.0)
    easy_offset_range: tuple[float, float] = (0.0, 50.0)
    easy_descent_rate_range: tuple[float, float] = (2.0, 10.0)
    easy_horizontal_speed_range: tuple[float, float] = (0.0, 3.0)
    # --- reward ---
    shaping_scale: float = 100.0
    shaping_pos_scale: float = 1000.0  # m
    shaping_vel_scale: float = 50.0  # m/s
    shaping_gamma: float = 0.999  # should match the agent's gamma for the shaping guarantee
    # True: phi = 0 at terminal states, which makes the shaping strictly potential-based
    # (the optimal policy is provably unchanged), but the total shaping is then the same
    # however the episode ends. False: keep phi at the final state, so an ending that is
    # fast or far from the target stays penalised (LunarLander-style; a stronger learning
    # signal, at the cost of the guarantee).
    shaping_zero_at_terminal: bool = True
    propellant_cost: float = 0.1  # reward per kg burned
    success_reward: float = 100.0
    off_target_reward: float = -50.0
    failure_reward: float = -100.0  # crash (binary mode), glide-slope violation, out of propellant
    # "binary": a touchdown pays success / off-target / failure reward.
    # "graded": anything short of success earns partial credit that falls off with how far
    # it missed (a 3 m/s touchdown scores far better than a 40 m/s impact), which gives
    # the agent a gradient towards the narrow success window.
    touchdown_reward: str = "binary"
    graded_base: float = 50.0  # a near miss; a success still pays success_reward
    graded_per_ms: float = 10.0  # per m/s over either speed limit
    graded_per_m: float = 1.0  # per metre outside the landing zone
    graded_floor: float = -100.0

    @property
    def m_min(self) -> float:
        return self.m0 - self.propellant

    @property
    def alpha(self) -> float:
        """Mass flow per newton of thrust, kg/(N s)."""
        return 1.0 / (self.isp * G0)

    @property
    def g(self) -> np.ndarray:
        return np.array([0.0, 0.0, -self.gravity])


def action_to_thrust(action: np.ndarray, cfg: DescentConfig) -> np.ndarray:
    """Map an action in [-1, 1]^3 smoothly onto the feasible thrust set.

    a0 sets the magnitude (-1 -> thrust_min, +1 -> thrust_max). (a1, a2), clipped to
    the unit disk, set the tilt from vertical (radius x max_tilt) and its azimuth.
    """
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    mag = cfg.thrust_min + (a[0] + 1) / 2 * (cfg.thrust_max - cfg.thrust_min)
    h = a[1:3]
    rho = float(np.hypot(*h))
    if rho > 1.0:
        h, rho = h / rho, 1.0
    tilt = rho * math.radians(cfg.max_tilt_deg)
    horiz = h / rho * math.sin(tilt) if rho > 0 else np.zeros(2)
    return mag * np.array([horiz[0], horiz[1], math.cos(tilt)])


def thrust_to_action(thrust: np.ndarray, cfg: DescentConfig) -> np.ndarray:
    """Inverse of action_to_thrust, projecting infeasible thrusts onto the feasible set
    (magnitude clipped to the bounds, tilt clipped to the limit). Used by hand-written
    and optimisation-based controllers."""
    t = np.asarray(thrust, dtype=float)
    mag = float(np.linalg.norm(t))
    a0 = 2 * (min(max(mag, cfg.thrust_min), cfg.thrust_max) - cfg.thrust_min) / (cfg.thrust_max - cfg.thrust_min) - 1
    if mag < 1e-9:
        return np.array([a0, 0.0, 0.0])
    tilt = math.acos(max(-1.0, min(1.0, t[2] / mag)))
    rho = min(tilt / math.radians(cfg.max_tilt_deg), 1.0)
    az = math.atan2(t[1], t[0])
    return np.array([a0, rho * math.cos(az), rho * math.sin(az)])


def rk4_step(r, v, m, thrust, dt, cfg: DescentConfig):
    """One RK4 step of the point-mass dynamics with constant thrust."""
    g, alpha, tmag = cfg.g, cfg.alpha, float(np.linalg.norm(thrust))

    def deriv(r_, v_, m_):
        return v_, g + thrust / m_, -alpha * tmag

    k1 = deriv(r, v, m)
    k2 = deriv(r + dt / 2 * k1[0], v + dt / 2 * k1[1], m + dt / 2 * k1[2])
    k3 = deriv(r + dt / 2 * k2[0], v + dt / 2 * k2[1], m + dt / 2 * k2[2])
    k4 = deriv(r + dt * k3[0], v + dt * k3[1], m + dt * k3[2])
    r_new = r + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    v_new = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    m_new = m + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return r_new, v_new, m_new


RANGES = ("altitude_range", "offset_range", "descent_rate_range", "horizontal_speed_range")


class Descent3DEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": [], "render_fps": 5}

    def __init__(self, render_mode: str | None = None, **overrides):
        if overrides.get("glide_slope_mode", "terminate") not in ("terminate", "penalty"):
            raise ValueError("glide_slope_mode must be 'terminate' or 'penalty'")
        if overrides.get("touchdown_reward", "binary") not in ("binary", "graded"):
            raise ValueError("touchdown_reward must be 'binary' or 'graded'")
        known = {f.name for f in fields(DescentConfig)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"unknown descent3d parameters: {sorted(unknown)}")
        for key in RANGES:  # YAML gives lists
            for k in (key, "easy_" + key):
                if k in overrides:
                    overrides[k] = tuple(overrides[k])
        self.cfg = DescentConfig(**overrides)
        self.difficulty = self.cfg.difficulty
        self.render_mode = render_mode  # drawing is done by the viewer's scene, not the env
        self.metadata = {**self.metadata, "render_fps": round(1 / self.cfg.control_dt)}
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        # [r (m), v (m/s), propellant remaining (kg)]; generous physical bounds
        hi = np.array([1e4, 1e4, 1e4, 500, 500, 500, self.cfg.propellant], np.float32)
        lo = np.array([-1e4, -1e4, 0, -500, -500, -500, 0], np.float32)
        self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)
        self.n_substeps = round(self.cfg.control_dt / self.cfg.physics_dt)

    def set_difficulty(self, difficulty: float) -> None:
        """Curriculum hook, called by training (see train.py's curriculum option)."""
        self.difficulty = min(max(float(difficulty), 0.0), 1.0)

    def start_range(self, name: str) -> tuple[float, float]:
        full = getattr(self.cfg, name)
        if self.difficulty >= 1.0:
            return full
        easy, d = getattr(self.cfg, "easy_" + name), self.difficulty
        return easy[0] + (full[0] - easy[0]) * d, easy[1] + (full[1] - easy[1]) * d

    # --- helpers ----------------------------------------------------------------

    def glide_slope_margin(self, r: np.ndarray) -> float:
        """Metres above the glide-slope cone (negative = violated). The cone rises
        from the edge of the landing zone, so a valid landing anywhere inside the
        zone never violates it."""
        if self.cfg.glide_slope_deg is None:
            return math.inf
        d = max(0.0, float(np.hypot(r[0], r[1])) - self.cfg.landing_radius)
        return float(r[2]) - math.tan(math.radians(self.cfg.glide_slope_deg)) * d

    def glide_angle_margin_deg(self, r: np.ndarray) -> float:
        """Elevation of the vehicle above the landing-zone edge minus the glide-slope
        angle, in degrees (inf inside the zone, where the cone doesn't constrain)."""
        if self.cfg.glide_slope_deg is None:
            return math.inf
        d = float(np.hypot(r[0], r[1])) - self.cfg.landing_radius
        return math.degrees(math.atan2(float(r[2]), d)) - self.cfg.glide_slope_deg if d > 0 else math.inf

    def _potential(self) -> float:
        c = self.cfg
        return -(np.linalg.norm(self.r) / c.shaping_pos_scale + np.linalg.norm(self.v) / c.shaping_vel_scale)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.r, self.v, [self.m - self.cfg.m_min]]).astype(np.float32)

    def _terminal_reward(self, outcome: str) -> float:
        c = self.cfg
        if outcome == "landed on target":
            return c.success_reward
        if self.touchdown is None:  # glide-slope violation (terminate mode) or out of propellant
            return c.failure_reward
        if c.touchdown_reward == "binary":
            return c.off_target_reward if outcome == "landed off target" else c.failure_reward
        td = self.touchdown
        miss = (
            c.graded_per_ms * (max(0.0, td["vz"] - c.max_touchdown_vz) + max(0.0, td["vh"] - c.max_touchdown_vh))
            + c.graded_per_m * max(0.0, td["position_error"] - c.landing_radius)
        )
        return max(c.graded_base - miss, c.graded_floor)

    def _classify_touchdown(self, r: np.ndarray, v: np.ndarray) -> str:
        c = self.cfg
        if -v[2] <= c.max_touchdown_vz and np.hypot(v[0], v[1]) <= c.max_touchdown_vh:
            return "landed on target" if np.hypot(r[0], r[1]) <= c.landing_radius else "landed off target"
        return "crashed"

    # --- gym API ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c, rng = self.cfg, self.np_random
        alt = rng.uniform(*self.start_range("altitude_range"))
        dist, phi = rng.uniform(*self.start_range("offset_range")), rng.uniform(0, 2 * math.pi)
        vz = -rng.uniform(*self.start_range("descent_rate_range"))
        vh, psi = rng.uniform(*self.start_range("horizontal_speed_range")), rng.uniform(0, 2 * math.pi)
        self.r = np.array([dist * math.cos(phi), dist * math.sin(phi), alt])
        self.v = np.array([vh * math.cos(psi), vh * math.sin(psi), vz])
        self.m = c.m0
        self.t = 0.0
        self.thrust = np.zeros(3)
        self.outcome: str | None = None
        self.touchdown: dict[str, float] | None = None
        self.min_glide_margin_deg = self.glide_angle_margin_deg(self.r)
        self.glide_violation_s = 0.0
        self._phi = self._potential()
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        self.thrust = action_to_thrust(action, c)
        m_start = self.m
        outcome = None
        depth = 0.0  # deepest glide-slope violation this step (penalty mode)
        for _ in range(self.n_substeps):
            r, v, m = rk4_step(self.r, self.v, self.m, self.thrust, c.physics_dt, c)
            if r[2] <= 0.0:
                # Interpolate to the moment of ground contact for accurate touchdown values.
                f = self.r[2] / (self.r[2] - r[2])
                r, v, m = self.r + f * (r - self.r), self.v + f * (v - self.v), self.m + f * (m - self.m)
                r[2] = 0.0
                self.t += f * c.physics_dt
                self.r, self.v, self.m = r, v, m
                outcome = self._classify_touchdown(r, v)
                self.touchdown = {
                    "position_error": float(np.hypot(r[0], r[1])),
                    "vz": float(-v[2]),
                    "vh": float(np.hypot(v[0], v[1])),
                }
                break
            self.r, self.v, self.m = r, v, max(m, c.m_min)
            self.t += c.physics_dt
            self.min_glide_margin_deg = min(self.min_glide_margin_deg, self.glide_angle_margin_deg(r))
            margin = self.glide_slope_margin(r)
            if margin < 0:
                if c.glide_slope_mode == "terminate":
                    outcome = "glide-slope violation"
                    break
                depth = max(depth, -margin)
                self.glide_violation_s += c.physics_dt
            if m <= c.m_min:
                outcome = "out of propellant"
                break

        terminated = outcome is not None
        truncated = not terminated and self.t >= c.max_time - 1e-9
        self.outcome = outcome or ("timeout" if truncated else None)
        if self.outcome and self.glide_violation_s > 0:  # penalty mode: the violation is never forgotten
            self.outcome = f"{self.outcome} after glide-slope violation"

        # Shaping F = gamma * phi(s') - phi(s) (see shaping_zero_at_terminal), plus the propellant cost.
        phi_new = 0.0 if terminated and c.shaping_zero_at_terminal else self._potential()
        reward = c.shaping_scale * (c.shaping_gamma * phi_new - self._phi) - c.propellant_cost * (m_start - self.m)
        self._phi = phi_new
        if depth > 0:
            reward -= c.glide_slope_penalty * (1 + depth / 10.0)
        if terminated:  # by how the episode physically ended; a violation was already penalised per step
            reward += self._terminal_reward(outcome)

        info = {}
        if terminated or truncated:
            info["is_success"] = self.outcome in SUCCESS  # picked up by SB3's EvalCallback
        return self._obs(), float(reward), terminated, truncated, info


# --- registry hooks ---------------------------------------------------------------


def scene():
    """The 3D scene when OpenGL is available, else the 2D one. Imported lazily: only the
    viewer needs drawing code."""
    try:
        from lander.envs.descent3d_scene3d import Descent3DScene3D

        return Descent3DScene3D()
    except Exception as e:  # noqa: BLE001 - no OpenGL, missing driver, ...
        print(f"3D viewer unavailable ({e!r}); using the 2D scene", flush=True)
        return scene_2d()


def scene_2d():
    from lander.envs.descent3d_scene import Descent3DScene

    return Descent3DScene()


def make(render_mode: str | None = None, **kwargs) -> gym.Env:
    return Descent3DEnv(render_mode=render_mode, **kwargs)


def outcome(env: gym.Env, terminated: bool, truncated: bool) -> str:
    return env.unwrapped.outcome or "flying"


def metrics(env: gym.Env, observations: list, actions: list) -> dict[str, float]:
    u = env.unwrapped
    td = u.touchdown or {"position_error": math.nan, "vz": math.nan, "vh": math.nan}
    return {
        "position_error_m": td["position_error"],
        "touchdown_vz_ms": td["vz"],
        "touchdown_vh_ms": td["vh"],
        "propellant_used_kg": u.cfg.m0 - u.m,
        # closest approach to the glide-slope cone outside the landing zone (inf if never outside)
        "min_glide_margin_deg": u.min_glide_margin_deg,
        "glide_violation_s": u.glide_violation_s,  # time below the cone (penalty mode only)
        "flight_time_s": u.t,
    }
