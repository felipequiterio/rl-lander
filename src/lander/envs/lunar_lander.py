"""Gymnasium LunarLander-v3 (continuous actions).

Observation (8): x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact.
Action (2, in [-1, 1]): main engine throttle (off below 0), side engines (|a| < 0.5 = off).

Reward is shaped every step: moving closer to the pad / slower / more upright earns
points, each leg contact +10, firing engines costs a little fuel. The episode ends with
+100 for coming to rest or -100 for crashing / flying off-screen. A good landing on
the pad totals roughly 200-300 per episode.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W

# Gymnasium's official "solved" bar for LunarLander: an average episode return of 200.
SOLVED_THRESHOLD = 200.0

# What we actually want, independent of reward: come to rest between the flags.
SUCCESS = frozenset({"landed on pad"})


def make(render_mode: str | None = None, continuous: bool = True, **kwargs) -> gym.Env:
    return gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode, **kwargs)


def scene():
    # Imported lazily: only the viewer needs pygame drawing code.
    from lander.envs.lunar_lander_scene import LunarScene

    return LunarScene()


def metrics(env: gym.Env, observations: list, actions: list) -> dict[str, float]:
    """Landing quality. Units are the sim's world units (the pad is 4 wide)."""
    u = env.unwrapped
    obs = np.asarray(observations, dtype=float)
    a = np.asarray(actions, dtype=float).reshape(-1, 2)
    # Speed at first leg contact. The observation stores velocity scaled by
    # (half viewport size in world units) / FPS, so undo that.
    contact = np.flatnonzero((obs[:, 6] > 0) | (obs[:, 7] > 0))
    if len(contact):
        o = obs[contact[0]]
        vx = o[2] / (VIEWPORT_W / SCALE / 2) * FPS
        vy = o[3] / (VIEWPORT_H / SCALE / 2) * FPS
        touchdown_speed = math.hypot(vx, vy)
    else:
        touchdown_speed = float("nan")
    # Engine power per step, mirroring LunarLander.step(): main 50-100% when a0 > 0, side 50-100% when |a1| > 0.5.
    main = np.where(a[:, 0] > 0, (np.clip(a[:, 0], 0, 1) + 1) / 2, 0.0)
    side = np.where(np.abs(a[:, 1]) > 0.5, np.clip(np.abs(a[:, 1]), 0.5, 1), 0.0)
    return {
        "pad_offset": abs(u.lander.position.x - VIEWPORT_W / SCALE / 2),  # distance from pad centre
        "touchdown_speed": touchdown_speed,
        "final_tilt_deg": abs(math.degrees(u.lander.angle)),
        "main_engine_s": float(main.sum() / FPS),  # fuel proxy: seconds at full main-engine power
        "side_engine_s": float(side.sum() / FPS),
        "duration_s": len(a) / FPS,
    }


def outcome(env: gym.Env, terminated: bool, truncated: bool) -> str:
    """Mirror the termination logic in LunarLander.step()."""
    if not (terminated or truncated):
        return "flying"
    if truncated:
        return "timeout (hovering)"  # TimeLimit hit: 1000 steps without landing or crashing
    u = env.unwrapped
    if u.game_over:
        return "crashed"
    half_w = VIEWPORT_W / SCALE / 2  # world is centred on the pad
    x = u.lander.position.x
    if abs(x - half_w) / half_w >= 1.0:
        return "flew off-screen"
    if not u.lander.awake:
        return "landed on pad" if u.helipad_x1 <= x <= u.helipad_x2 else "landed off pad"
    return "ended"
