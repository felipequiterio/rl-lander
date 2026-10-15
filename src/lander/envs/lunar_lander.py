"""Gymnasium LunarLander-v3 (continuous actions).

Observation (8): x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact.
Action (2, in [-1, 1]): main engine throttle (off below 0), side engines (|a| < 0.5 = off).

Reward is shaped every step: moving closer to the pad / slower / more upright earns
points, each leg contact +10, firing engines costs a little fuel. The episode ends with
+100 for coming to rest or -100 for crashing / flying off-screen. A good landing on
the pad totals roughly 200-300 per episode.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_W

# Gymnasium's official "solved" bar for LunarLander: an average episode return of 200.
SOLVED_THRESHOLD = 200.0


def make(render_mode: str | None = None, continuous: bool = True, **kwargs) -> gym.Env:
    return gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode, **kwargs)


def scene():
    # Imported lazily: only the viewer needs pygame drawing code.
    from lander.envs.lunar_lander_scene import LunarScene

    return LunarScene()


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
