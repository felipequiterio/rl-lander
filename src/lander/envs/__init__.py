"""Environment registry.

Training, evaluation and the viewer never construct environments directly: they
look one up here by the name given in the run config (``env.name``). To add the
Phase 2 powered-descent env, write its module and add one ``EnvSpec`` entry to
``REGISTRY`` - nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from lander.envs import lunar_lander


@dataclass(frozen=True)
class EnvSpec:
    # Builds one env instance: make(render_mode=None|"rgb_array", **kwargs) -> gym.Env
    make: Callable[..., gym.Env]
    # Classifies how an episode ended, for the viewer overlay:
    # outcome(env, terminated, truncated) -> short label like "landed" / "crashed".
    outcome: Callable[[gym.Env, bool, bool], str]
    # Mean eval reward at which the task counts as "solved" (None = no threshold).
    solved_threshold: float | None = None


REGISTRY: dict[str, EnvSpec] = {
    "lunar_lander": EnvSpec(
        make=lunar_lander.make,
        outcome=lunar_lander.outcome,
        solved_threshold=lunar_lander.SOLVED_THRESHOLD,
    ),
}


def get_spec(name: str) -> EnvSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown env {name!r}. Registered: {sorted(REGISTRY)}") from None


def make_env(name: str, render_mode: str | None = None, **kwargs: Any) -> gym.Env:
    return get_spec(name).make(render_mode=render_mode, **kwargs)
