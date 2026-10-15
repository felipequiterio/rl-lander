"""Environment registry.

Training, evaluation and the viewer never construct environments directly: they
look one up here by the name given in the run config (``env.name``). To add the
Phase 2 powered-descent env, write its module and add one ``EnvSpec`` entry to
``REGISTRY`` - nothing else needs to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from lander.envs import descent3d, lunar_lander


@dataclass(frozen=True)
class EnvSpec:
    # Builds one env instance: make(render_mode=None|"rgb_array", **kwargs) -> gym.Env
    make: Callable[..., gym.Env]
    # Classifies how an episode ended, for the viewer overlay:
    # outcome(env, terminated, truncated) -> short label like "landed" / "crashed".
    outcome: Callable[[gym.Env, bool, bool], str]
    # Mean eval reward at which the task counts as "solved" (None = no threshold).
    solved_threshold: float | None = None
    # Which outcome labels count as mission success (for lander-eval's success rate).
    success: frozenset[str] = frozenset()
    # Optional per-episode metrics for lander-eval: metrics(env, observations, actions)
    # -> {name: value}, called after the episode with its full trajectory
    # (observations has one more entry than actions: the reset observation).
    metrics: Callable[[gym.Env, list, list], dict[str, float]] | None = None
    # Optional custom renderer for the viewer: scene() -> object with the methods
    # of lander.envs.lunar_lander_scene.LunarScene (reset/update/draw/gauges/...).
    # Without one, the viewer shows the env's own rgb_array frames.
    scene: Callable[[], Any] | None = None
    # Optional named alternatives, selectable with `lander-viewer --scene NAME`.
    scenes: dict[str, Callable[[], Any]] = field(default_factory=dict)


REGISTRY: dict[str, EnvSpec] = {
    "lunar_lander": EnvSpec(
        make=lunar_lander.make,
        outcome=lunar_lander.outcome,
        solved_threshold=lunar_lander.SOLVED_THRESHOLD,
        scene=lunar_lander.scene,
        success=lunar_lander.SUCCESS,
        metrics=lunar_lander.metrics,
    ),
    # Phase 2: 3D point-mass lunar powered descent (docs/phase2-plan.md).
    "descent3d": EnvSpec(
        make=descent3d.make,
        outcome=descent3d.outcome,
        success=descent3d.SUCCESS,
        metrics=descent3d.metrics,
        scene=descent3d.scene,
        scenes={"3d": descent3d.scene, "2d": descent3d.scene_2d},
    ),
}


def get_spec(name: str) -> EnvSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown env {name!r}. Registered: {sorted(REGISTRY)}") from None


def make_env(name: str, render_mode: str | None = None, **kwargs: Any) -> gym.Env:
    return get_spec(name).make(render_mode=render_mode, **kwargs)
