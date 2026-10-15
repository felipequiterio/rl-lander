"""Smoke tests for the viewer scenes: they run headless and draw every state."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from lander.envs import get_spec, make_env
from lander.hud import TONES


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


@pytest.mark.parametrize(("env_name", "kwargs"), [("lunar_lander", {"continuous": True}), ("descent3d", {})])
def test_scene_runs_an_episode_and_draws(env_name, kwargs):
    spec = get_spec(env_name)
    scene = spec.scene()
    env = make_env(env_name, render_mode=scene.render_mode, **kwargs)
    canvas = pygame.Surface(scene.size)
    rng = np.random.default_rng(0)
    env.reset(seed=1)
    scene.reset(env)
    substeps = getattr(scene, "substeps", 1)
    for _ in range(40):
        action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        _, _, te, tr, _ = env.step(action)
        for k in range(substeps):
            scene.update(env, action, 0.02, frac=(k + 1) / substeps)
        scene.draw(canvas, env)
        if te or tr:
            break
    label = spec.outcome(env, te, tr)
    scene.on_outcome(env, label)
    scene.update(env, None, 0.02)
    scene.draw(canvas, env)
    assert all(g.tone in TONES for g in scene.gauges(env))
    _, pan = scene.sound_levels(env)
    assert -1.5 <= pan <= 1.5


def test_descent3d_banners_cover_every_outcome():
    from lander.envs.descent3d_scene import BANNERS

    scene = get_spec("descent3d").scene()
    for base in ("landed on target", "landed off target", "crashed", "glide-slope violation", "out of propellant", "timeout"):
        assert base in BANNERS
        for label in (base, f"{base} after glide-slope violation"):
            title, _, tone = scene.banner(label)
            assert title == BANNERS[base][0] and tone in TONES
            assert scene.outcome_sound(label) in ("success", "touchdown", "explosion", "fail")
    assert scene.banner("landed on target after glide-slope violation")[2] == "bad"
