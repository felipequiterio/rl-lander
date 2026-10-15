"""Uneven terrain: heightmaps, target selection, and descent3d's terrain mode."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lander.config import parse_env_args, with_env_args
from lander.envs import make_env
from lander.envs.ground import Heightmap, available_sites


def plane(gx: float, gy: float, n: int = 200, dx: float = 5.0) -> Heightmap:
    """A tilted plane z = gx * x + gy * y over [-n*dx/2, n*dx/2]^2."""
    c = (np.arange(n) + 0.5) * dx - n * dx / 2
    x, y = np.meshgrid(c, c)
    return Heightmap(gx * x + gy * y, (-n * dx / 2, -n * dx / 2, n * dx / 2, n * dx / 2), name="plane")


def test_bilinear_heights_are_exact_on_a_plane():
    m = plane(0.1, -0.05)
    rng = np.random.default_rng(0)
    for x, y in rng.uniform(-400, 400, (50, 2)):
        assert m.height(x, y) == pytest.approx(0.1 * x - 0.05 * y, abs=1e-9)
    xs, ys = rng.uniform(-400, 400, (2, 100))
    np.testing.assert_allclose(m.heights(xs, ys), [m.height(x, y) for x, y in zip(xs, ys, strict=True)])
    assert m.height(10_000.0, 0.0) == pytest.approx(m.height(497.5, 0.0))  # clamped at the edge


def test_safe_targets_avoid_slopes_and_edges():
    m = plane(0.0, 0.0)
    m.z[:, 120:] += np.arange(80) * 1.0  # a 1:5 ramp (11 deg) east of x = 100
    t = m.safe_targets(6.0, 15.0, 100.0)
    assert len(t) > 0
    assert t[:, 0].max() < 100 - 15 + 5  # nothing near the ramp
    assert np.abs(t).max() <= 500 - 100  # nothing near the edge
    near = m.safe_targets(6.0, 15.0, 100.0, max_dist=50.0)
    assert 0 < len(near) < len(t) and np.hypot(*near.T).max() <= 50.0


def terrain_env(m: Heightmap, target=(0.0, 0.0), **kwargs):
    """A descent3d env flying over heightmap m (needs a real site to build, then swaps it)."""
    env = make_env("descent3d", terrain="apollo11", glide_slope_deg=None, **kwargs).unwrapped
    env.sites, env.targets = [m], [np.array([target])]
    env.reset(seed=0)
    return env


def place(env, r, v):
    env.r, env.v = np.array(r, float), np.array(v, float)
    env._phi = env._potential()


def fly(env, action=(-1.0, 0.0, 0.0), max_steps=200):
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(np.array(action, np.float32))
        if terminated or truncated:
            return info
    raise AssertionError("episode didn't end")


def test_frame_is_target_centred():
    env = terrain_env(plane(0.05, 0.0), target=(100.0, 50.0))
    assert env.ground_height(0.0, 0.0) == pytest.approx(0.0)
    assert env.ground_height(20.0, 0.0) == pytest.approx(1.0)  # 20 m east, 5% grade
    assert env.r[2] - env.ground_height(*env.r[:2]) >= env.cfg.min_start_agl - 1e-6


def test_lands_on_raised_ground_and_tips_over_on_slopes():
    m = plane(0.0, 0.0)
    m.z[:, 130:] = 40.0  # a 40 m mesa from x = 150
    env = terrain_env(m)
    place(env, [0.0, 0.0, 0.3], [0.0, 0.0, -1.0])
    fly(env)
    assert env.outcome == "landed on target" and env.touchdown["slope_deg"] < 1
    env.reset(seed=0)
    place(env, [300.0, 0.0, 40.3], [0.0, 0.0, -1.0])  # touches down on the mesa, 40 m up
    fly(env)
    assert env.outcome == "landed off target" and env.r[2] == pytest.approx(40.0, abs=0.01)
    env.reset(seed=0)
    place(env, [0.0, 0.0, 20.0], [40.0, 0.0, 0.0])  # flies east, below the top, into the mesa's wall
    fly(env)
    assert env.outcome == "crashed" and 147 < env.r[0] < 153 and 0 < env.r[2] < 20

    steep = terrain_env(plane(math.tan(math.radians(20)), 0.0))  # 20 deg everywhere
    place(steep, [0.0, 0.0, 0.3], [0.0, 0.0, -1.0])
    fly(steep)
    assert steep.outcome == "tipped over" and steep.touchdown["slope_deg"] == pytest.approx(20, abs=0.1)


def test_graded_reward_charges_for_slope():
    steep = terrain_env(plane(math.tan(math.radians(20)), 0.0), touchdown_reward="graded")
    place(steep, [0.0, 0.0, 0.3], [0.0, 0.0, -1.0])
    fly(steep)
    c = steep.cfg
    assert steep._terminal_reward(steep.outcome) == pytest.approx(c.graded_base - c.graded_per_deg * (20 - c.max_landing_slope_deg), abs=1)


def test_terrain_observations():
    m = plane(0.0, 0.0)
    m.z[:, 110:] = 30.0  # a 30 m step from x = 50
    env = terrain_env(m)
    place(env, [200.0, 0.0, 100.0], [-10.0, 0.0, 0.0])
    obs = env._obs()
    c = env.cfg
    assert obs.shape == (7 + 1 + c.profile_points + len(c.lookahead_s),)
    assert obs[7] == pytest.approx(70.0)  # radar altimeter: 100 m up, over the 30 m step
    profile = obs[8 : 8 + c.profile_points]
    # along the line down to the target: the step sticks up through it at 1/9 of the way,
    # the line passes over it at 3/4, and only the valley floor is under its last part
    f = np.arange(1, c.profile_points + 1) / (c.profile_points + 1)
    expect = np.where(200 * (1 - f) > 50, 30.0, 0.0) - 100 * (1 - f)
    np.testing.assert_allclose(profile, expect, atol=1e-4)
    ahead = obs[8 + c.profile_points :]
    np.testing.assert_allclose(ahead, [100 - 30.0 if 200 - 10 * t > 50 else 100.0 for t in c.lookahead_s], atol=1e-4)


def test_relief_follows_the_curriculum():
    env = make_env("descent3d", terrain="apollo17").unwrapped
    env.set_difficulty(0.0)
    env.reset(seed=3)
    assert env.relief == 0.0 and env.ground_height(500.0, -300.0) == 0.0
    assert env.r[2] <= env.cfg.easy_altitude_range[1]  # the easy starts stay low (no clearance bump on flat ground)
    env.set_difficulty(1.0)
    env.reset(seed=3)
    xs = np.linspace(-1000, 1000, 21)
    assert max(abs(env.ground_height(x, x)) for x in xs) > 1.0  # real relief


@pytest.mark.parametrize("site", ["apollo11", "apollo17"])
def test_real_sites(site):
    assert site in available_sites()
    env = make_env("descent3d", terrain=site)
    check_env(env, skip_render_check=True)
    u = env.unwrapped
    for seed in range(20):
        env.reset(seed=seed)
        assert u.ground_slope_deg(0.0, 0.0) <= u.cfg.target_max_slope_deg + 1.0
        assert u.r[2] - u.ground_height(*u.r[:2]) >= u.cfg.min_start_agl - 1e-6


def test_flat_ground_is_unchanged():
    env = make_env("descent3d").unwrapped
    obs, _ = env.reset(seed=0)
    assert obs.shape == (7,) and env.site is None and env.ground_height(123.0, 45.0) == 0.0


def test_env_args():
    over = parse_env_args(["terrain=apollo11", "propellant=800", "lookahead_s=[1, 2]"])
    assert over == {"terrain": "apollo11", "propellant": 800, "lookahead_s": [1, 2]}
    record = {"env": {"name": "descent3d", "kwargs": {"terrain": "apollo17", "propellant_cost": 0.3}}}
    assert with_env_args(record, over)["env"]["kwargs"] == {"terrain": "apollo11", "propellant_cost": 0.3, "propellant": 800,
                                                           "lookahead_s": [1, 2]}
    assert record["env"]["kwargs"]["terrain"] == "apollo17"  # the original is untouched
    with pytest.raises(ValueError):
        parse_env_args(["terrain"])
