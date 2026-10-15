"""Physics and interface checks for the descent3d env."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lander.controllers import CONTROLLERS
from lander.envs import get_spec, make_env
from lander.envs.descent3d import G0, DescentConfig, action_to_thrust, rk4_step, thrust_to_action

CFG = DescentConfig()


def fresh_env(**kwargs):
    env = make_env("descent3d", **kwargs)
    env.reset(seed=0)
    return env.unwrapped


def place(env, r, v, propellant=None):
    """Put the vehicle in a chosen state (tests only)."""
    env.r, env.v = np.array(r, float), np.array(v, float)
    if propellant is not None:
        env.m = env.cfg.m_min + propellant
    env.min_glide_margin_deg = env.glide_angle_margin_deg(env.r)
    env._phi = env._potential()


def test_gymnasium_api():
    check_env(make_env("descent3d"), skip_render_check=True)


def test_free_fall_matches_closed_form():
    # With zero thrust the acceleration is constant, which RK4 integrates exactly.
    r0, v0, t = np.array([10.0, -5.0, 1000.0]), np.array([3.0, 1.0, -20.0]), 0.0
    r, v, m = r0, v0, 8000.0
    for _ in range(40):
        r, v, m = rk4_step(r, v, m, np.zeros(3), 0.05, CFG)
        t += 0.05
    np.testing.assert_allclose(r, r0 + v0 * t + 0.5 * CFG.g * t**2, atol=1e-9)
    np.testing.assert_allclose(v, v0 + CFG.g * t, atol=1e-9)
    assert m == 8000.0


def test_hover_holds_velocity():
    m = 8000.0
    thrust = np.array([0.0, 0.0, m * CFG.gravity])
    r, v = np.array([0.0, 0.0, 500.0]), np.zeros(3)
    for _ in range(20):  # 1 s; mass loss makes T/m creep up by ~0.06%
        r, v, m = rk4_step(r, v, m, thrust, 0.05, CFG)
    assert np.abs(v).max() < 0.01


def test_mass_flow_rate():
    env = fresh_env()
    m0 = env.m
    action = np.array([0.3, 0.2, -0.4], np.float32)
    env.step(action)
    expected = np.linalg.norm(action_to_thrust(action, CFG)) / (CFG.isp * G0) * CFG.control_dt
    assert m0 - env.m == pytest.approx(expected, rel=1e-9)


def test_actions_always_map_to_feasible_thrust():
    rng = np.random.default_rng(0)
    for a in rng.uniform(-3, 3, size=(2000, 3)):  # including out-of-range actions
        t = action_to_thrust(a, CFG)
        mag = np.linalg.norm(t)
        assert CFG.thrust_min - 1e-6 <= mag <= CFG.thrust_max + 1e-6
        assert math.degrees(math.acos(t[2] / mag)) <= CFG.max_tilt_deg + 1e-6


def test_thrust_to_action_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(500):
        a = rng.uniform(-1, 1, 3)
        if np.hypot(a[1], a[2]) > 1:
            continue
        np.testing.assert_allclose(thrust_to_action(action_to_thrust(a, CFG), CFG), a, atol=1e-9)


def test_infeasible_thrust_is_projected():
    a = thrust_to_action(np.array([1e6, 0.0, -1.0]), CFG)  # far too strong, pointing down
    t = action_to_thrust(a, CFG)
    assert np.linalg.norm(t) == pytest.approx(CFG.thrust_max)
    assert math.degrees(math.acos(t[2] / np.linalg.norm(t))) == pytest.approx(CFG.max_tilt_deg)


def test_same_seed_same_start():
    e1, e2 = make_env("descent3d"), make_env("descent3d")
    np.testing.assert_array_equal(e1.reset(seed=42)[0], e2.reset(seed=42)[0])
    assert not np.array_equal(e1.reset(seed=42)[0], e1.reset(seed=43)[0])


def test_start_conditions_within_ranges():
    env = make_env("descent3d")
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        assert 1000 <= obs[2] <= 2000 and np.hypot(obs[0], obs[1]) <= 1000
        assert -40 <= obs[5] <= -15 and np.hypot(obs[3], obs[4]) <= 30
        assert obs[6] == pytest.approx(CFG.propellant)


def hold_min_thrust(env):
    return env.step(np.array([-1.0, 0.0, 0.0], np.float32))


def run_until_done(env, max_steps=100):
    for _ in range(max_steps):
        out = hold_min_thrust(env)
        if out[2] or out[3]:
            return out
    raise AssertionError("episode did not end")


@pytest.mark.parametrize(
    ("r", "v", "expected"),
    [
        ([2.0, 0.0, 0.1], [0.0, 0.0, -1.0], "landed on target"),
        ([30.0, 0.0, 0.1], [0.0, 0.0, -1.0], "landed off target"),  # (glide slope disabled below)
        ([0.0, 0.0, 0.1], [0.0, 0.0, -5.0], "crashed"),
        ([0.0, 0.0, 0.1], [3.0, 0.0, -1.0], "crashed"),
    ],
)
def test_touchdown_classification(r, v, expected):
    env = fresh_env(glide_slope_deg=None)
    place(env, r, v)
    _, _, terminated, _, info = hold_min_thrust(env)
    assert terminated and env.outcome == expected
    assert env.r[2] == 0.0 and env.touchdown is not None
    assert info["is_success"] == (expected == "landed on target")


def test_touchdown_is_interpolated_to_ground_contact():
    env = fresh_env()
    place(env, [0, 0, 0.5], [0, 0, -1.2])
    t0 = env.t
    run_until_done(env)
    # Closed form with constant net acceleration a = g - T/m (mass change is negligible here):
    # 0.5 = 1.2 t + a t^2 / 2  ->  contact time and speed, independent of the 0.2 s step grid.
    a = CFG.gravity - CFG.thrust_min / env.m
    t_hit = (-1.2 + math.sqrt(1.2**2 + 2 * a * 0.5)) / a
    assert env.t - t0 == pytest.approx(t_hit, abs=1e-3)
    assert env.touchdown["vz"] == pytest.approx(1.2 + a * t_hit, abs=1e-3)


def test_glide_slope_violation_terminates():
    env = fresh_env()
    place(env, [400.0, 0.0, 50.0], [0.0, 0.0, -10.0])  # far out, low: below the 15 deg cone
    phi, m = env._phi, env.m
    _, reward, terminated, _, _ = hold_min_thrust(env)
    assert terminated and env.outcome == "glide-slope violation"
    # Terminal reward + shaping to phi = 0 (potential-based shaping) - propellant cost.
    c = env.cfg
    assert reward == pytest.approx(c.failure_reward - c.shaping_scale * phi - c.propellant_cost * (m - env.m))


def test_glide_slope_allows_landing_inside_zone():
    env = fresh_env()
    assert env.glide_slope_margin(np.array([9.0, 0.0, 0.01])) > 0  # inside the 10 m zone, at ground level
    assert env.glide_slope_margin(np.array([20.0, 0.0, 1.0])) < 0


def test_glide_angle_margin():
    env = fresh_env()
    assert env.glide_angle_margin_deg(np.array([110.0, 0.0, 100.0])) == pytest.approx(45.0 - 15.0)
    assert env.glide_angle_margin_deg(np.array([5.0, 0.0, 1.0])) == math.inf  # inside the zone


def test_penalty_mode_continues_but_never_counts_as_success():
    env = fresh_env(glide_slope_mode="penalty")
    place(env, [40.0, 0.0, 2.0], [-30.0, 0.0, -1.0])  # below the cone, but heading into the zone
    _, _, terminated, _, _ = hold_min_thrust(env)
    assert not terminated and env.glide_violation_s > 0
    # Then a perfect touchdown in the zone: physically a landing, but not a success.
    place(env, [2.0, 0.0, 0.1], [0.0, 0.0, -1.0])
    _, _, terminated, _, info = run_until_done(env)
    assert env.outcome == "landed on target after glide-slope violation"
    assert info["is_success"] is False


def test_penalty_scales_with_depth():
    shallow, deep = fresh_env(glide_slope_mode="penalty"), fresh_env(glide_slope_mode="penalty")
    place(shallow, [400.0, 0.0, 100.0], [0.0, 0.0, 0.0])  # ~4 m below the cone
    place(deep, [400.0, 0.0, 10.0], [0.0, 0.0, 0.0])  # ~94 m below
    r_shallow, r_deep = hold_min_thrust(shallow)[1], hold_min_thrust(deep)[1]
    assert r_deep < r_shallow


def test_graded_touchdown_reward_orders_misses():
    rewards = {}
    for vz in (1.0, 3.0, 7.0, 40.0):
        env = fresh_env(touchdown_reward="graded", glide_slope_deg=None)
        place(env, [0, 0, 0.01], [0, 0, -vz])
        hold_min_thrust(env)
        rewards[vz] = env._terminal_reward(env.outcome)
    assert rewards[1.0] == env.cfg.success_reward
    assert rewards[1.0] > rewards[3.0] > rewards[7.0] > rewards[40.0] == env.cfg.graded_floor
    assert rewards[3.0] == pytest.approx(env.cfg.graded_base - env.cfg.graded_per_ms * (3.0 - 2.0), abs=0.2)


def test_curriculum_ranges():
    env = make_env("descent3d").unwrapped
    env.set_difficulty(0.0)
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        assert 100 <= obs[2] <= 300 and np.hypot(obs[0], obs[1]) <= 50
    env.set_difficulty(0.5)
    assert env.start_range("altitude_range") == pytest.approx((550.0, 1150.0))
    # difficulty 1 (the default) draws exactly the original start states
    env.set_difficulty(1.0)
    np.testing.assert_array_equal(env.reset(seed=7)[0], make_env("descent3d").reset(seed=7)[0])


def test_out_of_propellant_terminates():
    env = fresh_env()
    place(env, [0, 0, 500], [0, 0, -10], propellant=0.5)  # ~0.34 s at minimum thrust
    run_until_done(env)
    assert env.outcome == "out of propellant"
    assert env.m == pytest.approx(env.cfg.m_min)


def test_timeout_truncates():
    env = fresh_env(max_time=1.0)
    place(env, [0, 0, 1000], [0, 0, 0])
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0], np.float32))
    assert truncated and not terminated and env.outcome == "timeout" and info["is_success"] is False


def test_shaping_terms_telescope():
    # Total shaping over an episode = -phi(s0) (phi is 0 at terminal states), whatever happens.
    # Check with gamma = 1 and no other reward terms: the return must equal -phi(s0).
    env = make_env("descent3d", shaping_gamma=1.0, propellant_cost=0.0, success_reward=0.0,
                   failure_reward=0.0, off_target_reward=0.0).unwrapped
    env.reset(seed=3)
    phi0 = env._potential()
    total, done = 0.0, False
    while not done:
        _, r, te, tr, _ = env.step(env.action_space.sample())
        total += r
        done = te or tr
    if te:
        assert total == pytest.approx(-env.cfg.shaping_scale * phi0, rel=1e-9)


def test_unknown_parameter_is_rejected():
    with pytest.raises(TypeError, match="unknown descent3d parameters"):
        make_env("descent3d", gravty=1.6)


def test_zem_controller_lands_nominal_cases():
    """The env is solvable: a classic closed-form guidance law lands it."""
    env = make_env("descent3d")
    spec = get_spec("descent3d")
    ctrl = CONTROLLERS["zem"](env)
    wins = 0
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, te, tr, _ = env.step(ctrl.predict(obs))
            done = te or tr
        wins += spec.outcome(env, te, tr) in spec.success
    assert wins >= 28
