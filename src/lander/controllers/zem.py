"""ZEM/ZEV guidance: a classic closed-form landing law, used as a sanity baseline.

At each step, assume a time-to-go t and compute the zero-effort miss and velocity
(where the vehicle would end up with no thrust, gravity only):

    ZEM = r_f - (r + v t + g t^2 / 2)
    ZEV = v_f - (v + g t)

The control acceleration that minimises the integral of |a|^2 over the remaining
time is a = 6 ZEM / t^2 - 2 ZEV / t, at an energy cost

    J(t) = 12 |ZEM|^2 / t^3 - 12 ZEM.ZEV / t^2 + 4 |ZEV|^2 / t.

We pick the t that minimises J (too short: harsh braking; too long: fighting
gravity), command T = m a, and project it onto the feasible thrust set. It
ignores thrust bounds and the glide slope (they're just clipped), and it's
energy-optimal, not fuel-optimal. That makes it a decent but beatable reference,
and a proof that the env is solvable.
"""

from __future__ import annotations

import numpy as np

from lander.envs.descent3d import DescentConfig, thrust_to_action


class ZemZevController:
    def __init__(self, cfg: DescentConfig, v_final: tuple[float, float, float] = (0.0, 0.0, -0.5)):
        self.cfg = cfg
        self.r_f = np.zeros(3)
        self.v_f = np.asarray(v_final)  # touch down gently rather than hover at z = 0
        self.t_grid = np.linspace(0.5, cfg.max_time, 600)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        r, v, propellant = obs[:3].astype(float), obs[3:6].astype(float), float(obs[6])
        g = self.cfg.g
        t = self.t_grid[:, None]
        zem = self.r_f - (r + v * t + 0.5 * g * t**2)
        zev = self.v_f - (v + g * t)
        t1 = t[:, 0]
        cost = 12 * (zem**2).sum(1) / t1**3 - 12 * (zem * zev).sum(1) / t1**2 + 4 * (zev**2).sum(1) / t1
        i = int(np.argmin(cost))
        accel = 6 * zem[i] / t1[i] ** 2 - 2 * zev[i] / t1[i]
        mass = self.cfg.m_min + propellant
        return thrust_to_action(mass * accel, self.cfg).astype(np.float32)
