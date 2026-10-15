# Phase 2 plan: 3D point-mass lunar powered descent

**Status:** proposal, awaiting approval. Nothing here is built yet.

## Goal

A Gymnasium env for the terminal phase of a lunar landing, posed as **exactly the problem
G-FOLD solves**. RL and convex-optimization guidance can then be compared on identical
dynamics, constraints and start conditions. Phase 2 covers the nominal problem
(env, viewer, G-FOLD baseline, RL training). The uncertainty study is Phase 3; the env
gets its uncertainty knobs now, switched off.

References:
- Açıkmeşe & Ploen, *Convex Programming Approach to Powered Descent Guidance for Mars
  Landing*, JGCD 2007: the problem, lossless convexification, and the SOCP.
- Blackmore, Açıkmeşe & Scharf, *Minimum-Landing-Error Powered-Descent Guidance for Mars
  Landing Using Convex Optimization*, JGCD 2010: what to do when the target can't be reached.
- Açıkmeşe, Carson & Blackmore, *Lossless Convexification of Nonconvex Control Bound and
  Pointing Constraints of the Soft Landing Optimal Control Problem*, IEEE TCST 2013:
  conditions under which the thrust-pointing constraint stays lossless.

## The model

Target-centred frame: `z` is up (altitude), `x` and `y` are horizontal, and the ground
is the plane `z = 0`.

```
r'' = g + T / m          g = (0, 0, -1.62) m/s²        (Moon)
m'  = -|T| / (Isp g0)    Isp = 311 s, g0 = 9.80665 m/s²
```

- **3DOF point mass:** attitude is assumed to follow the commanded thrust direction
  instantly, the standard G-FOLD assumption. Rotation dynamics arrive in the 6DOF phase.
- **Vehicle**, loosely Apollo LM descent stage near "high gate" (all values configurable):

  | parameter | value | note |
  |---|---|---|
  | initial mass `m0` | 8,000 kg | |
  | usable propellant | 600 kg | `m_min = 7,400 kg`; about 2.3 min of hover |
  | thrust range | 4.5–27.0 kN | LM descent engine throttle range of 10–60% of 45 kN; thrust-to-weight 0.35–2.1 |
  | max thrust tilt from vertical | 45° | thrust-pointing constraint |
  | glide slope | 15° | must stay above a cone of 15° elevation around the target |

- **The engine can't be switched off** (`|T| ≥ 4.5 kN`). This non-convex lower bound is
  the constraint G-FOLD's lossless convexification exists for, and the RL agent faces it too.
- **Integration:** RK4 at 0.05 s, with thrust held constant across each 0.2 s control
  step (5 Hz control). A 60 s descent is then about 300 decisions, similar to
  LunarLander episode lengths.

## Env interface

- **Observation** (physical units; training uses `normalize: true`, built in Phase 1.5):
  `r` (3), `v` (3), propellant remaining (1) = 7 dims.
- **Action** `a ∈ [-1, 1]³`, mapped smoothly into the feasible set, so the thrust bounds
  and pointing constraint can't be violated by construction:
  - `a0` sets the thrust magnitude, from 4.5 kN (at `a0 = -1`) to 27 kN (at `a0 = +1`).
  - `(a1, a2)`, clipped to the unit disk, sets the tilt: radius × 45°, with azimuth
    `atan2(a2, a1)`.
- **Start conditions**, drawn per episode from the reset seed (so `lander-eval` episodes
  are paired across controllers):
  - altitude 1,000–2,000 m
  - horizontal offset 0–1,000 m, random direction
  - descent rate 15–40 m/s
  - horizontal speed 0–30 m/s, random direction

  Some draws may be infeasible with 600 kg of propellant. The G-FOLD solver identifies
  those, and the eval reports them separately rather than counting them against RL.
- **Episode end and outcomes:**

  | outcome | condition |
  |---|---|
  | `landed on target` | touchdown at ≤ 2 m/s vertical, ≤ 1 m/s horizontal, within 10 m of target (**success**) |
  | `landed off target` | soft touchdown, more than 10 m from target |
  | `crashed` | touchdown above the speed limits |
  | `glide-slope violation` | dropped below the approach cone (ends the episode: it's a hard constraint for G-FOLD, so it is one here too) |
  | `out of propellant` | mass reached `m_min` while still airborne |
  | `timeout` | 150 s of sim time |

  The speed limits are in line with Apollo LM touchdown limits (10 ft/s vertical,
  4 ft/s horizontal), slightly tightened.
- **Reward:**
  - **Shaping:** each step pays `F = γΦ(s') − Φ(s)`, with `Φ = −(|r|/1000 m + |v|/50 m/s)`
    (it rewards getting closer and slower). This is potential-based shaping, which is proven not to
    change which policy is optimal (Ng, Harada & Russell 1999), unlike LunarLander's
    ad-hoc shaping.
  - **Propellant cost:** `−c · Δm` per step. This aligns RL's objective with G-FOLD's
    minimum-fuel objective.
  - **Terminal:** +100 for success; −100 for crash, glide-slope violation, running out
    of propellant, or timeout; a smaller negative for landing off target.
- **Landing metrics** for `lander-eval`: touchdown position error, vertical and horizontal
  touchdown speed, propellant used, minimum glide-slope margin, and flight time.
- **Uncertainty knobs** (Phase 3; config, all off by default), sampled per episode from
  the seed:
  - true vs nominal mass, Isp and gravity
  - thrust magnitude scale error and thrust misalignment angle
  - sensor noise on `r` and `v`
  - first-order engine lag
  - widened start conditions

## Work plan

1. **Env + tests.**
   - Write `src/lander/envs/descent3d.py` and register it in `envs/__init__.py`.
   - Add pytest with physics checks:
     - free fall matches the closed form
     - hovering at `|T| = m g` holds velocity at zero
     - mass drops at `|T|/(Isp g0)`
     - action mapping never leaves the feasible set
     - Gymnasium's `check_env` passes
   - Add a **hand-written guidance law** (e.g. Apollo-style quadratic guidance) that
     lands the nominal cases. It proves the env is solvable before any learning.
2. **Viewer scene.**
   - Side view following the lander, with a top-down inset showing the target, the
     glide-slope cone and the horizontal track.
   - Trajectory trail.
   - Telemetry: altitude, speeds, propellant, thrust and tilt.
   - Reuses the HUD and the lunar backdrop from Phase 1.
3. **G-FOLD baseline** (`cvxpy`, Clarabel solver; new deps `cvxpy`, `pytest`).
   - Lossless convexification:
     - slack variable `Γ ≥ |T|`
     - change of variables `u = T/m`, `σ = Γ/m`, `z = ln m`
     - Taylor-approximated thrust bounds in `z`
   - Search over time of flight, with the paper's bounds on it.
   - Check it against the paper's Mars example before using it on the Moon.
   - Run it as a controller in two modes:
     - **open-loop:** solve once, then replay the plan
     - **MPC:** re-solve every ~1 s from the current state, so it can correct
   - Same `predict(obs)` interface as an RL policy, so `lander-eval --controller gfold`
     works. Record solve times; they matter for real-time use.
4. **RL training:** PPO with normalization on the existing pipeline, plus SAC as a
   second algorithm (it's often stronger on smooth control problems).
   - **Targets:** at least 95% success on feasible nominal starts, with propellant use
     reported against the G-FOLD optimum for the same seed.
5. **Nominal comparison report:** paired `lander-eval` over the same seeds, covering
   success rate with confidence intervals, propellant gap to G-FOLD, touchdown errors,
   and compute per decision.

Phase 3 (not in this plan) then sweeps the uncertainty knobs and compares how both
controllers degrade.

## Findings so far

- **Milestone 1 (done):**
  - The env passes `check_env` and a 22-test physics suite.
  - ZEM/ZEV guidance lands 200/200 nominal seeds (position error < 0.05 m, 352 kg
    propellant mean). The env is solvable.
- **Untuned PPO, 1M steps: 0% success.** It converged to "dive and violate the glide slope
  after ~30 s, burning ~50 kg". With the potential zeroed at terminal states, the total
  shaping is the same for every ending, so diving costs only the −100 terminal reward
  and saves propellant. A landing pays more, but exploration never finds one, because the
  cone ends exploratory episodes early. Candidate fixes for Milestone 4, roughly in order:
  1. **The planned fallback:** in training, a glide-slope violation becomes a penalty
     proportional to how deep and how long it is, instead of ending the episode.
     Eval keeps counting it as a failure.
  2. **A start-condition curriculum:** begin close to and above the target, then widen
     to the full box.
  3. **Rebalance the terminal rewards** (a bigger failure penalty, or a success bonus that
     depends on touchdown accuracy).
  4. **SAC**, whose replay buffer and entropy-driven exploration often cope better with this.

## Decisions to confirm

1. **Glide-slope violation ends the episode.** That matches G-FOLD's hard constraint.
   The fallback, if RL struggles to learn, is a penalty in training only, with
   violations still counted as failures in eval.
2. **5 Hz control, 0.05 s physics.**
3. **Success thresholds:** 2 m/s vertical, 1 m/s horizontal, 10 m radius.
4. **The RL policy observes true mass in the nominal setting.** Under uncertainty
   (Phase 3) it gets a biased propellant estimate, the same information G-FOLD plans with.
5. **Vehicle numbers and the start-condition box** as in the tables above.
