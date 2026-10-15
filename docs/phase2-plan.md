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

- **Quick fix 1, glide-slope penalty instead of termination (2 seeds):** 0% success.
  The dive was gone, but the agent then either flew ~100 s and fell in at 40+ m/s (with
  phi = 0 at termination, crash and landing get the same total shaping) or, with phi
  kept at termination, hovered until the propellant ran out (touching down looked
  riskier than not).
- **Fix 2: penalty + graded touchdown reward + start curriculum + -150 for running
  out of propellant (4 seeds, 3M steps):** it works.
  - Every seed produced checkpoints at 77-94% success over 100 fixed seeds, with
    touchdown ~1.1 m/s vertical, ~0.2 m/s horizontal, ~5 m from the target, and no
    cone violations.
  - **But it's fuel-hungry:** 540-590 kg per landing (ZEM/ZEV: 352 kg). Almost all
    failures are running out of propellant. Training success swings between 90% and
    0% as small policy changes cross the budget, and all four final (3M-step) models
    had drifted to 100% running dry.
  - **Checkpoint selection by eval return** correlated only 0.12-0.28 with success. It
    happened to keep good checkpoints here, but it isn't selecting for success.
  - **Next:** raise `propellant_cost` (0.1 per kg makes 200 kg of waste cost only 20
    points, cheaper than the safety a slow descent buys), and rank checkpoints by
    success rate (with return as a tie-break) when the env reports it.

- **Fix 3: rank checkpoints by success rate, then return (done; pipeline-wide).**
  The runs' saved bests became 100% (95% CI 98-100) and 98% success over 200 fixed
  seeds, picked automatically. Ranking by return had picked checkpoints almost at random.
- **Fix 4: higher propellant price (0.3 and 0.5 per kg, 2 seeds each, 3M steps): no.**
  - **0.3:** propellant barely moved (526-531 kg), late training stayed unstable (2-16%
    of post-ramp evals at >= 80% success), and the final models still run dry.
  - **0.5:** never lands. It crashes deliberately, because at that price a landing's
    propellant (about -180) costs more than a crash (-100).
  - The propellant price only works together with a larger success reward. The late
    collapse looks like a separate optimisation-stability issue.
- **Next:** success +300 with propellant 0.3 per kg (a 360 kg landing nets about +190
  against -100 for a crash), plus learning-rate and clip-range decay to zero (the
  rl-zoo `lin_` schedules) against late-training drift.

- **Milestone 2 (viewer scene): done.**
  - Side view along the approach direction with a following, zooming camera; the
    glide-slope cone as shaded no-go wedges; a distance ruler; the landing zone with
    beacons.
  - Top-down inset with the exact 3D cone limit at the current altitude.
  - Telemetry, outcome banners and sounds.
  - 10 interpolated frames per 5 Hz decision, so real-time playback is smooth.
  - The shared art (sky, Earth, lander sprite) moved to `scene_art.py`; the
    LunarLander scene renders pixel-identically to before.
  - What it showed: the PPO champion spends 40 s in the last 100 m (ZEM/ZEV: 12 s),
    which is about half of its 173 kg excess propellant. The rest is a less efficient
    high-altitude phase.

- **3D viewer (branch `viewer-3d`):** the descent3d scene is now real 3D (moderngl
  offscreen + pygame HUD). The 2D side view hid cross-track motion, so the policy's
  3D control looked planar.
  - Optional Apollo LM from the user's Bluedog Design Bureau install, read at runtime
    with an independent `.mu` reader. It's CC BY-NC-SA, so it's never copied.
  - Low-sun shadows, normal-mapped craters (the physics ground stays flat), and four
    cameras.
  - The glide-slope limit shown as a ring at the lander's altitude.
  - About 6 ms per frame on the GPU, about 18 ms on the software renderer; 2D fallback.

- **Real terrain (branch `real-terrain`):** the 3D ground is the Apollo 11 landing site
  from LROC NAC imagery (0.5 m/px near the target, 2 m/px over 4.2 x 5.2 km, tiled
  beyond).
  - The sun direction is fitted from the 2 m DTM against the photo's brightness, so the
    lander's lighting and shadow match the imagery.
  - The landing target sits 25 m south of the real Apollo 11 descent stage, which is
    visible and labelled.
  - The physics ground stays flat (imagery only). If the real relief should ever enter
    the dynamics, the DTM is there; but it would also change the G-FOLD problem.

- **Uneven terrain, stage 1 (branch `uneven-terrain`):** descent3d's `terrain` option
  flies over a real LROC terrain model (Apollo 17: 5 m posts, 1.4 km of relief; Apollo 11:
  4 m, 116 m). Each episode draws a gentle target (≤ 6°) on the site.
  - Physics: contact with the real surface; a soft touchdown on more than 12° tips over.
  - Observation (+13): radar altitude, terrain profile to the target, clearances ahead.
  - Curriculum: the relief scales with difficulty, from flat to real.
  - Finding: with this start box (≥ 45° above the target) and a 15° glide slope, real
    terrain never gets in the way. Only 2% of Apollo 17 targets have terrain above the
    cone within 1 km, and terrain-blind ZEM/ZEV lands 200/200 there.
  - PPO: 3/4 seeds learn, reaching 53-63.5% success at Apollo 17 and 75% at Apollo 11
    (never trained on), with zero terrain-caused failures. The failures are the familiar
    glide-slope dips and running dry, at about 540 kg of propellant.
  - Propellant price: 0.3 per kg made 8/8 terrain seeds free-fall into the ground. At
    0.1 they learn.
  - Next: make terrain matter. Use shallow approach-phase starts, or let the target
    move (divert to a safer spot): stage 2.

## Decisions to confirm

1. **Glide-slope violation ends the episode.** That matches G-FOLD's hard constraint.
   The fallback, if RL struggles to learn, is a penalty in training only, with
   violations still counted as failures in eval.
2. **5 Hz control, 0.05 s physics.**
3. **Success thresholds:** 2 m/s vertical, 1 m/s horizontal, 10 m radius.
4. **The RL policy observes true mass in the nominal setting.** Under uncertainty
   (Phase 3) it gets a biased propellant estimate, the same information G-FOLD plans with.
5. **Vehicle numbers and the start-condition box** as in the tables above.
