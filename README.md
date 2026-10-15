# rl-lander

Autonomous powered-descent landing, learned (RL) vs. optimization-based guidance.

**Phase 1 (this repo state):** the infrastructure. PPO agents train headless on
Gymnasium's `LunarLander-v3` (continuous actions), several runs at once. A separate
viewer window continuously flies the best model found so far and swaps in better
ones as training produces them. Later phases plug a custom 3DOF powered-descent env
into the same training/eval/viewer code. **Next:** a 3D point-mass lunar descent env
posed as the G-FOLD problem; see the [Phase 2 plan](docs/phase2-plan.md).

![Viewer: the champion policy on final approach](docs/viewer.jpg)

```
configs/ppo_lunar.yaml      run config: env, seed, PPO hyperparameters (annotated)
src/lander/envs/            env registry: config name -> constructor, outcome classifier, scene
src/lander/envs/lunar_lander_scene.py   the viewer's LunarLander renderer (lander, terrain, effects)
src/lander/train.py         one headless training run          (lander-train)
src/lander/launch.py        N runs as parallel processes        (lander-launch)
src/lander/champion.py      per-run best + global champion
src/lander/checkpoints.py   save/load policies (+ observation-normalisation stats)
src/lander/evaluate.py      fixed-seed evaluation with success rates and CIs (lander-eval)
src/lander/viewer.py        live viewer, hot-reloads champion   (lander-viewer)
src/lander/hud.py           viewer overlay: model card, telemetry, banners (env-agnostic)
src/lander/sound.py         procedural sound effects for the viewer
Makefile                    shortcuts: `make help`
runs/                       outputs (gitignored)
```

## Setup

Developed and tested on **WSL2 (Ubuntu 24.04) on Windows 11**. The viewer window
appears on the Windows desktop through WSLg, with no X server to configure.

Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`. Python 3.11.

```bash
# once: install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
# in the repo: creates .venv with Python 3.11 and the exact locked versions
uv sync
```

Notes on the usual pain points:

- **Box2D/swig:** not needed anymore. Gymnasium ≥ 1.4 depends on the `box2d` package,
  which ships prebuilt wheels (Linux and Windows), instead of `box2d-py`, which had
  to be compiled with swig. If an older guide tells you to install swig, ignore it.
- **PyTorch** comes from the CPU-only index (see `[tool.uv.sources]` in
  `pyproject.toml`), which is ~200 MB instead of ~3 GB of CUDA libraries. A 64x64 MLP
  trains no faster on a GPU, since data transfer dominates.

**Native Windows (untested):** the lock file includes win_amd64 wheels for every
dependency, so this should work without a compiler. Use PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync            # uv downloads Python 3.11 itself; your system Python 3.14 is not used
uv run lander-viewer
```

## Usage

The quickest route is the Makefile. Run `make help` to list targets; override variables
inline, e.g. `make launch SEEDS="0 1"` or `make replay MODEL=runs/<run>/best/step_000420000.zip`.

```bash
make setup         # uv sync
make launch        # 4 seeds in parallel (SEEDS="0 1 2 3")
make viewer        # in a second terminal
make tensorboard   # in a third
```

The underlying commands, run from the repo root (prefix with `uv run`, or activate
`.venv` first):

```bash
# One run (~3 min for 1M steps)
uv run lander-train configs/ppo_lunar.yaml --seed 0
uv run lander-train configs/ppo_lunar.yaml --total-timesteps 5e4   # ~15 s smoke test (PPO rounds up to whole 16k-step rollouts)

# Several runs in parallel: every config x every seed, one process each
uv run lander-launch configs/ppo_lunar.yaml --seeds 0 1 2 3

# Viewer (separate terminal). Start it before, during or after training.
uv run lander-viewer
uv run lander-viewer --model runs/<run>/best/step_000420000.zip   # replay one checkpoint
uv run lander-viewer --fps 150 --scale 0.8                        # faster / smaller window
uv run lander-viewer --plain                                      # the env's own basic rendering
uv run lander-viewer --mute                                       # start silent (M toggles sound)

# Evaluate on 200 fixed seeds: the champion, or any checkpoints side by side
uv run lander-eval
uv run lander-eval runs/a/best/step_X.zip runs/b/best/step_Y.zip --episodes 500 --out results.json

# TensorBoard: one curve per run; open http://localhost:6006 in your Windows browser
uv run tensorboard --logdir runs
```

To compare hyperparameters, copy the config, edit it, and pass both files to
`lander-launch`. Run names are `<config-file>_s<seed>_<timestamp>`.

The champion is global to a runs directory, including runs from earlier
sessions. For a fresh competition (or a different env, whose rewards aren't
comparable), use another directory: `--runs-dir runs/exp2` on both launcher and viewer.

### What the viewer shows

- **Scene:** a custom renderer draws the lander, terrain and landing pad from the
  live physics state, so the picture is exactly what the policy experiences.
  - The legs move on their sprung joints, and the pad beacons blink red, turning
    green after a landing on the pad.
  - The main-engine plume scales with throttle, the side RCS jets fire on the side
    actually firing, and engine dust sprays off the ground near the surface.
  - A crash ends in an explosion and a scorched wreck.
- **Model card** (top left): the run, training step and eval score (mean ± std over
  the eval episodes). When a better champion appears mid-episode, the card says so,
  and a toast confirms the swap after the episode ends.
- **Telemetry** (top right): altitude, vertical/horizontal speed and tilt, coloured
  green, amber or red near the ground; main-engine throttle, side-thruster state and
  leg contact. Values are in the simulator's world units.
- **Episode strip** (bottom): episode number, sim time, running reward, and a tally
  of outcomes so far.
- **Outcome banner** after each episode: `TOUCHDOWN` (on or off the pad), `CRASH`,
  `LOST` (flew off-screen) or `TIMEOUT` (still hovering after 1000 steps).
- **Sound:** an engine rumble that follows the throttle, a hiss from the side
  thrusters, a touchdown thud (with a chime for an on-pad landing), an explosion on a
  crash, a falling tone when the lander is lost, and a chime when a new champion
  loads. Sounds are panned to the lander's position. Press `M` to toggle;
  `--mute` starts silent, and `--volume 0..1` sets the level. Everything is
  synthesised with numpy at startup (`src/lander/sound.py`), so there are no audio
  files. Without an audio device the viewer just runs silent. Under WSL2, sound goes
  to Windows through WSLg automatically.

The window is resizable and the picture scales to fit. Close it or press `q`/`Esc`
to quit. The console also logs each load and each episode's outcome.

### Evaluating

`lander-eval` is the measuring stick; the eval score in `champion.json` and the viewer
are not. Episode *i* always starts from seed `--seed + i`, so every model sees exactly
the same start conditions and comparisons are paired. It reports:

- **success rate** with a 95% confidence interval (a Wilson interval, which stays sensible
  near 0% and 100%). "Success" is the env's definition of mission success: for
  LunarLander that's `landed on pad`, not a reward threshold;
- the count of every outcome, and mean return with a 95% confidence interval;
- **landing metrics on the successful episodes**: distance from the pad centre,
  touchdown speed (at first leg contact), tilt, engine use (a fuel proxy) and duration.

`--out file.json` also saves every episode, for your own analysis. 200 episodes take
about 10 s.

### Observation normalisation

Set `normalize: true` in a config (or `normalize: {obs: true, reward: true}`) to wrap
training in SB3's `VecNormalize`. It rescales every observation by a running estimate of
its mean and standard deviation, so inputs with very different physical scales reach the
network at a similar size. The network then only understands normalised inputs, so the
statistics are part of the policy: they're saved beside every checkpoint as
`step_*.vecnormalize.pkl`, and the viewer and `lander-eval` load them automatically.
LunarLander's observations are already roughly unit-scale, so the provided config leaves
normalisation off. Phase 2's physical-unit env needs it. (In a test, one seed with it
on solved LunarLander at 170k steps instead of 300k.)

### Outputs

```
runs/champion.json                 global best: run, model path, step, mean/std, env, timestamp
runs/<run>/config.yaml             the exact config used (incl. seed)
runs/<run>/train.log               stdout (when started by lander-launch)
runs/<run>/best.json               this run's best (same format as champion.json)
runs/<run>/best/step_*.zip         every time the run improved: an immutable checkpoint
runs/<run>/best/step_*.vecnormalize.pkl   its normalisation stats (only with `normalize`)
runs/<run>/final_model.zip         the model at the end (or at Ctrl+C)
runs/<run>/evaluations.npz         all eval results (SB3 EvalCallback format)
runs/<run>/tb/                     TensorBoard events
```

## What to expect

Measured on this machine (20-core CPU) with the provided config, 4 seeds launched
together:

- **Reliability of the final champion** (`lander-eval`, 200 fixed seeds): 200/200 on the
  pad (95% CI 98-100%), touchdown at about 1 world unit/s, about 0.4 from the pad centre.
- **Speed:** ~6,000 env steps/s per run, so 1M steps takes **~3 min**. 4 parallel
  runs finished in 168-176 s each, so they barely slow each other down (one torch
  thread each).
- **Learning curve:** mean eval reward is negative (crashing) until ~100-150k steps.
  First landings come around 150-200k. The runs were **solved (≥ 200) at 250k, 300k,
  330k and 460k steps** for seeds 1, 0, 3 and 2. By 1M steps the best eval means
  were 267-288.
- **"Solved" is not "reliable".** Two checkpoints, each re-evaluated on 100 fresh
  episodes:

  | checkpoint | eval score when selected | 100-episode mean | on pad | off pad | crashed / off-screen |
  |---|---|---|---|---|---|
  | seed 1 @ 250k (first "solved") | 234.5 | 135.7 ± 152 | 34 | 33 | 33 |
  | seed 0 @ 940k (final champion) | 287.8 | 274.2 ± 21 | **100** | 0 | 0 |

  The first "solved" checkpoint crashes a third of the time. After ~1M steps the
  champion lands on the pad every time. Both checkpoints score lower on fresh
  episodes than when they were selected (see *winner's curse* below).

## RL notes (why things are the way they are)

**Reward scale.** LunarLander gives a shaped reward every step: points for moving
toward the pad, slowing down and staying upright; +10 per leg touching down; a
small cost per frame of engine use. On top of that comes a terminal +100 for coming
to rest or -100 for crashing. A clean landing totals roughly 200-300. Random flying
scores about -200.

**What "solved" means.** Gymnasium's bar for this env is an average return of
**200**. That is a statement about reward, not about pad accuracy. The +100 rest bonus
is paid wherever the lander comes to rest. The distance-to-pad term in the shaping is
mild, so a soft touchdown just outside the flags can still score 200+. The viewer
therefore reports *on pad* vs *off pad* separately. For Phase 2: reward ≠ mission
success, so measure the outcome you actually care about (touchdown position error,
velocity, tilt) directly.

**Why PPO.** It's the standard robust baseline for continuous control. It learns
from fresh batches of experience and clips each update, so one bad batch can't
wreck the policy. That makes it forgiving of hyperparameters, and it parallelises
across envs trivially. Off-policy methods (SAC, TD3) are more sample-efficient,
which matters when env steps are expensive. They aren't here. The hyperparameters
are the tuned set from [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
and are explained inline in `configs/ppo_lunar.yaml`.

**Why "best" = deterministic eval, not training reward.** During training the policy
samples actions from a Gaussian (exploration noise), and the policy changes while
the episodes are being collected. Training-episode returns therefore measure a
noisy, moving target. Every 10k steps, `EvalCallback` instead runs 10 fresh
episodes with the policy's *mean* action (deterministic). The checkpoint with the
highest mean of those is the run's best.

**The winner's curse.** The mean of 10 episodes is still noisy (std is often
50-150, so the mean is uncertain by ±20-50). Picking the maximum over ~100
evaluations favours checkpoints that got lucky, so the champion's displayed score is
optimistic. Expect it to do somewhat worse over many episodes in the viewer. More eval
episodes (`eval.n_episodes`) shrink this effect at the cost of eval time.

**Why `vec_env: dummy`.** `n_envs: 16` envs are stepped in lockstep and their
experience is batched. For an env this cheap, stepping all 16 in one process
(`DummyVecEnv`) beat one process per env (`SubprocVecEnv`). Inter-process
communication cost more than the parallelism saved: 28 s vs 32 s for 200k steps.
Switch to `subproc` if a future env has an expensive `step()`.

## Design notes

- **Atomic files.** Every write goes to a temp file in the same directory, then
  `os.replace`s it into place. Checkpoints are never overwritten (each new best gets
  a new `step_*.zip`), so `champion.json` is the only file ever replaced. The viewer
  can't read a half-written file.
- **Champion races.** Runs finishing an eval at the same time update
  `champion.json` under a file lock (`filelock`), so the read-compare-write can't
  interleave.
- **Viewer reloads between episodes only.** It polls `champion.json` every 2 s. A
  failed load is retried, and the previous model keeps flying meanwhile.
- **The viewer draws into its own window** instead of using the env's
  `render_mode="human"`. Each env can register a *scene* (`EnvSpec.scene`) that draws
  from its physics state. Envs without one fall back to their own `rgb_array` frames.
  The HUD on top is env-agnostic; scenes feed it telemetry as a list of `Gauge`s.
  The scene is purely visual and never affects the simulation.
- **Adding an env (Phase 2):** write `src/lander/envs/<name>.py` with
  `make(render_mode=None, **kwargs)` and `outcome(env, terminated, truncated)`, add
  an `EnvSpec` to `REGISTRY` in `src/lander/envs/__init__.py`, and set `env.name` in
  a config. Training, eval, champion tracking and the viewer need no changes. A
  custom scene is optional: copy the method set of `LunarScene` (`reset`, `update`,
  `on_outcome`, `draw`, `gauges`, `banner`, `outcome_tone`, and for sound
  `sound_levels`, `outcome_sound`).
