# rl-lander

Autonomous powered-descent landing, learned (RL) vs. optimization-based guidance.

**Phase 1 (this repo state):** the infrastructure. PPO agents train headless on
Gymnasium's `LunarLander-v3` (continuous actions), several runs at once. A separate
viewer window continuously flies the best model found so far and swaps in better
ones as training produces them. Later phases plug a custom 3DOF powered-descent env
into the same training/eval/viewer code.

```
configs/ppo_lunar.yaml      run config: env, seed, PPO hyperparameters (annotated)
src/lander/envs/            env registry: config name -> constructor + outcome classifier
src/lander/train.py         one headless training run          (lander-train)
src/lander/launch.py        N runs as parallel processes        (lander-launch)
src/lander/champion.py      per-run best + global champion, atomic writes
src/lander/viewer.py        live viewer, hot-reloads champion   (lander-viewer)
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

All commands are run from the repo root. Prefix them with `uv run`, or activate
`.venv` first.

```bash
# One run (~3 min for 1M steps)
uv run lander-train configs/ppo_lunar.yaml --seed 0
uv run lander-train configs/ppo_lunar.yaml --total-timesteps 5e4   # ~15 s smoke test (PPO rounds up to whole 16k-step rollouts)

# Several runs in parallel: every config x every seed, one process each
uv run lander-launch configs/ppo_lunar.yaml --seeds 0 1 2 3

# Viewer (separate terminal). Start it before, during or after training.
uv run lander-viewer
uv run lander-viewer --model runs/<run>/best/step_000420000.zip   # replay one checkpoint
uv run lander-viewer --fps 150 --scale 1.5                        # faster / bigger

# TensorBoard: one curve per run; open http://localhost:6006 in your Windows browser
uv run tensorboard --logdir runs
```

To compare hyperparameters, copy the config, edit it, and pass both files to
`lander-launch`. Run names are `<config-file>_s<seed>_<timestamp>`.

The champion is global to a runs directory, including runs from earlier
sessions. For a fresh competition (or a different env, whose rewards aren't
comparable), use another directory: `--runs-dir runs/exp2` on both launcher and viewer.

### What the viewer shows

The overlay shows the model's run, training step and eval score (mean ± std over
the eval episodes), plus the current episode's reward. After each episode it shows the
outcome: `landed on pad`, `landed off pad`, `crashed`, `flew off-screen` or
`timeout (hovering)`. The console prints a running tally of outcomes. When a new
champion appears mid-episode, the overlay says so; the swap happens after the
episode ends. Close the window or press `q`/`Esc` to quit.

### Outputs

```
runs/champion.json                 global best: run, model path, step, mean/std, env, timestamp
runs/<run>/config.yaml             the exact config used (incl. seed)
runs/<run>/train.log               stdout (when started by lander-launch)
runs/<run>/best.json               this run's best (same format as champion.json)
runs/<run>/best/step_*.zip         every time the run improved: an immutable checkpoint
runs/<run>/final_model.zip         the model at the end (or at Ctrl+C)
runs/<run>/evaluations.npz         all eval results (SB3 EvalCallback format)
runs/<run>/tb/                     TensorBoard events
```

## What to expect

Measured on this machine (20-core CPU) with the provided config, 4 seeds launched
together:

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
- **Viewer renders `rgb_array` frames into its own window** instead of using the
  env's `render_mode="human"`. That's what allows the overlay, and a new env only
  needs to implement `rgb_array` rendering.
- **Adding an env (Phase 2):** write `src/lander/envs/<name>.py` with
  `make(render_mode=None, **kwargs)` and `outcome(env, terminated, truncated)`, add
  an `EnvSpec` to `REGISTRY` in `src/lander/envs/__init__.py`, and set `env.name` in
  a config. Training, eval, champion tracking and the viewer need no changes.
