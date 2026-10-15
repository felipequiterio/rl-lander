"""Train a single run from a YAML config, headless.

    lander-train configs/ppo_lunar.yaml --seed 1
    lander-train configs/ppo_lunar.yaml --total-timesteps 50000   # quick smoke test

Why PPO: it's the robust default for continuous control. It is on-policy (learns
only from fresh experience), and it clips each update so one noisy batch can't
wreck the policy. That makes it forgiving of hyperparameters and stable across
seeds. Off-policy methods like SAC/TD3 are more sample-efficient, but here env
steps are cheap and PPO parallelises trivially across envs.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lander.champion import ChampionCallback, atomic_save_model
from lander.config import ALGOS, load_config
from lander.envs import get_spec, make_env


def default_run_name(config_path: Path, seed: int) -> str:
    return f"{config_path.stem}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"


def train(cfg: dict, run_name: str, runs_dir: Path) -> None:
    run_dir = runs_dir / run_name
    if (run_dir / "config.yaml").exists():
        sys.exit(f"{run_dir} already holds a run; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)  # may already exist: lander-launch puts train.log there
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    torch.set_num_threads(cfg["torch_threads"])
    env_name, env_kwargs = cfg["env"]["name"], cfg["env"]["kwargs"]
    env_fn = partial(make_env, env_name, **env_kwargs)
    seed = cfg["seed"]

    # make_vec_env wraps each env in a Monitor, which records episode returns for
    # the rollout/ep_rew_mean curve in TensorBoard.
    vec_cls = SubprocVecEnv if cfg["vec_env"] == "subproc" else DummyVecEnv
    train_env = make_vec_env(env_fn, n_envs=cfg["n_envs"], seed=seed, vec_env_cls=vec_cls)
    # A separate env for evaluation, seeded differently so it never replays the
    # training episodes. Eval runs in-process: it's only a few envs.
    eval_cfg = cfg["eval"]
    eval_env = make_vec_env(env_fn, n_envs=eval_cfg["n_envs"], seed=seed + 10_000, vec_env_cls=DummyVecEnv)

    algo = cfg["algo"]
    model = ALGOS[algo]("MlpPolicy", train_env, seed=seed, device="cpu", verbose=0, **cfg["ppo"])
    model.set_logger(configure(str(run_dir / "tb"), ["tensorboard"]))

    spec = get_spec(env_name)
    champion_cb = ChampionCallback(
        run_dir=run_dir,
        runs_dir=runs_dir,
        run_name=run_name,
        env_cfg=cfg["env"],
        algo=algo,
        solved_threshold=spec.solved_threshold,
    )
    # "Best" is judged on deterministic eval episodes (the policy's mean action,
    # no exploration noise), not on training-episode returns. Those are noisy
    # because the policy samples random actions while learning, and they average
    # over a policy that kept changing during the rollout.
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=champion_cb,
        n_eval_episodes=eval_cfg["n_episodes"],
        # EvalCallback counts calls, and each call advances all n_envs envs one step.
        eval_freq=max(eval_cfg["freq"] // cfg["n_envs"], 1),
        log_path=str(run_dir),
        best_model_save_path=None,  # ChampionCallback does the (atomic) saving
        deterministic=True,
        verbose=1,
    )

    print(f"[{run_name}] training {cfg['total_timesteps']:,} steps -> {run_dir}", flush=True)
    t0 = time.time()
    try:
        model.learn(total_timesteps=cfg["total_timesteps"], callback=eval_cb)
    except KeyboardInterrupt:
        print(f"[{run_name}] interrupted, saving final model", flush=True)
    finally:
        atomic_save_model(model, run_dir / "final_model.zip")
        train_env.close()
        eval_env.close()
    print(
        f"[{run_name}] done in {time.time() - t0:.0f}s, {model.num_timesteps:,} steps, "
        f"best eval mean {eval_cb.best_mean_reward:.1f}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--seed", type=int, help="override config seed")
    p.add_argument("--run-name", help="default: <config>_s<seed>_<timestamp>")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--total-timesteps", type=float, help="override config (e.g. 5e4 for a smoke test)")
    p.add_argument("--n-envs", type=int, help="override config")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)
    if args.n_envs is not None:
        cfg["n_envs"] = args.n_envs
    run_name = args.run_name or default_run_name(args.config, cfg["seed"])
    train(cfg, run_name, args.runs_dir.resolve())


if __name__ == "__main__":
    main()
