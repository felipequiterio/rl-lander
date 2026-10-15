"""Run config loading. See configs/ppo_lunar.yaml for the annotated format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

# Algorithm name (config "algo" / champion.json "algo") -> SB3 class.
ALGOS: dict[str, type[BaseAlgorithm]] = {"PPO": PPO}

DEFAULTS: dict[str, Any] = {
    "algo": "PPO",
    "seed": 0,
    "n_envs": 16,
    "vec_env": "dummy",  # "dummy" (all envs in-process) or "subproc" (one process per env)
    "torch_threads": 1,
    "eval": {"freq": 10_000, "n_episodes": 10, "n_envs": 5},
    # Running mean/std normalisation of observations (and optionally rewards); see checkpoints.py.
    "normalize": {"obs": False, "reward": False, "clip_obs": 10.0},
    "ppo": {},
}

# PyYAML parses "3e-4" as a string (YAML 1.1 wants "3.0e-4"). Coerce the
# numeric hyperparameters most likely to be written that way.
_FLOAT_KEYS = {"learning_rate", "ent_coef", "vf_coef", "max_grad_norm", "gamma", "gae_lambda", "clip_range"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = {**DEFAULTS, **user}
    cfg["eval"] = {**DEFAULTS["eval"], **user.get("eval", {})}
    norm = user.get("normalize", {})
    if isinstance(norm, bool):  # `normalize: true` is shorthand for observations only
        norm = {"obs": norm}
    cfg["normalize"] = {**DEFAULTS["normalize"], **norm}
    cfg["ppo"] = dict(user.get("ppo") or {})
    for k in _FLOAT_KEYS & cfg["ppo"].keys():
        cfg["ppo"][k] = float(cfg["ppo"][k])
    cfg["total_timesteps"] = int(float(cfg["total_timesteps"]))
    if "env" not in cfg or "name" not in cfg["env"]:
        raise ValueError(f"{path}: config needs env.name")
    cfg["env"].setdefault("kwargs", {})
    if cfg["algo"] not in ALGOS:
        raise ValueError(f"{path}: unknown algo {cfg['algo']!r}, known: {sorted(ALGOS)}")
    return cfg
