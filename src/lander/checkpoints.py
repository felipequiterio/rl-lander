"""Saving and loading policies, including observation-normalisation statistics.

With ``normalize`` enabled in the config, training wraps the envs in SB3's
``VecNormalize``. That rescales each observation using running estimates of
its mean and variance, so that inputs with very different physical scales
(altitude in hundreds of metres next to a velocity of a few m/s) reach the
network at a similar magnitude. The network only ever sees normalised inputs,
so the statistics are part of the policy: they're saved next to every
checkpoint (``step_X.zip`` + ``step_X.vecnormalize.pkl``) and loaded with it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from lander.atomic import atomic_write_with, read_json
from lander.config import ALGOS


def stats_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + ".vecnormalize.pkl")


def atomic_save_checkpoint(model: BaseAlgorithm, path: Path) -> None:
    """Save model (and its normalisation stats, if any), each file atomically.
    The stats go first, so whoever sees the model file also finds its stats."""
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        atomic_write_with(stats_path(path), vec_norm.save)
    # Keep a .zip suffix on the temp file, otherwise SB3 appends one to the name.
    atomic_write_with(path, model.save, suffix=".tmp.zip")


class Policy:
    """A loaded checkpoint: model + optional observation normalisation."""

    def __init__(self, model: BaseAlgorithm, obs_norm: VecNormalize | None):
        self.model = model
        self.obs_norm = obs_norm

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize_obs(obs)
        return self.model.predict(obs, deterministic=deterministic)[0]


def _run_config(model_path: Path) -> dict[str, Any] | None:
    for d in (model_path.parent, model_path.parent.parent):
        if (d / "config.yaml").exists():
            with open(d / "config.yaml") as f:
                return yaml.safe_load(f)
    return None


def load_policy(model_path: Path, algo: str = "PPO") -> Policy:
    model = ALGOS[algo].load(model_path, device="cpu")
    obs_norm = None
    cfg = _run_config(model_path)
    if cfg and cfg.get("normalize", {}).get("obs") and not stats_path(model_path).exists():
        # Without its stats a normalised policy still runs, just badly. Fail loudly instead.
        raise FileNotFoundError(f"{model_path.name} was trained with observation normalisation but {stats_path(model_path).name} is missing")
    if stats_path(model_path).exists():
        # A VecNormalize pickles without its env; unpickled, it can still
        # normalise observations (it just can't step anything).
        with open(stats_path(model_path), "rb") as f:
            obs_norm = pickle.load(f)
        obs_norm.training = False
    return Policy(model, obs_norm)


def record_for_checkpoint(model_path: Path, env_name: str | None = None) -> dict[str, Any]:
    """Build a champion-like record (run, env, algo, eval stats) for any checkpoint.

    Looks for the run's config.yaml (written by train.py) next to or above the
    checkpoint to find out which env it was trained on.
    """
    record: dict[str, Any] = {
        "run_name": model_path.parent.name,
        "model_path": model_path.as_posix(),  # absolute, so resolve_model_path() returns it as-is
        "timestep": None,
        "mean_reward": None,
        "algo": "PPO",
    }
    for d in (model_path.parent, model_path.parent.parent):
        if (d / "config.yaml").exists():
            cfg = _run_config(model_path)
            record.update(run_name=d.name, env=cfg["env"], algo=cfg.get("algo", "PPO"))
            run_best = read_json(d / "best.json")
            if run_best and Path(run_best["model_path"]).name == model_path.name:
                record.update(timestep=run_best["timestep"], mean_reward=run_best["mean_reward"], std_reward=run_best["std_reward"])
            break
    if env_name is not None or "env" not in record:
        record["env"] = {"name": env_name or "lunar_lander", "kwargs": {}}
    if record["timestep"] is None and model_path.stem.startswith("step_"):
        record["timestep"] = int(model_path.stem.removeprefix("step_"))
    return record
