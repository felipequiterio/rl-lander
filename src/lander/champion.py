"""Per-run best checkpoints and the global champion pointer.

Layout inside a runs directory (default ``runs/``)::

    runs/champion.json                 <- global best across all runs (pointer + stats)
    runs/<run>/best/step_000420000.zip <- every new per-run best, never overwritten
    runs/<run>/best.json               <- the run's current best (same format as champion.json)

Two rules keep the viewer from ever seeing a half-written file:

1. Every write goes to a temp file in the same directory, then ``os.replace``
   swaps it in. A rename within one filesystem is atomic: readers see either the
   old file or the new one.
2. Checkpoints are immutable (a new best gets a new file name). So the champion
   JSON is the only file that is ever replaced, and it is tiny.

Several training processes may finish an evaluation at the same moment. The
read-compare-write on champion.json is therefore done under a file lock, or a
slower writer could overwrite a better champion with a worse one.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

CHAMPION_FILE = "champion.json"


def _replace(src: str, dst: Path) -> None:
    # On Windows, os.replace fails if another process has dst open at that instant
    # (e.g. the viewer reading it). It's a transient condition, so retry briefly.
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_save_model(model: BaseAlgorithm, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a .zip suffix on the temp file, otherwise SB3 appends one to the name.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp.zip")
    os.close(fd)
    try:
        model.save(tmp)
        _replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed file, or None if it doesn't exist (yet) or can't be read."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def resolve_model_path(record: dict[str, Any], runs_dir: Path) -> Path:
    # Paths are stored relative to the runs dir so the folder can be moved/copied
    # (e.g. between WSL and Windows) without breaking the pointers.
    return (runs_dir / record["model_path"]).resolve()


def try_update_champion(runs_dir: Path, record: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Make ``record`` the global champion if it beats the current one.

    Returns (became_champion, previous_champion).
    """
    path = runs_dir / CHAMPION_FILE
    with FileLock(str(path) + ".lock", timeout=30):
        current = read_json(path)
        if current is not None and current["mean_reward"] >= record["mean_reward"]:
            return False, current
        atomic_write_json(path, record)
        return True, current


class ChampionCallback(BaseCallback):
    """Runs as ``EvalCallback(callback_on_new_best=...)``: fires each time this run's
    mean eval reward reaches a new high. Saves an immutable checkpoint, updates the
    run's best.json, and tries to become the global champion."""

    parent: EvalCallback

    def __init__(
        self,
        run_dir: Path,
        runs_dir: Path,
        run_name: str,
        env_cfg: dict[str, Any],
        algo: str,
        solved_threshold: float | None,
    ):
        super().__init__(verbose=1)
        self.run_dir = run_dir
        self.runs_dir = runs_dir
        self.run_name = run_name
        self.env_cfg = env_cfg
        self.algo = algo
        self.solved_threshold = solved_threshold
        self.solved_logged = False

    def _on_step(self) -> bool:
        rewards = self.parent.evaluations_results[-1]  # episode returns from the eval just run
        mean, std = float(np.mean(rewards)), float(np.std(rewards))
        step = self.num_timesteps

        model_path = self.run_dir / "best" / f"step_{step:09d}.zip"
        atomic_save_model(self.model, model_path)

        solved = self.solved_threshold is not None and mean >= self.solved_threshold
        record = {
            "run_name": self.run_name,
            "model_path": model_path.relative_to(self.runs_dir).as_posix(),
            "timestep": step,
            "mean_reward": mean,
            "std_reward": std,
            "n_eval_episodes": len(rewards),
            "solved": solved,
            "algo": self.algo,
            "env": self.env_cfg,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        atomic_write_json(self.run_dir / "best.json", record)
        print(f"[{self.run_name}] new run best at step {step:,}: {mean:.1f} +/- {std:.1f}", flush=True)

        if solved and not self.solved_logged:
            self.solved_logged = True
            self.logger.record("eval/solved_at_timestep", step)
            print(f"[{self.run_name}] *** SOLVED at step {step:,}: mean eval reward {mean:.1f} >= {self.solved_threshold:g} ***", flush=True)

        became, previous = try_update_champion(self.runs_dir, record)
        if became:
            prev = f"{previous['mean_reward']:.1f} ({previous['run_name']})" if previous else "none"
            print(f"[{self.run_name}] NEW GLOBAL CHAMPION: {mean:.1f} (previous: {prev})", flush=True)
        return True
