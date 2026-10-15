"""Launch several training runs as parallel processes.

    lander-launch configs/ppo_lunar.yaml --seeds 0 1 2 3
    lander-launch configs/ppo_lunar.yaml configs/ppo_lunar_big.yaml --seeds 0 1

Every config is run once per seed. Each run writes its stdout to
<runs-dir>/<run>/train.log. A status table (each run's best eval score and the
global champion) is printed periodically. Ctrl+C stops all runs; each one
saves its final model before exiting.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from lander.atomic import read_json
from lander.champion import CHAMPION_FILE
from lander.train import default_run_name


def status_table(runs: list[tuple[str, subprocess.Popen]], runs_dir: Path) -> str:
    rows = []
    for name, proc in runs:
        best = read_json(runs_dir / name / "best.json")
        state = "running" if proc.poll() is None else f"exit {proc.returncode}"
        succ = f" ({best['success_rate']:.0%} success)" if best and best.get("success_rate") is not None else ""
        score = f"{best['mean_reward']:8.1f} @ {best['timestep']:>9,}{succ}" if best else f"{'-':>8}"
        rows.append(f"  {name:<40} {state:<9} best {score}")
    champ = read_json(runs_dir / CHAMPION_FILE)
    if champ:
        succ = f" ({champ['success_rate']:.0%} success)" if champ.get("success_rate") is not None else ""
        rows.append(f"  champion: {champ['run_name']} {champ['mean_reward']:.1f} @ {champ['timestep']:,}{succ}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="+", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--total-timesteps", help="override every config (e.g. 5e4 for a smoke test)")
    p.add_argument("--status-every", type=float, default=30.0, help="seconds between status tables")
    args = p.parse_args(argv)

    runs_dir = args.runs_dir.resolve()
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # keep train.log live
    runs: list[tuple[str, subprocess.Popen]] = []
    for config in args.configs:
        for seed in args.seeds:
            name = default_run_name(config, seed)
            cmd = [sys.executable, "-m", "lander.train", str(config), "--seed", str(seed),
                   "--run-name", name, "--runs-dir", str(runs_dir)]
            if args.total_timesteps:
                cmd += ["--total-timesteps", args.total_timesteps]
            (runs_dir / name).mkdir(parents=True)
            # The child inherits the file handle, so the parent can close its copy.
            with open(runs_dir / name / "train.log", "w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env)
            runs.append((name, proc))
            print(f"started {name} (pid {proc.pid})", flush=True)

    print(f"\n{len(runs)} runs -> {runs_dir}   (tensorboard --logdir {args.runs_dir})\n", flush=True)
    try:
        last_status = time.monotonic()
        while any(proc.poll() is None for _, proc in runs):
            if time.monotonic() - last_status >= args.status_every:
                print(time.strftime("%H:%M:%S") + "\n" + status_table(runs, runs_dir) + "\n", flush=True)
                last_status = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        # Ctrl+C in a terminal reaches every process in the foreground group, so
        # the runs already got it and are saving their final models. Forwarding
        # a second SIGINT could interrupt that save, so just wait for them.
        print("\nCtrl+C: waiting for runs to save and exit...", flush=True)
        for _, proc in runs:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("\nfinal:\n" + status_table(runs, runs_dir), flush=True)


if __name__ == "__main__":
    main()
