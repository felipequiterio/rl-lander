"""Evaluate checkpoints on a fixed, seeded set of episodes.

    lander-eval                                   # the current champion
    lander-eval runs/a/best/step_X.zip runs/b/best/step_Y.zip --episodes 500
    lander-eval --out results.json                # also save per-episode results
    lander-eval --env descent3d --controller zem  # a hand-written / optimisation controller
    lander-eval --runs-dir runs/terrain --env-arg terrain=apollo11  # a terrain policy on another site

Episode i is always reset with seed (--seed + i), so every model faces exactly
the same start conditions. That makes comparisons paired, which cuts the noise
when comparing two controllers (and later, RL vs G-FOLD).

Reported per model:
- success rate with a 95% Wilson confidence interval. Success means the env's
  definition of mission success (EnvSpec.success), not a reward threshold.
- counts of every outcome;
- mean episode return with a 95% confidence interval;
- the env's landing metrics (EnvSpec.metrics), averaged over successful
  episodes only, since that's where "how good was the landing" is meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from lander.atomic import read_json
from lander.champion import CHAMPION_FILE, resolve_model_path
from lander.checkpoints import load_policy, record_for_checkpoint
from lander.config import parse_env_args, with_env_args
from lander.controllers import CONTROLLERS
from lander.envs import get_spec, make_env


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; well-behaved near 0% and 100%,
    unlike the textbook p +/- 1.96*sqrt(p(1-p)/n)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_episodes(make_policy, record: dict[str, Any], episodes: int, seed: int, deterministic: bool) -> list[dict]:
    """``make_policy(env)`` returns anything with ``predict(obs, deterministic) -> action``."""
    spec = get_spec(record["env"]["name"])
    env = make_env(record["env"]["name"], **record["env"]["kwargs"])
    policy = make_policy(env)
    results = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        observations, actions, ret, done = [obs], [], 0.0, False
        while not done:
            action = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            ret += float(reward)
            done = terminated or truncated
        outcome = spec.outcome(env, terminated, truncated)
        results.append({
            "seed": seed + i,
            "outcome": outcome,
            "success": outcome in spec.success,
            "return": ret,
            "steps": len(actions),
            "metrics": spec.metrics(env, observations, actions) if spec.metrics else {},
        })
    env.close()
    return results


def summarize(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    wins = [r for r in results if r["success"]]
    returns = np.array([r["return"] for r in results])
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    metrics = {}
    for key in (wins[0]["metrics"] if wins else {}):
        v = np.array([r["metrics"][key] for r in wins])
        metrics[key] = {"mean": float(v.mean()), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    return {
        "episodes": n,
        "success_rate": len(wins) / n,
        "success_ci95": wilson(len(wins), n),
        "outcomes": dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_ci95_halfwidth": float(1.96 * returns.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "metrics_on_success": metrics,
    }


def print_summary(name: str, record: dict[str, Any], s: dict[str, Any]) -> None:
    lo, hi = s["success_ci95"]
    step = f"{record['timestep']:,}" if record.get("timestep") is not None else "?"
    print(f"\n{name}" + (f"  (run {record['run_name']}, step {step})" if record.get("model_path") else ""))
    print(f"  success      {s['success_rate'] * 100:5.1f}%   95% CI [{lo * 100:.1f}, {hi * 100:.1f}]   over {s['episodes']} episodes")
    print("  outcomes     " + ", ".join(f"{k}: {v}" for k, v in s["outcomes"].items()))
    print(f"  return       {s['return_mean']:.1f} +/- {s['return_ci95_halfwidth']:.1f} (95% CI of the mean), std {s['return_std']:.1f}")
    if s["metrics_on_success"]:
        print("  on successful episodes:            mean       p50       p95")
        for key, m in s["metrics_on_success"].items():
            print(f"    {key:<28} {m['mean']:9.3f} {m['p50']:9.3f} {m['p95']:9.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", type=Path, help="checkpoints to evaluate (default: the current champion)")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1_000_000, help="episode i uses seed + i (kept away from training/eval seeds)")
    p.add_argument("--env", help="env registry name (for --controller, or if a checkpoint's run config can't be found)")
    p.add_argument("--controller", action="append", default=[], choices=sorted(CONTROLLERS),
                   help="evaluate a non-learned controller (repeatable); needs --env")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="override an env parameter for every model, e.g. terrain=apollo11 (repeatable)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    p.add_argument("--out", type=Path, help="write summaries and per-episode results to this JSON file")
    args = p.parse_args(argv)

    paths = [m.resolve() for m in args.models]
    env_args = parse_env_args(args.env_arg)
    if args.controller and not args.env:
        raise SystemExit("--controller needs --env")
    if not paths and not args.controller:
        champ = read_json(args.runs_dir / CHAMPION_FILE)
        if champ is None:
            raise SystemExit(f"no champion in {args.runs_dir}; pass checkpoint paths instead")
        paths = [resolve_model_path(champ, args.runs_dir.resolve())]

    report = {"episodes": args.episodes, "seed": args.seed, "deterministic": not args.stochastic, "models": []}
    jobs = []  # (label, record, policy factory)
    for path in paths:
        record = with_env_args(record_for_checkpoint(path, args.env), env_args)
        label = str(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)
        jobs.append((label, record, lambda env, p=path, r=record: load_policy(p, r.get("algo", "PPO"))))
    for name in args.controller:
        record = {"run_name": f"controller:{name}", "model_path": None, "timestep": None, "env": {"name": args.env, "kwargs": env_args}}
        jobs.append((f"controller {name}", record, CONTROLLERS[name]))

    for label, record, make_policy in jobs:
        t0 = time.time()
        results = run_episodes(make_policy, record, args.episodes, args.seed, not args.stochastic)
        summary = summarize(results)
        print_summary(label, record, summary)
        print(f"  ({time.time() - t0:.0f}s)")
        report["models"].append({"label": label, "record": record, "summary": summary, "episodes": results})

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
