"""Watch the current champion (or a fixed checkpoint) attempt landings, forever.

    lander-viewer                      # follow runs/champion.json, hot-reloading
    lander-viewer --model runs/<run>/best/step_000420000.zip

The viewer renders frames itself (env render_mode="rgb_array") and draws them in
its own pygame window, so it can add a text overlay. A new env only needs to
support rgb_array rendering to work here.

Champion changes are picked up between episodes, never mid-episode. Start the
viewer before, during or after training: with no champion yet it waits.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import yaml

from lander.champion import CHAMPION_FILE, read_json, resolve_model_path
from lander.config import ALGOS
from lander.envs import get_spec, make_env

POLL_SECONDS = 2.0
OUTCOME_PAUSE_SECONDS = 1.5


class Window:
    def __init__(self, fps: int, scale: float):
        pygame.init()
        pygame.display.set_caption("lander viewer")
        self.fps, self.scale = fps, scale
        self.screen: pygame.Surface | None = None  # opened on the first frame, sized to it
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15)
        self.closed = False

    def pump(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                self.closed = True

    def show(self, frame: np.ndarray | None, lines: list[str], tick: bool = True) -> None:
        self.pump()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        else:
            surf = pygame.Surface((600, 400))
            surf.fill((20, 20, 30))
        if self.screen is None:
            w, h = surf.get_size()
            self.screen = pygame.display.set_mode((int(w * self.scale), int(h * self.scale)), pygame.RESIZABLE)
        # The window may have been resized (by the user or the OS): scale the frame to fit.
        if surf.get_size() != self.screen.get_size():
            surf = pygame.transform.smoothscale(surf, self.screen.get_size())
        self.screen.blit(surf, (0, 0))
        if lines:
            rendered = [self.font.render(t, True, (255, 255, 255)) for t in lines]
            box = pygame.Surface((max(r.get_width() for r in rendered) + 12, 18 * len(rendered) + 8), pygame.SRCALPHA)
            box.fill((0, 0, 0, 150))
            self.screen.blit(box, (6, 6))
            for i, r in enumerate(rendered):
                self.screen.blit(r, (12, 10 + 18 * i))
        pygame.display.flip()
        if tick:
            self.clock.tick(self.fps)

    def wait(self, seconds: float, frame: np.ndarray | None, lines: list[str]) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.closed:
            self.show(frame, lines, tick=False)
            time.sleep(0.05)


class ChampionSource:
    """Follows <runs_dir>/champion.json."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.path = runs_dir / CHAMPION_FILE
        self.last_poll = 0.0
        self.latest: dict[str, Any] | None = None

    def poll(self, force: bool = False) -> dict[str, Any] | None:
        if force or time.monotonic() - self.last_poll >= POLL_SECONDS:
            self.last_poll = time.monotonic()
            record = read_json(self.path)
            if record is not None:
                self.latest = record
        return self.latest


def record_for_checkpoint(model_path: Path, env_name: str | None) -> dict[str, Any]:
    """Build a champion-like record for an arbitrary checkpoint.

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
            with open(d / "config.yaml") as f:
                cfg = yaml.safe_load(f)
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


def describe(record: dict[str, Any]) -> list[str]:
    step = f"{record['timestep']:,}" if record.get("timestep") is not None else "?"
    if record.get("mean_reward") is not None:
        ev = f"{record['mean_reward']:.1f} +/- {record.get('std_reward', 0):.1f}"
    else:
        ev = "n/a"
    return [f"run:  {record['run_name']}", f"step: {step}", f"eval: {ev}"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--model", type=Path, help="replay this checkpoint instead of following the champion")
    p.add_argument("--env", help="env registry name for --model when the run's config.yaml can't be found")
    p.add_argument("--fps", type=int, default=50, help="playback speed (LunarLander runs at 50 steps/s)")
    p.add_argument("--scale", type=float, default=1.0, help="window scale factor")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    p.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = forever)")
    args = p.parse_args(argv)

    runs_dir = args.runs_dir.resolve()
    win = Window(args.fps, args.scale)
    source = None if args.model else ChampionSource(runs_dir)
    fixed = record_for_checkpoint(args.model.resolve(), args.env) if args.model else None

    model, loaded, env, env_key = None, None, None, None
    outcomes: dict[str, int] = {}
    episode = 0
    said_waiting = False
    while not win.closed and (args.episodes == 0 or episode < args.episodes):
        # --- Between episodes: (re)load the model if the champion changed. ---
        target = fixed or source.poll(force=True)
        if target is None:
            if not said_waiting:
                print(f"waiting for {source.path} ...", flush=True)
                said_waiting = True
            win.wait(POLL_SECONDS, None, ["waiting for first champion...", str(source.path)])
            continue
        if loaded is None or target["model_path"] != loaded["model_path"]:
            path = resolve_model_path(target, runs_dir)
            try:
                model = ALGOS[target.get("algo", "PPO")].load(path, device="cpu")
            except Exception as e:  # noqa: BLE001 - keep the viewer alive whatever goes wrong
                print(f"failed to load {path}: {e!r}; retrying", flush=True)
                if model is None:
                    win.wait(POLL_SECONDS, None, ["failed to load model, retrying...", str(path)])
                    continue
                # Keep showing the previous champion; we'll retry after this episode.
            else:
                verb = "loaded" if loaded is None else "hot-reloaded"
                loaded = target
                print(f"{verb} {path.name}: " + ", ".join(describe(target)), flush=True)
                if (target["env"]["name"], repr(target["env"]["kwargs"])) != env_key:
                    if env is not None:
                        env.close()
                    env_key = (target["env"]["name"], repr(target["env"]["kwargs"]))
                    env = make_env(target["env"]["name"], render_mode="rgb_array", **target["env"]["kwargs"])
        spec = get_spec(loaded["env"]["name"])

        # --- One episode. ---
        episode += 1
        obs, _ = env.reset()
        ep_reward, steps, done = 0.0, 0, False
        terminated = truncated = False
        while not done and not win.closed:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated
            pending = source is not None and (c := source.poll()) is not None and c["model_path"] != loaded["model_path"]
            lines = describe(loaded) + [f"ep {episode}  step {steps}  reward {ep_reward:7.1f}"]
            if pending:
                lines.append(f"new champion ({c['mean_reward']:.1f}) loads next episode")
            win.show(env.render(), lines)
        if win.closed:
            break

        result = spec.outcome(env, terminated, truncated)
        outcomes[result] = outcomes.get(result, 0) + 1
        tally = ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items()))
        print(f"episode {episode}: {result}, reward {ep_reward:.1f}, {steps} steps   [{tally}]", flush=True)
        lines = describe(loaded) + [f"ep {episode}: {result.upper()}  reward {ep_reward:.1f}"]
        win.wait(OUTCOME_PAUSE_SECONDS, env.render(), lines)

    if env is not None:
        env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
