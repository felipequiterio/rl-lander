"""Watch the current champion (or a fixed checkpoint) attempt landings, forever.

    lander-viewer                      # follow runs/champion.json, hot-reloading
    lander-viewer --model runs/<run>/best/step_000420000.zip
    lander-viewer --env descent3d --controller zem --seed 1000   # a hand-written controller

Drawing is done by the env's "scene" (see EnvSpec.scene): for LunarLander a
custom renderer that reads the physics state and draws the lander, terrain,
flames and dust. Envs without a scene fall back to their own rgb_array frames.
On top sits an env-agnostic HUD (lander.hud).

Champion changes are picked up between episodes, never mid-episode. Start the
viewer before, during or after training: with no champion yet it waits.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from lander.atomic import read_json
from lander.champion import CHAMPION_FILE, resolve_model_path
from lander.checkpoints import load_policy, record_for_checkpoint
from lander.controllers import CONTROLLERS
from lander.envs import get_spec, make_env
from lander.hud import Hud
from lander.sound import SoundFx

POLL_SECONDS = 2.0
OUTCOME_PAUSE_SECONDS = 2.2


class Display:
    """Window plus a fixed-size canvas; the canvas is letterboxed into the
    (resizable) window, so scenes draw at one resolution regardless of window size."""

    def __init__(self, fps: int, scale: float):
        pygame.init()
        pygame.display.set_caption("lander viewer")
        self.fps, self.scale = fps, scale
        self.canvas = pygame.Surface((1200, 800))
        self.window: pygame.Surface | None = None
        self.clock = pygame.time.Clock()
        self.closed = False
        self.on_key: dict[int, Any] = {}  # extra key bindings: key -> callback

    def use_canvas(self, size: tuple[int, int]) -> pygame.Surface:
        if self.canvas.get_size() != tuple(size):
            self.canvas = pygame.Surface(size)
        return self.canvas

    def present(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                self.closed = True
            elif event.type == pygame.KEYDOWN and event.key in self.on_key:
                self.on_key[event.key]()
        cw, ch = self.canvas.get_size()
        if self.window is None:
            self.window = pygame.display.set_mode((int(cw * self.scale), int(ch * self.scale)), pygame.RESIZABLE)
        ww, wh = self.window.get_size()
        if (ww, wh) == (cw, ch):
            self.window.blit(self.canvas, (0, 0))
        else:
            k = min(ww / cw, wh / ch)
            size = (max(1, int(cw * k)), max(1, int(ch * k)))
            self.window.fill((0, 0, 0))
            self.window.blit(pygame.transform.smoothscale(self.canvas, size), ((ww - size[0]) // 2, (wh - size[1]) // 2))
        pygame.display.flip()
        self.clock.tick(self.fps)


class FrameScene:
    """Fallback scene for envs without a custom renderer: shows env.render() frames."""

    render_mode = "rgb_array"

    def __init__(self) -> None:
        self.size = (600, 400)

    def reset(self, env) -> None:
        try:
            frame = env.render()
        except NotImplementedError:
            frame = None
        if frame is None:
            raise SystemExit(f"{env.unwrapped.__class__.__name__} has no viewer scene (EnvSpec.scene) and no rgb_array rendering")
        h, w = frame.shape[:2]
        self.size = (w, h)

    def update(self, env, action, dt, frac: float = 1.0) -> None:
        pass

    def on_outcome(self, env, label) -> None:
        pass

    def draw(self, canvas, env) -> None:
        canvas.blit(pygame.surfarray.make_surface(env.render().swapaxes(0, 1)), (0, 0))

    def gauges(self, env) -> list:
        return []

    def banner(self, label: str) -> tuple[str, str, str]:
        return label.upper(), "", "normal"

    def outcome_tone(self, label: str) -> str:
        return "normal"

    def outcome_sound(self, label: str) -> str | None:
        return None

    def sound_levels(self, env) -> tuple[dict[str, float], float]:
        return {}, 0.0


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
    p.add_argument("--fps", type=int, help="frames per second (default: real time; e.g. 150 = 3x for both envs)")
    p.add_argument("--scale", type=float, default=1.0, help="initial window size relative to the 1200x800 canvas")
    p.add_argument("--plain", action="store_true", help="use the env's own rendering instead of the custom scene")
    p.add_argument("--scene", help="pick one of the env's alternative scenes (descent3d: 3d, 2d)")
    p.add_argument("--bdb", help="path to a KSP Bluedog_DB folder, for the 3D Apollo LM model (or set LANDER_BDB)")
    p.add_argument("--mute", action="store_true", help="start with sound off (toggle any time with M)")
    p.add_argument("--volume", type=float, default=0.7, help="master volume, 0..1")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    p.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = forever)")
    p.add_argument("--controller", choices=sorted(CONTROLLERS), help="fly a hand-written controller instead of a trained policy")
    p.add_argument("--seed", type=int, help="episode i starts from seed + i (same starts across viewers); default: random")
    p.add_argument("--title", help="window title")
    p.add_argument("--window-pos", help="initial window position 'X,Y' (e.g. to put two viewers side by side)")
    args = p.parse_args(argv)
    if args.window_pos:
        os.environ["SDL_VIDEO_WINDOW_POS"] = args.window_pos  # read by SDL when the window is created

    if args.bdb:
        os.environ["LANDER_BDB"] = args.bdb
    runs_dir = args.runs_dir.resolve()
    display = Display(args.fps or 30, args.scale)
    hud = Hud()
    fx = SoundFx(enabled=not args.mute, volume=args.volume)
    display.on_key[pygame.K_m] = lambda: hud.show_toast("Sound on" if fx.toggle() else "Sound muted", 1.5)
    if args.controller:
        source, mode = None, "controller"
        fixed = {
            "run_name": {"zem": "ZEM/ZEV guidance"}.get(args.controller, args.controller),
            "model_path": f"controller:{args.controller}",
            "line2": "closed-form guidance law, no training",
            "line3": "energy-optimal feedback (not fuel-optimal)",
            "env": {"name": args.env or "descent3d", "kwargs": {}},
        }
    else:
        source = None if args.model else ChampionSource(runs_dir)
        fixed = record_for_checkpoint(args.model.resolve(), args.env) if args.model else None
        mode = "replay" if fixed else "champion"
    pygame.display.set_caption(f"lander viewer: {args.title or mode}")

    model, loaded, env, env_key, scene = None, None, None, None, None
    outcomes: dict[str, int] = {}
    episode = 0
    said_waiting = False
    t0 = time.monotonic()

    def frame(sim_time: float, ep_reward: float, pending: dict | None, banner=None) -> None:
        canvas = display.use_canvas(scene.size)
        scene.draw(canvas, env)
        card = hud.model_card(canvas, mode, loaded, pending)
        hud.toast(canvas, card, 1 / display.fps)
        hud.gauges(canvas, scene.gauges(env))
        tally = [(name, n, scene.outcome_tone(name)) for name, n in sorted(outcomes.items(), key=lambda kv: -kv[1])]
        hud.episode_strip(canvas, episode, sim_time, ep_reward, tally)
        if banner:
            hud.banner(canvas, *banner)
        hud.sound_hint(canvas, fx.enabled, fx.available)
        levels, pan = scene.sound_levels(env)
        fx.update(levels, pan)
        display.present()

    while not display.closed and (args.episodes == 0 or episode < args.episodes):
        # --- Between episodes: (re)load the model if the champion changed. ---
        target = fixed or source.poll(force=True)
        if target is None:
            if not said_waiting:
                print(f"waiting for {source.path} ...", flush=True)
                said_waiting = True
            end = time.monotonic() + POLL_SECONDS
            while time.monotonic() < end and not display.closed:
                hud.waiting(display.canvas, str(source.path), time.monotonic() - t0)
                display.present()
            continue
        if loaded is None or target["model_path"] != loaded["model_path"]:
            # The env (and its scene) first: controllers read their parameters from it.
            new_key = (target["env"]["name"], repr(target["env"]["kwargs"]))
            if new_key != env_key:
                if env is not None:
                    env.close()
                spec = get_spec(target["env"]["name"])
                if args.plain or not spec.scene:
                    scene = FrameScene()
                elif args.scene:
                    if args.scene not in spec.scenes:
                        raise SystemExit(f"--scene {args.scene}: {target['env']['name']} has {sorted(spec.scenes) or 'no alternative scenes'}")
                    scene = spec.scenes[args.scene]()
                else:
                    scene = spec.scene()
                for k, fn in getattr(scene, "key_bindings", {}).items():  # scene-specific keys (e.g. C: camera)
                    display.on_key[k] = fn
                env = make_env(target["env"]["name"], render_mode=scene.render_mode, **target["env"]["kwargs"])
                env_key = new_key
            try:
                if target["model_path"].startswith("controller:"):
                    model = CONTROLLERS[target["model_path"].removeprefix("controller:")](env)
                    path = Path(target["model_path"])
                else:
                    path = resolve_model_path(target, runs_dir)
                    model = load_policy(path, target.get("algo", "PPO"))
            except Exception as e:  # noqa: BLE001 - keep the viewer alive whatever goes wrong
                print(f"failed to load {target['model_path']}: {e!r}; retrying", flush=True)
                if model is None:
                    end = time.monotonic() + POLL_SECONDS
                    while time.monotonic() < end and not display.closed:
                        hud.waiting(display.canvas, f"failed to load {Path(target['model_path']).name}, retrying", time.monotonic() - t0)
                        display.present()
                    continue
                # Keep showing the previous champion; we'll retry after this episode.
            else:
                if loaded is not None:
                    hud.show_toast(f"New champion loaded: {target['run_name']}  {target['mean_reward']:.1f}")
                    fx.play("chime")
                print(f"{'loaded' if loaded is None else 'hot-reloaded'} {path.name}: " + ", ".join(describe(target)), flush=True)
                loaded = target
        spec = get_spec(loaded["env"]["name"])
        dt = 1.0 / env.metadata.get("render_fps", 50)  # simulated seconds per env step
        # Scenes may draw several interpolated frames per env step (descent3d: 10 at 5 Hz).
        substeps = getattr(scene, "substeps", 1)
        display.fps = args.fps or round(substeps / dt)  # default: real time

        # --- One episode. ---
        episode += 1
        obs, _ = env.reset(seed=None if args.seed is None else args.seed + episode - 1)
        scene.reset(env)
        ep_reward, steps, done = 0.0, 0, False
        terminated = truncated = False
        while not done and not display.closed:
            action = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated
            for k in range(substeps):
                frac = (k + 1) / substeps
                scene.update(env, action, dt / substeps, frac=frac)
                c = source.poll() if source else None
                frame((steps - 1 + frac) * dt, ep_reward, c if c and c["model_path"] != loaded["model_path"] else None)
                if display.closed:
                    break
        if display.closed:
            break

        result = spec.outcome(env, terminated, truncated)
        outcomes[result] = outcomes.get(result, 0) + 1
        tally = ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items()))
        print(f"episode {episode}: {result}, reward {ep_reward:.1f}, {steps} steps   [{tally}]", flush=True)

        # --- Outcome pause: keep animating (dust settling, explosion) under a banner. ---
        scene.on_outcome(env, result)
        fx.play(scene.outcome_sound(result), scene.sound_levels(env)[1])
        title, subtitle, tone = scene.banner(result)
        subtitle = f"{subtitle}  \u00b7  reward {ep_reward:.1f}" if subtitle else f"reward {ep_reward:.1f}"
        fps = display.fps
        n_frames = int(OUTCOME_PAUSE_SECONDS * fps)
        for i in range(n_frames):
            if display.closed:
                break
            scene.update(env, None, dt / substeps)
            fade = min(1.0, i / (0.15 * fps), (n_frames - i) / (0.3 * fps))
            c = source.poll() if source else None
            frame(steps * dt, ep_reward, c if c and c["model_path"] != loaded["model_path"] else None, (title, subtitle, tone, fade))

    if env is not None:
        env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
