# Shortcuts for the common workflows. Run `make help` (or just `make`) for the list.
# Variables can be overridden on the command line, e.g. `make launch SEEDS="0 1"`
# or `make viewer VIEWER_ARGS=--mute`.

CONFIG ?= configs/ppo_lunar.yaml
SEED   ?= 0
SEEDS  ?= 0 1 2 3
RUNS   ?= runs
FPS    ?=
STEPS  ?= 5e4
MODEL  ?=
EPISODES ?= 200
VIEWER_ARGS ?=
BDB    ?= $(LANDER_BDB)
RUNS3D ?= runs/descent3d
RUNS_TERRAIN ?= runs/terrain
SITE   ?= apollo17
VIEW_SEED ?= 1000
COMPARE_SCALE ?= 0.75
# ZEM/ZEV over real terrain: the same site and target distribution as configs/ppo_descent3d_terrain.yaml
TERRAIN_ARGS ?= --env-arg terrain=$(SITE) --env-arg target_max_dist=2300

UV := uv run

.DEFAULT_GOAL := help
.PHONY: help setup train smoke launch viewer replay viewer-plain viewer-ppo viewer-zem viewer-zev viewer-compare _compare-ppo _compare-zem \
	train-terrain viewer-terrain viewer-terrain-zem viewer-terrain-compare _tcompare-ppo _tcompare-zem eval-terrain \
	import-assets import-terrain eval tensorboard test lint clean

help: ## Show this help
	@echo "Usage: make <target> [VAR=value ...]"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables (current value):"
	@echo "  CONFIG=$(CONFIG)  SEED=$(SEED)  SEEDS=\"$(SEEDS)\"  RUNS=$(RUNS)"
	@echo "  FPS=$(FPS)  STEPS=$(STEPS)  EPISODES=$(EPISODES)  MODEL=$(MODEL)  VIEWER_ARGS=$(VIEWER_ARGS)"
	@echo "  RUNS3D=$(RUNS3D)  RUNS_TERRAIN=$(RUNS_TERRAIN)  SITE=$(SITE)  VIEW_SEED=$(VIEW_SEED)  COMPARE_SCALE=$(COMPARE_SCALE)"
	@echo "  BDB=$(BDB)  (optional: re-import the Apollo LM live from KSP instead of the bundled copy)"

setup: ## Create .venv with the locked dependencies (Python 3.11)
	uv sync

train: ## Train one run: CONFIG, SEED
	$(UV) lander-train $(CONFIG) --seed $(SEED) --runs-dir $(RUNS)

smoke: ## Quick smoke-test run (STEPS env steps, default 5e4)
	$(UV) lander-train $(CONFIG) --seed $(SEED) --runs-dir $(RUNS) --total-timesteps $(STEPS)

launch: ## Train several seeds in parallel: CONFIG, SEEDS
	$(UV) lander-launch $(CONFIG) --seeds $(SEEDS) --runs-dir $(RUNS)

viewer: ## Watch the current champion, hot-reloading (VIEWER_ARGS=--mute for silence)
	$(UV) lander-viewer --runs-dir $(RUNS) $(if $(FPS),--fps $(FPS)) $(if $(BDB),--bdb "$(BDB)") $(VIEWER_ARGS)

replay: ## Replay one checkpoint: MODEL=runs/<run>/best/step_*.zip
	@test -n "$(MODEL)" || { echo "usage: make replay MODEL=runs/<run>/best/step_000420000.zip"; exit 1; }
	$(UV) lander-viewer --model $(MODEL) $(if $(FPS),--fps $(FPS)) $(if $(BDB),--bdb "$(BDB)") $(VIEWER_ARGS)

viewer-ppo: ## Watch the descent3d PPO champion (RUNS3D), episodes from VIEW_SEED
	$(UV) lander-viewer --runs-dir $(RUNS3D) --seed $(VIEW_SEED) --title "PPO champion" $(if $(FPS),--fps $(FPS)) $(if $(BDB),--bdb "$(BDB)") $(VIEWER_ARGS)

viewer-zem: ## Watch the ZEM/ZEV guidance law on descent3d, same episodes as viewer-ppo
	$(UV) lander-viewer --env descent3d --controller zem --seed $(VIEW_SEED) --title "ZEM/ZEV guidance" $(if $(FPS),--fps $(FPS)) $(if $(BDB),--bdb "$(BDB)") $(VIEWER_ARGS)

viewer-zev: viewer-zem ## Alias of viewer-zem (ZEM/ZEV is one guidance law)

viewer-compare: ## PPO and ZEM/ZEV side by side, same start conditions (sound from PPO only)
	@$(MAKE) --no-print-directory -j2 _compare-ppo _compare-zem

_compare-ppo:
	$(UV) lander-viewer --runs-dir $(RUNS3D) --seed $(VIEW_SEED) --title "PPO champion" --scale $(COMPARE_SCALE) --window-pos 0,40 $(if $(FPS),--fps $(FPS)) $(VIEWER_ARGS)

_compare-zem:
	$(UV) lander-viewer --env descent3d --controller zem --seed $(VIEW_SEED) --title "ZEM/ZEV guidance" --scale $(COMPARE_SCALE) --window-pos 910,40 --mute $(if $(FPS),--fps $(FPS)) $(VIEWER_ARGS)

train-terrain: ## Train PPO over real terrain (Apollo 17), several seeds in parallel: SEEDS
	$(UV) lander-launch configs/ppo_descent3d_terrain.yaml --seeds $(SEEDS) --runs-dir $(RUNS_TERRAIN)

viewer-terrain: ## Watch the real-terrain PPO champion (RUNS_TERRAIN) land at SITE (apollo17; apollo11 = never trained on)
	$(UV) lander-viewer --runs-dir $(RUNS_TERRAIN) --seed $(VIEW_SEED) --title "PPO champion, real terrain" --env-arg terrain=$(SITE) $(if $(FPS),--fps $(FPS)) $(if $(BDB),--bdb "$(BDB)") $(VIEWER_ARGS)

viewer-terrain-zem: ## Watch ZEM/ZEV guidance (terrain-blind) over real terrain, same episodes as viewer-terrain
	$(UV) lander-viewer --env descent3d --controller zem $(TERRAIN_ARGS) --seed $(VIEW_SEED) --title "ZEM/ZEV guidance, real terrain" $(if $(FPS),--fps $(FPS)) $(if $(BDB),--bdb "$(BDB)") $(VIEWER_ARGS)

viewer-terrain-compare: ## PPO and ZEM/ZEV side by side over real terrain
	@$(MAKE) --no-print-directory -j2 _tcompare-ppo _tcompare-zem

_tcompare-ppo:
	$(UV) lander-viewer --runs-dir $(RUNS_TERRAIN) --seed $(VIEW_SEED) --title "PPO champion, real terrain" --env-arg terrain=$(SITE) --scale $(COMPARE_SCALE) --window-pos 0,40 $(if $(FPS),--fps $(FPS)) $(VIEWER_ARGS)

_tcompare-zem:
	$(UV) lander-viewer --env descent3d --controller zem $(TERRAIN_ARGS) --seed $(VIEW_SEED) --title "ZEM/ZEV guidance, real terrain" --scale $(COMPARE_SCALE) --window-pos 910,40 --mute $(if $(FPS),--fps $(FPS)) $(VIEWER_ARGS)

eval-terrain: ## Evaluate the real-terrain champion and ZEM/ZEV at SITE on EPISODES fixed seeds
	$(UV) lander-eval --runs-dir $(RUNS_TERRAIN) --env descent3d --controller zem $(TERRAIN_ARGS) --episodes $(EPISODES)
	$(UV) lander-eval --runs-dir $(RUNS_TERRAIN) --env-arg terrain=$(SITE) --episodes $(EPISODES)

import-assets: ## Re-import the Apollo LM from a KSP Bluedog_DB install (BDB=...) into src/lander/assets
	@test -n "$(BDB)" || { echo 'usage: make import-assets BDB="/path/to/GameData/Bluedog_DB"'; exit 1; }
	$(UV) python -m lander.viz3d.import_bdb "$(BDB)"

import-terrain: ## Re-import landing sites from NASA LROC (all, or SITE=apollo17; a few hundred MB, cached)
	$(UV) python -m lander.viz3d.import_lroc $(if $(filter command line,$(origin SITE)),$(SITE))

viewer-plain: ## Viewer with the env's own (plain) rendering
	$(UV) lander-viewer --runs-dir $(RUNS) $(if $(FPS),--fps $(FPS)) --plain $(VIEWER_ARGS)

eval: ## Evaluate on EPISODES fixed seeds: the champion, or MODEL="a.zip b.zip"
	$(UV) lander-eval $(MODEL) --runs-dir $(RUNS) --episodes $(EPISODES)

tensorboard: ## Open TensorBoard on the runs dir (http://localhost:6006)
	$(UV) tensorboard --logdir $(RUNS)

test: ## Run the test suite
	$(UV) pytest -q

lint: ## Lint the code with ruff
	$(UV) ruff check src/ tests/

clean: ## Remove Python caches (never touches runs/)
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
