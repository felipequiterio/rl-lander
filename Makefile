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

UV := uv run

.DEFAULT_GOAL := help
.PHONY: help setup train smoke launch viewer replay viewer-plain eval tensorboard test lint clean

help: ## Show this help
	@echo "Usage: make <target> [VAR=value ...]"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables (current value):"
	@echo "  CONFIG=$(CONFIG)  SEED=$(SEED)  SEEDS=\"$(SEEDS)\"  RUNS=$(RUNS)"
	@echo "  FPS=$(FPS)  STEPS=$(STEPS)  EPISODES=$(EPISODES)  MODEL=$(MODEL)  VIEWER_ARGS=$(VIEWER_ARGS)"

setup: ## Create .venv with the locked dependencies (Python 3.11)
	uv sync

train: ## Train one run: CONFIG, SEED
	$(UV) lander-train $(CONFIG) --seed $(SEED) --runs-dir $(RUNS)

smoke: ## Quick smoke-test run (STEPS env steps, default 5e4)
	$(UV) lander-train $(CONFIG) --seed $(SEED) --runs-dir $(RUNS) --total-timesteps $(STEPS)

launch: ## Train several seeds in parallel: CONFIG, SEEDS
	$(UV) lander-launch $(CONFIG) --seeds $(SEEDS) --runs-dir $(RUNS)

viewer: ## Watch the current champion, hot-reloading (VIEWER_ARGS=--mute for silence)
	$(UV) lander-viewer --runs-dir $(RUNS) $(if $(FPS),--fps $(FPS)) $(VIEWER_ARGS)

replay: ## Replay one checkpoint: MODEL=runs/<run>/best/step_*.zip
	@test -n "$(MODEL)" || { echo "usage: make replay MODEL=runs/<run>/best/step_000420000.zip"; exit 1; }
	$(UV) lander-viewer --model $(MODEL) $(if $(FPS),--fps $(FPS)) $(VIEWER_ARGS)

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
