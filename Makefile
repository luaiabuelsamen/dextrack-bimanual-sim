# Everything runs locally. No Modal.
# Override for your environment, e.g.
#   make test PY=/path/to/venv/bin/python
PY  ?= python3
# a stale system-site anyio plugin breaks collection; this venv needs none
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1
GPU ?= ./mjx_env.sh python

.PHONY: help install test test-fast axis hand-axis expert bc parity figures clean
help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

install:          ## editable install into the CPU venv
	$(PY) -m pip install -e .

test-fast:        ## invariants that run in seconds
	$(PY) -m pytest -q -m "not slow and not gpu"

test:             ## everything except GPU
	$(PY) -m pytest -q -m "not gpu"

inventory:        ## what hands and arms this machine can build
	$(PY) -m experiments.inventory

axis:             ## M2 part 1: opposition floor, four hands
	$(PY) -m experiments.opposition_axis

hand-axis:        ## M2 part 2: which block widths each hand holds
	$(PY) -m experiments.hand_axis

expert:           ## bimanual expert + the one-handed control
	$(PY) -m experiments.bimanual_expert

bc:               ## behaviour cloning: collect, train, evaluate
	$(PY) -m experiments.bc

parity:           ## CPU MuJoCo vs warp, outcome level (needs GPU)
	$(GPU) -m experiments.warp_parity

figures:          ## regenerate the figures
	$(PY) -m experiments.fig_axis

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

vendor:           ## fetch the MuJoCo Menagerie hand models (sparse, ~30 MB)
	@test -d vendor/mujoco_menagerie || git clone --depth 1 --filter=blob:none --sparse \
	  https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
	@cd vendor/mujoco_menagerie && git sparse-checkout set leap_hand shadow_hand wonik_allegro
	@echo "menagerie ready"
