# Everything runs locally. No Modal.
# Override for your environment, e.g.
#   make test PY=/path/to/venv/bin/python
PY  ?= python3
# a stale system-site anyio plugin breaks collection; this venv needs none
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1
export PYTHONPATH := $(CURDIR)
GPU ?= ./mjx_env.sh python

.PHONY: help install test test-fast axis inventory expert matched matched-fig \
        g1 g1-analysis render-tasks render-tracking vec parity parity-mjx clean vendor

help:
	@grep -E '^[a-z0-9-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

install:          ## editable install into the CPU venv
	$(PY) -m pip install -e .

test-fast:        ## invariants that run in seconds
	$(PY) -m pytest -q -m "not slow and not gpu"

test:             ## everything except GPU
	$(PY) -m pytest -q -m "not gpu"

# -- measurements ----------------------------------------------------------
axis:             ## M1: the opposition axis (floor + aperture, with provenance)
	$(PY) -m oppdef.hands.axis

inventory:        ## what hands and arms this machine can build
	$(PY) -m experiments.grasp_metrics.inventory

# -- tasks and comparisons -------------------------------------------------
g1:               ## T1: the pre-registered three-arm comparison (docs/G1_PREREGISTRATION.md)
	$(PY) experiments/grasp_metrics/g1.py

g1-analysis:      ## the pre-registered analysis of the above
	$(PY) experiments/grasp_metrics/g1_analysis.py

matched:          ## T1: budget-matched pose vs wrench (two arms)
	$(PY) experiments/grasp_metrics/matched.py

matched-fig:      ## figure for the matched comparison (statistics derived, not typed)
	$(PY) experiments/grasp_metrics/fig_matched.py

expert:           ## T2: the bimanual peg task and its one-handed control
	$(PY) -m experiments.grasp_metrics.bimanual_expert

render-tasks:     ## re-render every task GIF by replaying saved grasps
	MUJOCO_GL=egl $(PY) experiments/grasp_metrics/render_tasks.py

render-tracking:  ## capture and render the README's measured GRAB tracking demos
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=src:. $(PY) -m experiments.tracking.render_tracking mug bowl binoculars camera

# -- infrastructure --------------------------------------------------------
vec:              ## batched-stepping throughput and agreement, CPU backend
	$(PY) -m experiments.infra.vec_bench

parity:           ## CPU MuJoCo vs warp, outcome level (needs GPU)
	$(GPU) -m experiments.infra.warp_parity

parity-mjx:       ## CPU MuJoCo vs MJX (BLOCKED here: cuSolver, see NOTES)
	PYTHONPATH=src $(GPU) -m experiments.infra.mjx_parity

vendor:           ## fetch the MuJoCo Menagerie hand models (sparse, ~30 MB)
	@test -d vendor/mujoco_menagerie || git clone --depth 1 --filter=blob:none --sparse \
	  https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
	@cd vendor/mujoco_menagerie && git sparse-checkout set leap_hand shadow_hand wonik_allegro
	@echo "menagerie ready"

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
