# opposition-deficit

[![tests](https://img.shields.io/badge/tests-13%20passing-2a78d6)](tests/)
[![license](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)

**Question.** When a robot hand cannot oppose the way a human hand does, what
should human hand-object interaction data be retargeted *to*?

`NOTES.md` is the measurement log and the record of every retraction. Read it
before quoting any number.

## Results

**The opposition floor predicts what a hand can hold.** Four hands, one derived
closure procedure, one shared set of block widths. Same ordering on both axes.

| hand | closest thumb-finger gap | widths feasible | **widths held** |
|---|---|---|---|
| Shadow (24 DoF) | 0.25 cm | 8 | **7** |
| LEAP (16 DoF) | 0.80 cm | 8 | **5** |
| Allegro (16 DoF) | 2.41 cm | 7 | **3** |
| Dexmate f5d6 (11 DoF) | 3.40 cm | 4 | **0** |

f5d6 is the only hand that cannot oppose, and the only one that holds nothing --
not even the four widths it can nominally close on. Its uncapped floor, 3.08 cm,
replicates 3.1 cm measured months earlier by a completely different method.
Figure: `figures/16_opposition_axis.png`.

**A bimanual task that provably needs two hands.** Two LEAP hands; a peg whose
socket friction (1.20 N) exceeds the base weight (0.78 N), so a one-handed pull
lifts the base instead of extracting anything.

| | peg out | base lift | success |
|---|---|---|---|
| two-handed expert | 12.98 cm | +0.57 cm | yes |
| one-handed control | 9.07 cm | **+6.10 cm** | no |

6/6 across socket-friction and base-mass settings. The control is run, not assumed.

**GPU physics locally, without degrading the scene.** JAX CUDA works on this
Jetson. The port strips only *visual* geoms -- proven bit-identical over 400
steps -- and leaves collision geometry and the solver untouched. 2,655
env-steps/s at 256 worlds through `mujoco_warp`.

**Learned policy.** BC from 20 randomised expert demos: 67% +/- 47% against 0%
for do-nothing and random. It does **not** beat the expert under perception
error (61% +/- 21% vs 50%, inside the spread) -- see NOTES for why the noise
augmentation was self-defeating.

## Layout

    src/oppdef/          library code, no CLI, no sys.path games
      paths.py           every external path, in one place
      metrics/epsilon.py Ferrari-Canny epsilon over MuJoCo contacts
      hands/             hand specs (closures DERIVED) and the opposition axis
      envs/              bimanual peg task, single-hand grasp benches
      control/           scripted expert, MPPI
      sim/               MJX/warp port and the upstream warp workaround
      learning/          behaviour cloning
      viz/               renderers
    experiments/         every runnable thing; `python -m experiments.<name>`
    tests/               invariants -- each one is a bug that actually happened
    attic/               superseded code, with a table saying why
    results/             live results; results/retracted/ may NOT be quoted
    docs/                goal, plan, protocol, next steps
    NOTES.md             the measurement log -- the only authoritative numbers

## Setup

    make vendor       # MuJoCo Menagerie hand models (sparse clone, ~30 MB)
    make install      # editable install
    make test         # 13 invariants, ~10 s

Three of the four hands (Shadow, Allegro, LEAP) come from Menagerie and need
nothing else. The Dexmate f5d6 rows additionally need that robot's URDF, which
is not redistributed here:

    export OPPDEF_VEGA_URDF=/path/to/vega_1u_f5d6-obj.urdf
    export OPPDEF_DEXTRACK=/path/to/a/urdf-to-mjcf-compiler   # strips .glb visuals

GPU work is optional (`pip install -e '.[gpu]'`) and goes through `mjx_env.sh`,
which puts the venv's own CUDA libraries on `LD_LIBRARY_PATH` -- the one thing
JAX needs to find a Jetson's GPU.

## Running

    make install      # editable install
    make test-fast    # invariants, seconds
    make test         # + the claim tests (physics, ~10 s)
    make axis         # opposition floor, four hands
    make hand-axis    # which widths each hand holds
    make expert       # bimanual expert + one-handed control
    make bc           # collect, train, evaluate
    make parity       # CPU vs warp, outcome level (GPU)
    make figures

## House rules these follow

Trivial baselines every time (do-nothing, random, scripted expert). Three seeds
with the spread. Success criteria fixed before the run. A 0/N is a defect until
paired with a working reference under the identical protocol. Render a frame
before sweeping. Everything validated locally; Modal is for scale, not access.
