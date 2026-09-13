# opposition-deficit

[![tests](https://img.shields.io/badge/tests-13%20passing-2a78d6)](tests/)
[![license](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)

**Question.** When a robot hand cannot oppose the way a human hand does, what
should human hand-object interaction data be retargeted *to*?

`NOTES.md` is the measurement log and the record of every retraction. Read it
before quoting any number.

## Results

**RETRACTED 2026-09-12 — there is no opposition deficit.** The fingertips were
tracked at distal joint origins on *every* hand; the most distal joint of every
finger moves those bodies by 0.000 mm. Re-derived with the fingertip taken from
each distal link's own collision geometry (validated against f5d6's URDF tip
frames to 0.4 mm), with the hand's mimic couplings enforced and self-collision
scoped to the fingers, and measured thumb-to-nearest-fingertip rather than to
their mean:

| hand | corrected floor | previously reported |
|---|---|---|
| Allegro | 0.06 cm | 2.41 cm |
| Shadow | 0.21 cm | 0.25 cm |
| Dexmate f5d6 | **0.26 cm** | **3.40 cm** |
| LEAP | 0.34 cm | 0.80 cm |

All four hands oppose within 2.8 mm of one another, and f5d6 — the hand that
motivated this project — opposes better than LEAP. The "widths held" column was
separately retracted as an invalid fixture (the palm fell 1.05 m during
closure). See NOTES.md; every result that conditioned on a deficit cohort is
withdrawn.

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
