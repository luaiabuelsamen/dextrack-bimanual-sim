# What is actually built, and the one next step

## What survived

1,065 live lines of 4,882 written. Five files:

| file | what it is | status |
|---|---|---|
| `analysis/epsilon.py` | Ferrari-Canny epsilon over MuJoCo contacts | passes an analytic self-test; superseded in principle by DexGraspBench but small and validated |
| `scripts/opposition_axis.py` | the opposition floor and aperture, four hands | **the project's spine.** f5d6 3.08 cm independently replicates the 3.1 cm measured in May |
| `scripts/grasp_bench.py` | pre-grasp/grasp/squeeze bench, feasibility + clean-start checks | LEAP holds and lifts 0.05 kg cleanly |
| `scripts/mppi_grasp.py` | sampling MPC on `mujoco.rollout`, plus its no-planner baseline | cuts peak grip force ~15x; refines a grasp, does not discover one |
| `scripts/render_axis.py` | floor-pose renders | the visual behind the axis |

Everything else is in `scripts/attic/` with a table saying what each was and why
it died. Several of those failures are cited by name in NOTES.md and are the
reason the live code is shaped the way it is; they are archived, not deleted.

## What is NOT built

Nothing here has learned anything. There is no policy, no training run, no
demonstration data, and no bimanual task that requires two hands. The only
learning infrastructure in the wider project is `dextrack_vega`'s MJX/brax setup
on Modal (~104k env-steps/s on an A10G), and it is wired to the f5d6 scene,
which this project has now shown cannot grasp.

## The one next step

**Build a single bimanual environment with hands that can actually grasp, and a
task that cannot be done with one hand -- and prove a scripted expert solves it
before any learning is attempted.**

Concretely, in order, with a kill criterion each:

1. **Two LEAP hands on floating 6-DoF bases.** Not on the Vega arms. The arm's
   reachable-orientation set is tight and already measured (90 deg reorientations
   unreachable, no vertical headroom), and it would confound every result with a
   reachability limit that is a property of the robot, not the hands. Add arms
   back later if hardware demands it.
   *Kill:* if two LEAP hands cannot be posed facing each other with overlapping
   workspaces, the base parameterisation is wrong.

2. **A role-asymmetric task.** One hand stabilises, the other actuates: a hinged
   lid, a jar, scissors. Two hands doing the same thing to opposite faces is a
   gripper with extra steps -- that is what the Vega squeeze was, and it is not
   bimanual manipulation. ARCTIC's objects are articulated and are exactly this
   shape, which also lines the task up with the retargeting question.
   *Kill:* if the task is solvable one-handed, it is the wrong task.

3. **Scripted expert first.** Same rule as everywhere else in this project: a
   0/N means nothing without a working reference under the identical protocol.
   The expert also becomes the BC warm start.
   *Kill:* no expert, no learning run.

4. **Port that scene to MJX and confirm throughput.** The asset constraints are
   already documented in `mjx-modal-port`: no cylinder-box collision, mesh
   collision compiles pathologically slowly, needs armature and damping for
   stability, and `done` must be true termination only.
   *Kill:* if MJX parity with the mesh scene is worse than the sign of the
   result, train on CPU and say so.

5. **Then learn.** BC from the expert, then RL residual on top -- the recipe that
   already worked for push in `dextrack_vega`.

## Why this order

The four months before today trained RL against a scene where the object spawned
4 cm inside the table and a hand that cannot oppose. The learning was never the
bottleneck. Every hour spent validating the environment before training is
repaid several times over, and the cheapest validation is a scripted expert and
a rendered frame.
