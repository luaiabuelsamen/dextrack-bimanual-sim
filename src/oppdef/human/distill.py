"""Stage 5: distil the per-reference solutions into one tracking controller.

Stages 3 and 4 solve references one at a time, and a per-reference solution is
not a controller -- it is a recording. MPPI costs minutes per clip and needs the
simulator in the loop, so it cannot run on a robot and cannot generalise to a
reference it has not already solved.

Distillation turns those solutions into a single feedforward network: the
observation is what the policy could actually measure (proprioception, the
object's state relative to the hand, and a short reference lookahead), and the
target is the correction MPPI chose. This is the step that makes DexTrack a
*controller* rather than a trajectory optimiser.

The evaluation that matters is on references the policy never saw. A policy
scored on its training clips is scored on memorisation, and this repository has
already been caught once by a behaviour-cloning result that open-loop replay
matched exactly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from oppdef.human import grab, track, homotopy


@dataclass
class Episode:
    seq: str
    subject: str
    obj: str
    obs: np.ndarray               # (T, n_obs)
    act: np.ndarray               # (T, n_action)
    pos_err: np.ndarray
    start: int
    solved_lambda: float
    meta: dict = field(default_factory=dict)


def record(rt, actions, start, steps=None) -> Episode:
    """Replay a solved action sequence and log what a policy would have seen."""
    import mujoco

    steps = rt.T - start if steps is None else steps
    rt.reset_at(start)
    O, A, E = [], [], []
    for i in range(steps):
        k = start + i
        O.append(rt.observe(k))
        a = actions[k] if actions is not None else np.zeros(rt.n_action)
        A.append(a)
        rt.apply(k, a)
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(rt.sim.model, rt.sim.data)
        E.append(rt.error(k)[0])
    return Episode(seq=rt.seq.name, subject=rt.seq.subject, obj=rt.seq.obj,
                   obs=np.array(O), act=np.array(A), pos_err=np.array(E),
                   start=start, solved_lambda=1.0)


def solve_reference(seq, hand="shadow", side="rhand", levels=(0.5, 1.0),
                    horizon=4, samples=20, grip=8.0, seed=0, verbose=False):
    """Solve one reference with grip establishment + homotopy, and record it.

    Grip establishment is not optional: G5 measured a 0.255 hold rate for the
    raw retarget, and that failures make 0.1 contacts at reset against 17.4 for
    successes. Its pre-registered decision rule makes establishment part of this
    stage.
    """
    rt = track.ReferenceTracker(seq, hand=hand, side=side)
    gf = rt.grasp_frames()
    if len(gf) == 0:
        return None, rt
    start = int(gf[0])
    res = homotopy.solve_with_curriculum(rt, levels=levels, start=start,
                                         horizon=horizon, samples=samples,
                                         seed=seed, verbose=verbose)
    if res.actions is None:
        return None, rt
    ep = record(rt, res.actions, start)
    ep.solved_lambda = res.solved_lambda
    ep.meta = {"levels": list(levels), "grasp_frames": int(len(gf)),
               "reached_full": bool(res.reached_full)}
    return ep, rt


def collect(rows, hand="shadow", limit=None, out=None, **kw):
    """Solve a list of inventory rows and gather their episodes."""
    eps = []
    for i, r in enumerate(rows[:limit]):
        t0 = time.time()
        try:
            seq = grab.load(f"{r['subject']}/{r['seq']}.npz", verts=True, stride=8)
            ep, _rt = solve_reference(seq, hand=hand, **kw)
        except Exception as e:                            # noqa: BLE001
            print(f"[{i+1}] {r['seq']}: FAILED {e!r}", flush=True)
            continue
        if ep is None:
            print(f"[{i+1}] {r['seq']}: no graspable frame", flush=True)
            continue
        eps.append(ep)
        print(f"[{i+1}/{len(rows[:limit])}] {ep.seq:30s} {ep.obj:12s} "
              f"lambda {ep.solved_lambda:.2f}  {len(ep.obs):3d} steps  "
              f"err {ep.pos_err.mean()*1000:7.1f} mm  ({time.time()-t0:.0f}s)",
              flush=True)
        if out:
            save(eps, out)
    return eps


def save(eps, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.concatenate([e.obs for e in eps]),
        act=np.concatenate([e.act for e in eps]),
        seq=np.array([e.seq for e in eps for _ in range(len(e.obs))]),
        obj=np.array([e.obj for e in eps for _ in range(len(e.obs))]),
        err=np.concatenate([e.pos_err for e in eps]),
        meta=json.dumps([{"seq": e.seq, "obj": e.obj, "subject": e.subject,
                          "start": e.start, "steps": int(len(e.obs)),
                          "solved_lambda": e.solved_lambda,
                          "mean_mm": float(e.pos_err.mean() * 1000)}
                         for e in eps]))
    return path


def split_by_object(meta, holdout_frac=0.3, seed=0):
    """Hold out whole OBJECTS, not random frames.

    Frames from one clip are near-duplicates, and clips of the same object share
    its geometry. Splitting by frame measures interpolation inside a trajectory;
    splitting by object measures what the policy is claimed to do.
    """
    objs = sorted({m["obj"] for m in meta})
    rng = np.random.default_rng(seed)
    rng.shuffle(objs)
    n = max(1, int(round(holdout_frac * len(objs))))
    test = set(objs[:n])
    return [m for m in meta if m["obj"] not in test], [m for m in meta if m["obj"] in test], test


def train(npz_path, holdout_frac=0.3, seed=0, epochs=300, hidden=512):
    """Train one tracking policy, held out by object."""
    from oppdef.learning.bc import train as bc_train

    z = np.load(npz_path, allow_pickle=True)
    O, A = z["obs"], z["act"]
    meta = json.loads(str(z["meta"]))
    _tr, _te, test_objs = split_by_object(meta, holdout_frac, seed)
    is_test = np.array([o in test_objs for o in z["obj"]])
    print(f"{len(O)} transitions; holding out objects {sorted(test_objs)} "
          f"({is_test.sum()} transitions)")
    model = bc_train(O[~is_test], A[~is_test], seed=seed, epochs=epochs,
                     hidden=hidden)
    model["test_objs"] = test_objs
    return model, (O, A, is_test)
