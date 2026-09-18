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

from handsim.human import grab, track, homotopy


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


def record(rt, actions, start, steps=None, absolute=True) -> Episode:
    """Replay a solved action sequence and log what a policy would have seen.

    The label is the ABSOLUTE command, not MPPI's correction. Distilling the
    correction does not work and the reason is measurable: regressing it on the
    observation gives a linear R^2 of **0.075 in sample**, because a sampling
    optimiser's per-step correction is dominated by its own noise draw rather
    than by the state. A policy trained on it learns the training clips and
    scores worse than a constant on held-out objects (ratio 2.235).

    The absolute command -- palm pose in the object frame, plus finger targets --
    is a function of the state by construction, which is what makes it a
    regression target at all. DexTrack distils an RL policy for the same reason:
    a policy is state-conditioned, a trajectory optimiser's output is not.
    """
    import mujoco

    steps = rt.T - start if steps is None else steps
    rt.reset_at(start)
    O, A, E = [], [], []
    for i in range(steps):
        k = start + i
        O.append(rt.observe(k))
        a = actions[k] if actions is not None else np.zeros(rt.n_action)
        if absolute:
            # what the hand is actually commanded to: the palm pose expressed in
            # the OBJECT frame (so it does not encode where GRAB's subject
            # stood) and the finger servo targets
            op, oq = rt.true_obj_pose()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, oq)
            R = R.reshape(3, 3)
            kk = int(np.clip(k, 0, rt.T - 1))
            pp = R.T @ (rt.P[kk] + a[:3] - op)
            rel = np.zeros(4)
            mujoco.mju_mulQuat(rel, np.array([oq[0], -oq[1], -oq[2], -oq[3]]),
                               rt.Q[kk])
            fing = rt.ctrl_for(rt.vals[kk]) + rt._grip_offset + a[6:]
            A.append(np.concatenate([pp, rel, fing]))
        else:
            A.append(a)
        rt.apply(k, a)
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(rt.sim.model, rt.sim.data)
        E.append(rt.error(k)[0])
    return Episode(seq=rt.seq.name, subject=rt.seq.subject, obj=rt.seq.obj,
                   obs=np.array(O), act=np.array(A), pos_err=np.array(E),
                   start=start, solved_lambda=1.0)


def solve_reference(seq, hand="shadow", side="rhand", levels=(0.5, 1.0),
                    horizon=4, samples=20, grip=8.0, seed=0, verbose=False,
                    synth=True):
    """Solve one reference with grip establishment + homotopy, and record it.

    Grip establishment is not optional: G5 measured a 0.255 hold rate for the
    raw retarget, and that failures make 0.1 contacts at reset against 17.4 for
    successes. Its pre-registered decision rule makes establishment part of this
    stage.
    """
    rt = track.ReferenceTracker(seq, hand=hand, side=side)
    if synth:
        # Without this, 12 of 14 references returned "no graspable frame": the
        # raw retarget holds the object in only ~0.32 of frames (G5), and a
        # reference with no holding frame has nothing to start a tracking
        # episode from. Guarded, so it can only add graspable frames.
        rt.synthesize_grasp()
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
    # A dropped episode is not a demonstration. Recorded without this check,
    # one of two collected episodes had 30957 mm of tracking error -- the
    # object in free fall -- and would have been distilled as if it were a
    # solution.
    if ep.pos_err[-1] > 0.10 or ep.pos_err.mean() > 0.15:
        return None, rt
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
    from handsim.learning.bc import train as bc_train

    z = np.load(npz_path, allow_pickle=True)
    # float32: the observations are built in float64 and torch's Linear is
    # float32, which fails at the first matmul rather than at load
    O, A = z["obs"].astype(np.float32), z["act"].astype(np.float32)
    meta = json.loads(str(z["meta"]))
    _tr, _te, test_objs = split_by_object(meta, holdout_frac, seed)
    is_test = np.array([o in test_objs for o in z["obj"]])
    print(f"{len(O)} transitions; holding out objects {sorted(test_objs)} "
          f"({is_test.sum()} transitions)")
    model = bc_train(O[~is_test], A[~is_test], seed=seed, epochs=epochs,
                     hidden=hidden)
    model["test_objs"] = test_objs
    return model, (O, A, is_test)


def policy_rollout(rt, model, start=None):
    """Run the distilled policy in the simulator and score it on the truth.

    MAE against the recorded command is a proxy: a policy can score well on it
    and still lose the object, because the errors that matter are the ones that
    break contact. This closes the loop -- the network's output IS the command,
    and the object's tracking error is the number.

    The policy emits the palm pose in the OBJECT frame plus the finger targets,
    so the world command is recovered by composing with the object's current
    pose. That composition is what makes the policy object-relative rather than
    tied to where GRAB's subject stood.
    """
    import mujoco
    from handsim.learning.bc import policy_fn
    from handsim.human import track as T

    act = policy_fn(model)
    if start is None:
        gf = rt.grasp_frames()
        start = int(gf[0]) if len(gf) else 0
    rt.reset_at(start)
    m, d = rt.sim.model, rt.sim.data
    errs = []
    for k in range(start, rt.T):
        y = act(rt.observe(k).astype(np.float32))
        op, oq = rt.true_obj_pose()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, oq)
        R = R.reshape(3, 3)
        palm_p = R @ y[:3] + op
        q = y[3:7] / max(np.linalg.norm(y[3:7]), 1e-9)
        palm_q = np.zeros(4)
        mujoco.mju_mulQuat(palm_q, oq, q)
        rt.mh.command(palm_p, palm_q)
        c = np.asarray(y[7:], float)
        d.ctrl[:] = np.clip(c, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(m, d)
        errs.append(rt.error(k)[0])
    return np.array(errs)


def expert_label(rt, k):
    """The expert's command at reference frame k, in the policy's output space.

    The expert here is the feedforward, which is a function of the frame index
    alone. That is what makes DAgger cheap in this setting: relabelling a state
    the policy wandered into costs one forward-kinematics call, not a fresh run
    of the trajectory optimiser. The expensive part of DAgger is usually the
    expert query, and here there isn't one.
    """
    import mujoco

    k = int(np.clip(k, 0, rt.T - 1))
    op, oq = rt.true_obj_pose()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, oq)
    R = R.reshape(3, 3)
    pp = R.T @ (rt.P[k] - op)
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, np.array([oq[0], -oq[1], -oq[2], -oq[3]]), rt.Q[k])
    fing = rt.ctrl_for(rt.vals[k]) + rt._grip_offset
    return np.concatenate([pp, rel, fing])


def mppi_expert_label(rt, k, horizon=3, samples=12, sigma_pos=0.004,
                      sigma_rot=0.03, sigma_fin=0.05, rho=1.0, rng=None):
    """A CLOSED-LOOP expert label: the feedforward plus an MPPI correction
    optimised from the state the policy actually reached.

    This is the piece DAgger needs and the feedforward cannot supply. The
    feedforward is a function of the frame index alone, so from a drifted state
    it recommends exactly what it would recommend from a good one -- relabelling
    with it taught the policy that no recovery is needed, and every DAgger round
    got worse (camera 1096 -> 1330 -> 1848 mm). MPPI re-optimises here, from
    this state, so its label says how to get BACK.

    One label costs one short MPPI solve: horizon x samples x ctrl_every MuJoCo
    steps, about 0.1 s. That is the honest price of a closed-loop expert, and it
    is why the cheap version was tried first.
    """
    import mujoco

    rng = rng or np.random.default_rng(0)
    n_fin = rt.n_action - 6
    scale = np.concatenate([np.full(3, sigma_pos), np.full(3, sigma_rot),
                            np.full(n_fin, sigma_fin)])
    state = rt.save()
    cand = rng.normal(size=(samples, horizon, rt.n_action)) * scale
    cand[0] = 0.0
    cost = np.empty(samples)
    for i in range(samples):
        rt.restore(state)
        c = 0.0
        for h in range(horizon):
            rt.apply(k + h, cand[i, h])
            for _ in range(rt.ctrl_every):
                mujoco.mj_step(rt.sim.model, rt.sim.data)
            pe, re = rt.error(k + h)
            c += pe + 0.1 * re + (2.0 if pe > 0.10 else 0.0)
        cost[i] = c
    lam = max(rho * float(np.std(cost)), 1e-6)
    w = np.exp(-(cost - cost.min()) / lam)
    w /= w.sum()
    best = np.einsum("i,iha->ha", w, cand)[0]
    rt.restore(state)
    lab = expert_label(rt, k)
    # fold the correction into the absolute command the policy emits
    lab[:3] = lab[:3] + best[:3]
    lab[7:] = lab[7:] + best[6:]
    return lab, best


def dagger(refs, rounds=3, epochs=200, holdout_frac=0.3, seed=0,
           hand="shadow", beta0=1.0, verbose=True, expert="feedforward",
           mppi_samples=12, mppi_horizon=3, beta_decay=0.5,
           keep_below_m=None):
    """Iterate behaviour cloning on the states the POLICY visits.

    Plain cloning failed in the loop and the reason is standard: the training
    set contains only states the expert visited, so the first command error
    takes the policy somewhere it has never seen and nothing after that is
    in distribution. Measured, the cloned policy dropped the object on every
    held-out reference while the expert it cloned tracked them to 18-31 mm.

    Each round rolls the current policy out, relabels every state it actually
    reached with the expert's command there, and retrains on the union. Round 0
    is ordinary cloning, so any improvement afterwards is attributable to the
    on-policy states rather than to more data in general.
    """
    import mujoco
    from handsim.human import grab, track
    from handsim.learning.bc import train as bc_train, policy_fn

    built = []
    for r in refs:
        seq = grab.load(f"{r['subject']}/{r['seq']}.npz", verts=True, stride=8)
        rt = track.ReferenceTracker(seq, hand=hand)
        rt.synthesize_grasp()
        gf = rt.grasp_frames()
        if len(gf) == 0:
            continue
        built.append((rt, int(gf[0]), r["object"]))
    if not built:
        return None, []

    objs = sorted({o for _rt, _k, o in built})
    rng = np.random.default_rng(seed)
    rng.shuffle(objs)
    n_test = max(1, int(round(holdout_frac * len(objs))))
    test = set(objs[:n_test])
    train_set = [(rt, k, o) for rt, k, o in built if o not in test]
    test_set = [(rt, k, o) for rt, k, o in built if o in test]
    if verbose:
        print(f"{len(built)} references, holding out objects {sorted(test)}",
              flush=True)

    O, A = [], []
    model, hist = None, []
    for rnd in range(rounds):
        for rt, k0, _o in train_set:
            rt.reset_at(k0)
            # Geometric decay, as DAgger specifies. Dropping straight to 0
            # after round 0 let the untrained policy drive the whole episode,
            # so the dataset filled with states where the object was ALREADY
            # lost -- which no expert can recover and which therefore teach
            # nothing. Both expert variants collapsed at round 2 that way
            # (camera 1399 -> 2206 mm).
            beta = beta0 * (beta_decay ** rnd)
            act = policy_fn(model) if model is not None else None
            for k in range(k0, rt.T):
                o = rt.observe(k)
                # Filter only the states the POLICY drove into. Applying this
                # in round 0 as well removed the states the EXPERT visits late
                # in an episode, where its own error naturally grows, and left
                # a policy that had never seen the end of a trajectory: held-out
                # error went from 523 mm to 17994 mm on a 20% smaller dataset.
                keep = (rnd == 0 or keep_below_m is None
                        or rt.error(k)[0] < keep_below_m)
                if keep:
                    O.append(o)
                if keep and expert == "mppi" and rnd > 0:
                    lab, _corr = mppi_expert_label(
                        rt, k, horizon=mppi_horizon, samples=mppi_samples,
                        rng=rng)
                    A.append(lab)
                elif keep:
                    A.append(expert_label(rt, k))
                if act is None or rng.random() < beta:
                    rt.apply(k)                      # expert drives
                else:
                    y = act(o.astype(np.float32))    # the policy drives
                    op, oq = rt.true_obj_pose()
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, oq)
                    R = R.reshape(3, 3)
                    pw = R @ y[:3] + op
                    qn = y[3:7] / max(np.linalg.norm(y[3:7]), 1e-9)
                    qw = np.zeros(4)
                    mujoco.mju_mulQuat(qw, oq, qn)
                    rt.mh.command(pw, qw)
                    rt.sim.data.ctrl[:] = np.clip(
                        y[7:], rt.sim.model.actuator_ctrlrange[:, 0],
                        rt.sim.model.actuator_ctrlrange[:, 1])
                for _ in range(rt.ctrl_every):
                    mujoco.mj_step(rt.sim.model, rt.sim.data)

        Oa = np.array(O, np.float32)
        Aa = np.array(A, np.float32)
        model = bc_train(Oa, Aa, seed=seed, epochs=epochs)
        scores = []
        for rt, k0, o in test_set:
            pe = policy_rollout(rt, model, start=k0)
            scores.append((o, float(pe.mean())))
        hist.append({"round": rnd, "n": len(Oa),
                     "held_out": {o: round(v * 1000, 1) for o, v in scores}})
        if verbose:
            print(f"  round {rnd}: {len(Oa)} transitions  held-out "
                  + "  ".join(f"{o} {v*1000:.0f} mm" for o, v in scores),
                  flush=True)
    return model, hist
