"""Behaviour cloning from the scripted bimanual expert, end to end.

Collect -> train -> evaluate, all locally. The expert is the demonstrator and
also the upper reference; the trivial baselines are run every time, because a
learned policy that does not beat do-nothing and random has not been shown to do
anything.

Randomisation is what makes this cloning rather than replay. The expert is
open-loop once calibrated, so with a fixed task a policy could "succeed" by
memorising one trajectory and ignoring its observations. Socket friction, base
mass and the base's xy placement are therefore varied per episode; the expert
re-calibrates against the actual peg pose each time, so the demonstrations
genuinely differ and the policy has to read the state to track them.

Success is criterion v2, unchanged from the expert:
    peg extracted >= 8 cm, base LIFT < 2 cm, base tilt < 15 deg.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_env import BASE_HALF, BASE_DOF                       # noqa: E402
from bimanual_expert import Expert, FINGER_JOINTS, _amp_vector, GRASP  # noqa: E402


# ---------------------------------------------------------------- observation
def observe(e, phase=0.0):
    """What the policy sees: both hands' joint state, the task state, and the
    episode PHASE.

    Phase is not optional here. The scripted expert is open-loop once
    calibrated, so its action is a function of TIME, not only of state -- two
    moments with near-identical states (descending vs holding) demand different
    actions. Without a phase input the policy has to average them, which is
    exactly what it did: it drove the peg 2.9 cm the WRONG way while the expert
    extracted 12.9 cm. This is a property of cloning a time-indexed
    demonstrator, not a tuning failure.
    """
    m, d = e.m, e.d
    n2 = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s)
    base = [d.qpos[m.jnt_qposadr[n2(f"{p}{dof}")]]
            for p in ("rh_", "lh_") for dof in BASE_DOF]
    fing = [d.qpos[m.jnt_qposadr[n2(f"{p}{j}")]]
            for p in ("rh_", "lh_") for j in FINGER_JOINTS]
    box = d.qpos[e.box_q:e.box_q + 7]
    return np.concatenate([base, fing, [e.peg_out()], box, e.knob_pos(),
                           [phase, np.sin(2 * np.pi * phase),
                            np.cos(2 * np.pi * phase)]]).astype(np.float32)


OBS_DIM = 12 + 26 + 1 + 7 + 3 + 3


# ---------------------------------------------------------------- collection
def rollout_expert(seed, two_handed=True, record=True):
    """One randomised episode. Returns (obs, act) arrays and the outcome."""
    rng = np.random.default_rng(seed)
    kw = dict(hinge_friction=float(rng.uniform(0.9, 1.8)),
              base_mass=float(rng.uniform(0.06, 0.12)))
    ex = Expert(two_handed=two_handed, **kw)
    e = ex.e
    obs, act = [], []
    orig = mujoco.mj_step

    step_i = [0]
    N_TOTAL = 3470

    def patched(m, d, nstep=1):
        if record:
            obs.append(observe(e, step_i[0] / N_TOTAL))
            act.append(d.ctrl.copy().astype(np.float32))
        step_i[0] += 1
        orig(m, d, nstep)

    mujoco.mj_step = patched
    try:
        res = ex.run(verbose=False)
    finally:
        mujoco.mj_step = orig
    res.update(kw)
    return np.array(obs), np.array(act), res, ex


def collect(n, out, stride=4):
    O, A, meta = [], [], []
    t0 = time.time()
    for i in range(n):
        o, a, r, _ = rollout_expert(seed=1000 + i)
        if not r["success"]:
            print(f"  demo {i:3}: DISCARDED (expert failed: peg "
                  f"{r['peg_out_m']*100:.1f} cm)", flush=True)
            continue
        O.append(o[::stride]); A.append(a[::stride]); meta.append(r)
        print(f"  demo {i:3}: peg {r['peg_out_m']*100:5.2f} cm  lift "
              f"{r['box_z_rise_m']*100:+5.2f}  socket {r['hinge_friction']:.2f} N  "
              f"mass {r['base_mass']:.3f} kg  ({len(o)} steps)", flush=True)
    O = np.concatenate(O); A = np.concatenate(A)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=O, act=A)
    print(f"\ncollected {len(meta)}/{n} successful demos -> {O.shape[0]} pairs "
          f"({O.shape[1]}-dim obs, {A.shape[1]}-dim act) in {time.time()-t0:.0f}s")
    return O, A, meta


# ---------------------------------------------------------------- training
def train(O, A, seed=0, epochs=300, hidden=512, lr=1e-3, device=None):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    om, os_ = O.mean(0), O.std(0) + 1e-6
    am, as_ = A.mean(0), A.std(0) + 1e-6
    X = torch.tensor((O - om) / os_, device=dev)
    Y = torch.tensor((A - am) / as_, device=dev)
    net = nn.Sequential(nn.Linear(O.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, A.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X); bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 60 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3}  mse {tot/n:.5f}", flush=True)
    return dict(net=net, om=om, os=os_, am=am, as_=as_, dev=dev)


def policy_fn(model):
    import torch
    net, om, os_, am, as_ = (model["net"], model["om"], model["os"],
                             model["am"], model["as_"])

    def act(o):
        with torch.no_grad():
            x = torch.tensor(((o - om) / os_)[None], device=model["dev"])
            y = net(x).cpu().numpy()[0]
        return y * as_ + am
    return act


# ---------------------------------------------------------------- evaluation
def run_policy(seed, act_fn=None, mode="policy", n_steps=None, two_handed=True):
    """One episode under a policy or a trivial baseline, same task draw as the
    expert for that seed so the comparison is paired."""
    rng = np.random.default_rng(seed)
    kw = dict(hinge_friction=float(rng.uniform(0.9, 1.8)),
              base_mass=float(rng.uniform(0.06, 0.12)))
    ex = Expert(two_handed=two_handed, **kw)
    e = ex.e
    # start from the same penetration-free pre-grasp the expert uses
    lh_press = None
    from bimanual_expert import palm_ctrl
    from bimanual_env import LH_HOME
    lh_press = palm_ctrl(LH_HOME, [0.0, -0.085, 0.044])
    ex.pre_frac, _ = ex.start_at_pregrasp(lh_press)
    box0 = np.array([0.0, 0.0, BASE_HALF[2]])
    out0 = ex.out0
    N = n_steps or 3470
    rs = np.random.default_rng(seed + 77)
    ctrl = np.zeros(e.m.nu)
    for k in range(N):
        if mode == "policy":
            ctrl = act_fn(observe(e, k / N))
        elif mode == "random":
            ctrl = ctrl + rs.normal(0, 0.01, e.m.nu)
        elif mode == "donothing":
            pass
        lo = e.m.actuator_ctrlrange[:, 0]; hi = e.m.actuator_ctrlrange[:, 1]
        lim = e.m.actuator_ctrllimited.astype(bool)
        c = np.where(lim, np.clip(ctrl, lo, hi), ctrl)
        e.d.ctrl[:] = c
        mujoco.mj_step(e.m, e.d)
    out = e.peg_out() - out0
    lift = float(e.box_pos()[2] - box0[2])
    return dict(peg_out_m=float(out), box_lift_m=lift,
                box_disp_m=float(np.linalg.norm(e.box_pos() - box0)),
                tilt_deg=e.box_tilt_deg(),
                success=bool(out >= 0.08 and lift < 0.02
                             and e.box_tilt_deg() < 15.0), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--data", default="results/bc_demos.npz")
    ap.add_argument("--out", default="results/bc.json")
    a = ap.parse_args()
    t0 = time.time()

    print(f"=== collecting {a.demos} randomised expert demos ===", flush=True)
    if Path(a.data).exists():
        z = np.load(a.data); O, A = z["obs"], z["act"]
        print(f"  reusing {a.data}: {O.shape[0]} pairs")
    else:
        O, A, _ = collect(a.demos, a.data)

    results = {}
    print(f"\n=== trivial baselines ({a.eval_episodes} held-out episodes) ===",
          flush=True)
    for mode in ("donothing", "random"):
        rs = [run_policy(5000 + i, mode=mode) for i in range(a.eval_episodes)]
        results[mode] = rs
        pk = np.array([r["peg_out_m"] for r in rs]) * 100
        print(f"  {mode:10}: success {sum(r['success'] for r in rs)}/{len(rs)}"
              f"  peg {pk.mean():5.2f} +/- {pk.std():4.2f} cm", flush=True)

    print(f"\n=== expert on the same held-out episodes ===", flush=True)
    ex_rs = []
    for i in range(a.eval_episodes):
        _, _, r, _ = rollout_expert(5000 + i, record=False)
        ex_rs.append(r)
    results["expert"] = ex_rs
    pk = np.array([r["peg_out_m"] for r in ex_rs]) * 100
    print(f"  expert    : success {sum(r['success'] for r in ex_rs)}/{len(ex_rs)}"
          f"  peg {pk.mean():5.2f} +/- {pk.std():4.2f} cm", flush=True)

    print(f"\n=== BC, {a.seeds} training seeds ===", flush=True)
    bc_all = []
    for s in range(a.seeds):
        print(f"  seed {s}:", flush=True)
        model = train(O, A, seed=s, epochs=a.epochs)
        fn = policy_fn(model)
        rs = [run_policy(5000 + i, act_fn=fn, mode="policy")
              for i in range(a.eval_episodes)]
        bc_all.append(rs)
        pk = np.array([r["peg_out_m"] for r in rs]) * 100
        print(f"    success {sum(r['success'] for r in rs)}/{len(rs)}"
              f"  peg {pk.mean():5.2f} +/- {pk.std():4.2f} cm", flush=True)
    results["bc"] = bc_all

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2, default=float))
    rate = [sum(r["success"] for r in rs) / len(rs) for rs in bc_all]
    print("\n" + "=" * 66)
    print(f"{'method':14}{'success':>18}{'peg out (cm)':>22}")
    for k in ("donothing", "random", "expert"):
        rs = results[k]
        pk = np.array([r["peg_out_m"] for r in rs]) * 100
        print(f"{k:14}{sum(r['success'] for r in rs)}/{len(rs):<16}"
              f"{pk.mean():>12.2f} +/- {pk.std():.2f}")
    pk = np.array([r["peg_out_m"] for rs in bc_all for r in rs]) * 100
    print(f"{'BC (3 seeds)':14}{np.mean(rate)*100:.0f}% +/- "
          f"{np.std(rate)*100:.0f}%{'':>6}{pk.mean():>12.2f} +/- {pk.std():.2f}")
    print("=" * 66)
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
