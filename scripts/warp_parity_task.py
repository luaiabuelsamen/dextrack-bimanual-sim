"""Outcome-level parity: does the TASK survive on the GPU?

Step-level agreement between CPU MuJoCo (float64) and warp (float32) decays
chaotically in a contact-rich scene -- measured at 3.3e-7 through step 50 and
3.1e-1 by step 200. Chasing that number is the wrong test. What licenses
training in warp and evaluating on CPU is whether the OUTCOME is preserved:
the two-handed expert extracting the peg and the one-handed control failing.

The expert is effectively open-loop -- its poses are calibrated up front, not
fed back -- so its control sequence can be recorded on CPU and replayed
verbatim through warp. If the outcome holds under that replay it holds for the
same reason a policy's would.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def record_expert(two_handed=True):
    """Run the expert on CPU, capturing every ctrl it applies."""
    from bimanual_expert import Expert
    ex = Expert(two_handed=two_handed)
    seq = []
    orig = mujoco.mj_step

    def patched(m, d, nstep=1):
        seq.append(d.ctrl.copy())
        orig(m, d, nstep)

    mujoco.mj_step = patched
    try:
        res = ex.run(verbose=False)
    finally:
        mujoco.mj_step = orig
    return ex, np.array(seq), res


def replay_warp(model, ctrl_seq, nworld=1, nconmax=256, njmax=512):
    import warp as wp
    import mujoco_warp as mjw
    import warp_fix
    warp_fix.apply(verbose=False)
    wp.init()

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    mw = mjw.put_model(model)
    dw = mjw.put_data(model, d, nworld=nworld, nconmax=nconmax, njmax=njmax)
    n = len(ctrl_seq)
    for c in ctrl_seq:
        dw.ctrl.assign(np.tile(np.asarray(c, np.float32), (nworld, 1)))
        mjw.step(mw, dw)
    wp.synchronize()
    return dw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=1)
    ap.add_argument("--out", default="results/warp_parity_task.json")
    a = ap.parse_args()
    rows = []
    for two in (True, False):
        label = "two-handed expert" if two else "one-handed control"
        ex, seq, cpu_res = record_expert(two)
        e = ex.e
        m = e.m
        print(f"\n=== {label} === ({len(seq)} control steps recorded on CPU)")
        print(f"  CPU : peg out {cpu_res['peg_out_m']*100:6.2f} cm  "
              f"base moved {cpu_res['box_disp_m']*100:5.2f} cm  "
              f"base z {cpu_res['box_z_rise_m']*100:+5.2f} cm  "
              f"ok={cpu_res['success']}", flush=True)

        t0 = time.time()
        dw = replay_warp(m, seq, nworld=a.nworld)
        qpos = dw.qpos.numpy()[0]
        peg_out = float(qpos[e.hinge_q]) - ex.out0
        box = qpos[e.box_q:e.box_q + 3]
        box0 = np.array([0.0, 0.0, 0.060])       # BASE_HALF[2] = 0.025 -> set below
        from bimanual_env import BASE_HALF
        box0 = np.array([0.0, 0.0, BASE_HALF[2]])
        disp = float(np.linalg.norm(box - box0))
        ok = bool(peg_out >= 0.08 and disp < 0.02)
        print(f"  WARP: peg out {peg_out*100:6.2f} cm  base moved {disp*100:5.2f} cm  "
              f"base z {(box[2]-box0[2])*100:+5.2f} cm  ok={ok}   "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(dict(condition=label, cpu=cpu_res,
                         warp=dict(peg_out_m=peg_out, box_disp_m=disp,
                                   box_z_rise_m=float(box[2] - box0[2]),
                                   success=ok)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2, default=float))
    two, one = rows
    agree = (two["cpu"]["success"] == two["warp"]["success"] and
             one["cpu"]["success"] == one["warp"]["success"])
    print("\n" + "=" * 68)
    print(f"{'condition':22}{'CPU ok':>9}{'WARP ok':>10}{'CPU peg':>10}{'WARP peg':>11}")
    for r in rows:
        print(f"{r['condition']:22}{str(r['cpu']['success']):>9}"
              f"{str(r['warp']['success']):>10}"
              f"{r['cpu']['peg_out_m']*100:>10.2f}{r['warp']['peg_out_m']*100:>11.2f}")
    print("=" * 68)
    print("OUTCOME PARITY HOLDS: warp reproduces both verdicts."
          if agree else
          "OUTCOME PARITY FAILS: the GPU scene does not reproduce the task verdicts, "
          "so training there cannot be evaluated on CPU without a stated gap.")


if __name__ == "__main__":
    main()
