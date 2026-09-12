"""LEAP grasps the block and lifts it -- no table.

Every failed lift so far jammed the hand into a table (378 N of hand-table
contact, base commanded 15 cm and reaching 1 cm). A floating-hand bench does not
need one: the block is pinned in mid-air at the planned grasp point, the fingers
close on it, the pin is released, and then the hand is raised. If the block is
gripped it comes up with the hand; if it is not, it falls. That is the whole
test, and it is strictly harder than lifting off a table because nothing ever
supports the block.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from leap_pinch import Pinch, quat_x_to


def pick(b, plan, f_target, lift_h=0.15, n_way=15, settle_s=1.0, hold_s=2.0):
    c = plan["centre"].copy()
    quat = quat_x_to(plan["axis"])
    mujoco.mj_resetData(b.m, b.d)
    b.d.qpos[b.qadr] = plan["q_open"]
    b.d.qpos[b.bz_q] = 0.0
    b.d.qpos[b.oq:b.oq + 3] = c
    b.d.qpos[b.oq + 3:b.oq + 7] = quat
    b.d.ctrl[b.aidx] = plan["q_open"]; b.d.ctrl[b.bz_a] = 0.0
    mujoco.mj_forward(b.m, b.d)
    b._pq = b.d.qpos[b.oq:b.oq + 7].copy()
    b.step(plan["q_open"], 0.0, 25, pin=True)
    ht = b.hand_table_force()
    q = b.seat(plan, f_target)
    g = b.metrics()

    b.step(q, 0.0, int(settle_s / b.m.opt.timestep))        # RELEASE, mid-air
    z_rel = float(b.d.qpos[b.oq + 2])
    held_static = b.metrics()["n_contacts"] > 0 and abs(z_rel - c[2]) < 0.01

    z0 = float(b.d.qpos[b.oq + 2])
    rel0 = z0 - float(b.d.qpos[b.bz_q])
    for i in range(1, n_way + 1):
        b.step(q, lift_h * i / n_way, 40)
    b.step(q, lift_h, int(hold_s / b.m.opt.timestep))
    zf = float(b.d.qpos[b.oq + 2])
    after = b.metrics()
    slip = abs((zf - float(b.d.qpos[b.bz_q])) - rel0)
    return dict(eps=g["epsilon"], n=g["n_contacts"], f_total=g["f_total"],
                delta=g["delta"], hand_table_N=float(ht),
                held_static=bool(held_static), settle_drop_m=float(c[2] - z_rel),
                base_actual=float(b.d.qpos[b.bz_q]), net_lift_m=float(zf - z0),
                n_after=after["n_contacts"], eps_after=after["epsilon"],
                slip_m=float(slip),
                success=bool(zf - z0 >= 0.10 and after["n_contacts"] > 0
                             and slip < 0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--out", default="results/leap_pick_midair.json")
    a = ap.parse_args()
    t0 = time.time(); rows = []; best = None
    print(f"{'w(cm)':>7}{'h(cm)':>7}{'marg':>6}{'ftgt':>6}{'HT(N)':>7}{'ncon':>6}"
          f"{'F(N)':>8}{'eps':>8}{'static':>8}{'base':>8}{'lift(cm)':>10}"
          f"{'n_end':>7}{'slip(cm)':>9}{'ok':>6}")
    for w in (0.030, 0.040, 0.050):
        for hgt in (0.05, 0.07):
            for marg in (0.004, 0.008):
                for ftgt in (4.0, 8.0):
                    # table pushed far below: it exists only so the model is
                    # unchanged from leap_pinch; it never touches anything
                    b = Pinch(width=w, height=hgt, mass=a.mass, table_z=-0.60)
                    pl = b.plan(margin=marg)
                    r = pick(b, pl, ftgt)
                    r.update(w=w, h=hgt, marg=marg, ftgt=ftgt)
                    rows.append(r)
                    print(f"{w*100:>7.1f}{hgt*100:>7.1f}{marg*1000:>6.0f}{ftgt:>6.1f}"
                          f"{r['hand_table_N']:>7.1f}{r['n']:>6}{r['f_total']:>8.2f}"
                          f"{r['eps']:>8.4f}{str(r['held_static']):>8}"
                          f"{r['base_actual']*100:>8.2f}{r['net_lift_m']*100:>10.2f}"
                          f"{r['n_after']:>7}{r['slip_m']*100:>9.2f}"
                          f"{str(r['success']):>6}", flush=True)
                    if best is None or (r["success"], r["net_lift_m"]) > \
                            (best["success"], best["net_lift_m"]):
                        best = r
    Path(a.out).write_text(json.dumps(dict(rows=rows, best=best), indent=2))
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}  ({time.time()-t0:.0f} s)")
    print("BEST:", json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
