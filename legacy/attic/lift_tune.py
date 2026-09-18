"""Sustained lift: the grip forms and reaches 12.6 cm, then loses contact.

Hypothesis: as the hands rise the arm runs out of vertical headroom (measured
previously on this robot), the IK cannot hold the commanded y-separation, and
the squeeze relaxes. Test it by (a) logging the IK residual and hand separation
through the lift, and (b) tightening the inset progressively as the hands rise,
so the commanded squeeze compensates for the drift.

Success is the pre-registered criterion: net >= 10 cm AND still held after a 3 s
hold at the top, hands still in contact.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dextrack_vega import config as C
from analysis.epsilon import grasp_metrics
from probe_v2 import build_probe, BOX
from squeeze_sweep import _approach, _settle_grip


def trial(p, inset, sq, lift_h, gain, n_way=10, diag=False):
    centre = np.array(BOX["obj_pos"], float)
    hy = float(p.m.geom_size[p.obj_gid][1])
    ex = (p.table_gid, p.floor_gid)
    gR = centre + np.array([0.0, -(hy - inset), 0.0])
    gL = centre + np.array([0.0, +(hy - inset), 0.0])
    q_app, q_in, f = _approach(p, gR, gL, sq)
    _settle_grip(p, q_app, q_in)
    z0 = p.obj_pos()[2]
    up = np.array([0.0, 0.0, 1.0])
    qq = q_in.copy()
    log = []
    for i in range(1, n_way + 1):
        d = lift_h * i / n_way
        ins = inset + gain * d                      # squeeze harder as we rise
        tR = centre + np.array([0.0, -(hy - ins), 0.0]) + d * up
        tL = centre + np.array([0.0, +(hy - ins), 0.0]) + d * up
        qq[0:7] = p.ik_side_to("R", tR)
        qq[18:25] = p.ik_side_to("L", tL)
        p.drive(qq, 26)
        if diag:
            eR = float(np.linalg.norm(p.ft("R").mean(0) - tR))
            eL = float(np.linalg.norm(p.ft("L").mean(0) - tL))
            sep = float(np.linalg.norm(p.ft("R").mean(0) - p.ft("L").mean(0)))
            g = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
            log.append(dict(i=i, d=d, ik_err_R=eR, ik_err_L=eL, sep=sep,
                            z=float(p.obj_pos()[2] - z0), n=g["n_contacts"],
                            f=g["f_total"], eps=g["epsilon"]))
    zmax = float(p.obj_pos()[2] - z0)
    p.hold_ctrl(qq, 60)                              # 3 s hold at the top
    g = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
    net = float(p.obj_pos()[2] - z0)
    mid = 0.5 * (p.ft("R").mean(0) + p.ft("L").mean(0))
    off = float(np.linalg.norm(p.obj_pos()[:2] - mid[:2]))
    return dict(inset=inset, sq=sq, lift_h=lift_h, gain=gain, net=net,
                peak=max(zmax, net), n_end=g["n_contacts"], eps_end=g["epsilon"],
                offset=off, success=bool(net >= 0.10 and g["n_contacts"] > 0
                                         and off < 0.05), log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="legacy/results/lift_tune.json")
    a = ap.parse_args()
    t0 = time.time()
    p = build_probe(seed=0, **BOX)

    print("--- diagnostic: the 12.6 cm run, instrumented ---")
    d0 = trial(p, 0.012, 1.0, 0.15, 0.0, diag=True)
    print(f"{'way':>4}{'d(cm)':>8}{'ikR(mm)':>9}{'ikL(mm)':>9}{'sep(cm)':>9}"
          f"{'z(cm)':>8}{'ncon':>6}{'F(N)':>9}{'eps':>8}")
    for r in d0["log"]:
        print(f"{r['i']:>4}{r['d']*100:>8.1f}{r['ik_err_R']*1000:>9.1f}"
              f"{r['ik_err_L']*1000:>9.1f}{r['sep']*100:>9.2f}{r['z']*100:>8.2f}"
              f"{r['n']:>6}{r['f']:>9.2f}{r['eps']:>8.4f}")

    print("\n--- sweep: progressive squeeze ---")
    rows = []
    print(f"{'inset(mm)':>10}{'sq':>6}{'lift(cm)':>10}{'gain':>7}"
          f"{'net(cm)':>9}{'peak(cm)':>10}{'ncon':>6}{'ok':>6}")
    for inset in (0.010, 0.012, 0.015):
        for sq in (0.85, 1.0):
            for lift_h in (0.12, 0.15):
                for gain in (0.0, 0.05, 0.10, 0.20):
                    r = trial(p, inset, sq, lift_h, gain)
                    rows.append(r)
                    print(f"{inset*1000:>10.1f}{sq:>6.2f}{lift_h*100:>10.1f}"
                          f"{gain:>7.2f}{r['net']*100:>9.2f}{r['peak']*100:>10.2f}"
                          f"{r['n_end']:>6}{str(r['success']):>6}", flush=True)
    Path(a.out).write_text(json.dumps(dict(diag=d0, rows=rows), indent=2))
    ok = [r for r in rows if r["success"]]
    print(f"\nsuccess {len(ok)}/{len(rows)}   ({time.time()-t0:.0f} s)")
    if ok:
        b = max(ok, key=lambda r: r["net"])
        print("BEST:", {k: v for k, v in b.items() if k != "log"})
    else:
        b = max(rows, key=lambda r: r["net"])
        print("best (no success):", {k: v for k, v in b.items() if k != "log"})


if __name__ == "__main__":
    main()
