"""The one physically plausible pick on this platform, measured properly.

Two f5d6 hands press the +-y faces of the box. Holding 0.05 kg by friction with
mu = 2.0 needs only ~0.12 N of normal force per side, so the question is not
whether there is enough friction -- it is whether a kp=600 position-controlled
arm can be commanded to apply a SMALL force instead of an enormous one.

So sweep the one parameter that sets it: how far past the face surface each
hand's fingertip-centroid target is placed. Report, for each offset, the
measured normal force, epsilon, and whether the box is still held after the
table is taken away. That curve is the grip margin, and its left edge is the
answer to "can this platform pick the block at all".

Scene: table lowered to 0.71 so the box rests at its original, reachability-
characterised centre height of 0.80 (see NOTES.md, the spawn defect).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco

sys.path.insert(0, "/home/jetson3/projects/dextrack_vega")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dextrack_vega import config as C
from analysis.epsilon import grasp_metrics
from probe_m15 import OPEN, SQUEEZE, lerp_fingers
from probe_v2 import build_probe, BOX


def _approach(p, gR, gL, sq, standoff=0.07):
    """Solve ALL the IK first, then pose the hands at the standoff, then squeeze
    in. Order matters: an earlier version reset the sim after posing the hands,
    so every trial drove from the home pose straight through the box and batted
    it off the table (displacement 8-33 cm at every inset)."""
    f = lerp_fingers(OPEN, SQUEEZE, sq)
    p.reset()
    aR0 = p.ik_side_to("R", gR + np.array([0, -standoff, 0]))
    aL0 = p.ik_side_to("L", gL + np.array([0, +standoff, 0]))
    aR = p.ik_side_to("R", gR)
    aL = p.ik_side_to("L", gL)
    q_app = p.q36(); q_app[0:7], q_app[18:25] = aR0, aL0
    q_app = p.fingers36(q_app, "R", f); q_app = p.fingers36(q_app, "L", f)
    q_in = q_app.copy(); q_in[0:7], q_in[18:25] = aR, aL
    return q_app, q_in, f


def _settle_grip(p, q_app, q_in):
    p.reset()
    p.set_q36(q_app)
    p.hold_ctrl(q_app, 20)
    p.drive(q_in, 60)
    p.hold_ctrl(q_in, 30)


def run(p, inset, sq, dz, hold_s=2.0, lift_h=0.15):
    """inset: metres past the face surface that each fingertip centroid targets."""
    centre = np.array(BOX["obj_pos"], dtype=float)
    hy = float(p.m.geom_size[p.obj_gid][1])
    gR = centre + np.array([0.0, -(hy - inset), dz])
    gL = centre + np.array([0.0, +(hy - inset), dz])
    ex = (p.table_gid, p.floor_gid)

    q_app, q_in, f = _approach(p, gR, gL, sq)
    _settle_grip(p, q_app, q_in)

    g = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
    disp = float(np.linalg.norm(p.obj_pos() - centre))

    # hold test: take the table away.
    # `held` requires the box to stay put AND still be in the hands. Drop alone
    # is degenerate -- a box lying on the floor has drop < 1 cm forever, which is
    # exactly how an earlier version scored 3/48 "held" at 1.35 m displacement.
    z0 = p.obj_pos()[2]
    p.m.geom_contype[p.table_gid] = 0; p.m.geom_conaffinity[p.table_gid] = 0
    mujoco.mj_forward(p.m, p.d)
    p.hold_ctrl(q_in, int(hold_s / p.env.dt))
    drop = float(z0 - p.obj_pos()[2])
    after = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
    disp_after = float(np.linalg.norm(p.obj_pos() - centre))
    p.m.geom_contype[p.table_gid] = 1; p.m.geom_conaffinity[p.table_gid] = 1

    held = bool(drop < 0.01 and after["n_contacts"] > 0 and disp_after < 0.05)
    rec = dict(inset=float(inset), sq=float(sq), dz=float(dz),
               eps=g["epsilon"], n=g["n_contacts"], f_total=g["f_total"],
               delta=g["delta"], displaced=disp, drop_m=drop,
               n_after=after["n_contacts"], disp_after=disp_after, held=held)

    if held:                     # only lift what is actually held
        _settle_grip(p, q_app, q_in)
        up = np.array([0.0, 0.0, 1.0])
        zl = p.obj_pos()[2]; zmax = zl
        qq = q_in.copy()
        for i in range(1, 9):
            d = lift_h * i / 8
            qq[0:7] = p.ik_side_to("R", gR + d * up)
            qq[18:25] = p.ik_side_to("L", gL + d * up)
            p.drive(qq, 26)
            zmax = max(zmax, p.obj_pos()[2])
        p.hold_ctrl(qq, 60)
        fin = grasp_metrics(p.m, p.d, p.obj_gid, p.obj_bid, ex)
        rec["net_lift_m"] = float(p.obj_pos()[2] - zl)
        rec["max_lift_m"] = float(zmax - zl)
        rec["n_after_lift"] = fin["n_contacts"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="legacy/results/squeeze_sweep.json")
    a = ap.parse_args()
    t0 = time.time()
    p = build_probe(seed=0, **BOX)
    rows = []
    print(f"{'inset(mm)':>10}{'sq':>6}{'dz':>7}{'ncon':>6}{'F(N)':>10}{'eps':>8}"
          f"{'disp(cm)':>10}{'drop(cm)':>10}{'held':>6}{'lift(cm)':>10}")
    for sq in (0.7, 0.85, 1.0):
        for dz in (0.0, -0.02):
            for inset in (0.000, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018):
                r = run(p, inset, sq, dz)
                rows.append(r)
                print(f"{inset*1000:>10.1f}{sq:>6.2f}{dz:>7.3f}{r['n']:>6}"
                      f"{r['f_total']:>10.2f}{r['eps']:>8.4f}"
                      f"{r['displaced']*100:>10.1f}{r['drop_m']*100:>10.1f}"
                      f"{str(r['held']):>6}"
                      f"{r.get('net_lift_m', float('nan'))*100:>10.1f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    held = [r for r in rows if r["held"]]
    print(f"\nheld {len(held)}/{len(rows)}")
    if held:
        b = max(held, key=lambda r: r.get("net_lift_m", -9))
        print("BEST:", json.dumps(b, indent=2))
    print(f"wrote {a.out} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
