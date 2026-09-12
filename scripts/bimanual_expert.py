"""Scripted expert for the bimanual lid task, and its one-handed control.

The point of this file is the CONTROL, not the expert. The task is designed so
that extracting the peg needs more upward force (the socket's 1.20 N of joint
friction) than the base weighs (0.78 N), so a one-handed pull should lift the
whole base off the table instead of extracting anything. Running only the
grasping hand is how that gets checked, rather than asserted.

Phases:
  0  settle
  1  stabilising hand presses the exposed flange of the base
  2  grasping hand descends alongside the peg with fingers open
  3  grasping hand closes, then squeezes
  4  grasping hand pulls straight up
  5  hold

Success (fixed in bimanual_env's docstring before any run):
  peg extracted >= 8 cm, base displacement < 2 cm, base tilt < 15 deg.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_env import (BimanualBox, RH_HOME, LH_HOME, BASE_HALF,  # noqa: E402
                          PEG_HALF, SOCKET_FRICTION)

PALM_DZ = -0.102                     # palm sits this far below its attach frame
OPEN = [0.15, 0.20, 0.10, 0.30, 0.15, 0.15, 0.10]
GRASP = [0.95, 1.25, 0.65, 1.70, 0.95, 1.05, 0.65]
FLAT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # stabilising hand: flat palm

# The grasp offset is CALIBRATED per run, not hard-coded. A fixed offset taken
# from one closure missed the knob by 2.65 cm and the fingers closed onto the lid
# instead (contacts: lid_geom vs rh_if_tip / rh_mf_tip, box_geom vs rh_th_tip),
# because the closure's thumb-to-finger gap is ~10 cm against a 6 cm knob -- the
# same feasibility problem grasp_bench.py already solves for a single hand.
FINGER_JOINTS = [f"{f}_{j}" for f in ("if", "mf", "rf") for j in ("mcp", "pip", "dip")] \
    + ["th_cmc", "th_axl", "th_mcp", "th_ipl"]


def _amp_vector(close):
    """Expand the 7-number closure spec to one value per finger joint."""
    v = []
    for _ in ("if", "mf", "rf"):
        v += list(close[:3])
    v += list(close[3:])
    return np.array(v, float)


def calibrate(e, prefix, rz, close, frac):
    """Kinematic gap and grasp-centre offset for this hand at closure `frac`.

    Returns (gap, grasp_centre - palm) in world coordinates, with the base
    rotation rz applied. Position of the palm does not depend on rz (the
    rotational joints sit at the palm origin), so the offset is all that is
    needed to aim the hand.
    """
    m, d = e.m, e.d
    q = d.qpos.copy()
    n2 = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s)
    for dof, val in zip(("x", "y", "z", "rx", "ry", "rz"), (0, 0, 0, 0, 0, rz)):
        q[m.jnt_qposadr[n2(f"{prefix}{dof}")]] = val
    amp = _amp_vector(close)
    for name, a in zip(FINGER_JOINTS, amp):
        q[m.jnt_qposadr[n2(f"{prefix}{name}")]] = frac * a
    saved = d.qpos.copy()
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{b}")
            for b in ("if_ds", "mf_ds", "rf_ds", "th_ds")]
    tp = d.xpos[tips].copy()
    palm = d.xpos[e.palms[prefix]].copy()
    fm, th = tp[:3].mean(0), tp[3]
    gap = float(np.linalg.norm(th - fm))
    centre = 0.5 * (fm + th)
    opp = (th - fm) / (gap + 1e-12)
    d.qpos[:] = saved
    mujoco.mj_kinematics(m, d)
    return gap, centre - palm, opp


def fit_closure(e, prefix, rz, close, width, n=160):
    """Closure fraction whose gap matches `width`, plus its offset and the
    achievable gap range. Returns None if `width` is outside that range."""
    fr = np.linspace(0.0, 1.0, n)
    gaps = np.array([calibrate(e, prefix, rz, close, f)[0] for f in fr])
    lo, hi = float(gaps.min()), float(gaps.max())
    if not (lo <= width <= hi):
        return None, dict(gap_min=lo, gap_max=hi, feasible=False)
    i = int(np.argmin(np.abs(gaps - width)))
    g, off, opp = calibrate(e, prefix, rz, close, float(fr[i]))
    return dict(frac=float(fr[i]), gap=g, offset=off, opp_axis=opp), \
        dict(gap_min=lo, gap_max=hi, feasible=True)


def palm_ctrl(home, target_palm):
    """Base-joint ctrl (x, y, z) that puts the palm at `target_palm`."""
    rest = np.array(home, float) + np.array([0.0, 0.0, PALM_DZ])
    return (np.asarray(target_palm, float) - rest).tolist()


def best_rz(e, prefix, close, n=48):
    """Base yaw whose opposition axis best aligns with the peg's graspable axis.

    The peg is 5.0 x 6.0 x 14 cm and LEAP's closure gap spans 5.32-19.26 cm, so
    the 5.0 cm faces are INFEASIBLE. The 6.0 cm y-face is the only graspable one,
    so the hand has to be yawed to oppose along y.
    """
    best = None
    for rz in np.linspace(-np.pi, np.pi, n):
        _, _, opp = calibrate(e, prefix, float(rz), close, 1.0)
        score = abs(float(opp[1]))
        if best is None or score > best[1]:
            best = (float(rz), score, opp)
    return best


class Expert:
    def __init__(self, two_handed=True, peg_width=2 * PEG_HALF[1],
                 grip_height=0.045, **kw):
        self.e = BimanualBox(**kw)
        self.two_handed = two_handed
        self.grip_height = grip_height
        e = self.e
        mujoco.mj_forward(e.m, e.d)
        for _ in range(400):
            e.d.ctrl[:] = e.ctrl_vec()
            mujoco.mj_step(e.m, e.d)
        self.peg0 = e.knob_pos().copy()
        self.out0 = e.peg_out()

        self.rz, self.align, self.opp = best_rz(e, "rh_", GRASP)
        self.sol, self.info = fit_closure(e, "rh_", self.rz, GRASP, peg_width)
        if self.sol is None:
            raise RuntimeError(
                f"peg width {peg_width*100:.1f} cm outside the hand's gap range "
                f"{self.info['gap_min']*100:.2f}-{self.info['gap_max']*100:.2f} cm")
        self.off = np.asarray(self.sol["offset"], float)
        self.frac = self.sol["frac"]
        self.pre_frac = max(self.frac - 0.22, 0.0)
        sq, _ = fit_closure(e, "rh_", self.rz, GRASP,
                            max(peg_width - 0.008, self.info["gap_min"] + 0.002))
        self.sq_frac = sq["frac"] if sq else min(self.frac + 0.06, 1.0)
        self.log = []

    def fing(self, frac):
        return [frac * v for v in GRASP]

    def grip_point(self, h):
        """Where the grasp centre should be when the peg is h metres out.

        Aimed at the UPPER part of the peg so the fingers reach alongside it
        rather than bottoming out on the base.
        """
        return self.peg0 + np.array([0.0, 0.0, self.grip_height + h])

    def rh_for_h(self, h, back=0.0):
        """Palm ctrl for grip height h, optionally `back` metres retracted along
        the opposition axis (used for a lateral approach)."""
        lat = np.array([self.opp[0], self.opp[1], 0.0])
        n = np.linalg.norm(lat)
        lat = lat / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        return palm_ctrl(RH_HOME, self.grip_point(h) - self.off + back * lat)

    def start_at_pregrasp(self, lh_xyz):
        """Place the grasping hand AROUND the peg, fingers open, before anything
        moves -- then open further until the start is penetration-free.

        The approach motion was the problem, not the grasp. Descending from above
        drove the fingers onto the peg's top face (163 N, gripping nothing);
        sweeping in from the side shoved the whole base 8 cm. Starting from a
        pre-grasp is what grasp_bench does and how Dexonomy's data is structured:
        each datapoint is a pre-grasp / grasp / squeeze triple, not a reach.
        """
        e = self.e
        m, d = e.m, e.d
        n2 = lambda s_: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s_)
        rh = self.rh_for_h(0.0)
        frac = self.pre_frac
        for _ in range(30):
            for dof, val in zip(("x", "y", "z", "rx", "ry", "rz"),
                                list(rh) + [0.0, 0.0, self.rz]):
                d.qpos[m.jnt_qposadr[n2(f"rh_{dof}")]] = val
            for dof, val in zip(("x", "y", "z", "rx", "ry", "rz"),
                                list(lh_xyz) + [0.0, 0.0, 0.0]):
                d.qpos[m.jnt_qposadr[n2(f"lh_{dof}")]] = val
            for name, a in zip(FINGER_JOINTS, _amp_vector(GRASP)):
                d.qpos[m.jnt_qposadr[n2(f"rh_{name}")]] = frac * a
                d.qpos[m.jnt_qposadr[n2(f"lh_{name}")]] = 0.0
            mujoco.mj_forward(m, d)
            touching = any(e.knob_gid in (d.contact[i].geom1, d.contact[i].geom2)
                           or e.box_gid in (d.contact[i].geom1, d.contact[i].geom2)
                           and ("rh_" in (mujoco.mj_id2name(
                               m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1) or "")
                               or "rh_" in (mujoco.mj_id2name(
                                   m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2) or ""))
                           for i in range(d.ncon))
            if not touching:
                return frac, True
            frac *= 0.90
        return frac, False

    def run(self, pull=0.14, verbose=True):
        e = self.e
        # LATERAL approach. Descending straight down drove the fingers onto the
        # peg's top face: the hand ended 2.2 cm above its commanded height with
        # 163 N of contact, gripping nothing, and simply unloaded as it rose.
        # Instead: come down beside the peg, then translate in along the
        # opposition axis, then close.
        rh_grasp = self.rh_for_h(0.0)
        rh_side = self.rh_for_h(0.0, back=0.12)
        rh_up = list(rh_side); rh_up[2] += 0.10
        lh_press = palm_ctrl(LH_HOME, [0.0, -0.085, 0.044])
        lh_up = list(lh_press); lh_up[2] += 0.10

        def cmd(rh_xyz, frac, lh_xyz, n=1):
            c = e.ctrl_vec(rh_pose=list(rh_xyz) + [0.0, 0.0, self.rz],
                           lh_pose=list(lh_xyz) + [0.0, 0.0, 0.0],
                           rh_close=self.fing(frac), lh_close=FLAT)
            for _ in range(n):
                e.d.ctrl[:] = c
                mujoco.mj_step(e.m, e.d)
                self.log.append(dict(t=float(e.d.time), out=e.peg_out(),
                                     box=e.box_pos().tolist(),
                                     tilt=e.box_tilt_deg()))

        lh_now = lh_press if self.two_handed else lh_up
        self.pre_frac, clear = self.start_at_pregrasp(lh_now)
        self.clear_start = clear
        cmd(rh_grasp, self.pre_frac, lh_now, n=300)
        if verbose:
            print(f"   after press: peg out {e.peg_out()*100:5.2f} cm  "
                  f"base {np.round(e.box_pos(), 4)}")

        for i in range(1, 61):
            cmd(rh_grasp, self.pre_frac + (i / 60) * (self.frac - self.pre_frac),
                lh_now, n=10)
        for i in range(1, 31):
            cmd(rh_grasp, self.frac + (i / 30) * (self.sq_frac - self.frac),
                lh_now, n=8)
        cmd(rh_grasp, self.sq_frac, lh_now, n=250)
        grasp_ncon = int(sum(1 for i in range(e.d.ncon)
                             if e.knob_gid in (e.d.contact[i].geom1,
                                               e.d.contact[i].geom2)))
        if verbose:
            print(f"   after grasp: peg contacts {grasp_ncon}  "
                  f"out {e.peg_out()*100:5.2f} cm")
        for i in range(1, 141):
            cmd(self.rh_for_h(pull * i / 140), self.sq_frac, lh_now, n=12)
        cmd(self.rh_for_h(pull), self.sq_frac, lh_now, n=400)

        box0 = np.array([0.0, 0.0, BASE_HALF[2]])
        disp = float(np.linalg.norm(e.box_pos() - box0))
        out = e.peg_out() - self.out0
        return dict(two_handed=self.two_handed, rz=self.rz, frac=self.frac,
                    sq_frac=self.sq_frac, gap_cm=self.sol["gap"] * 100,
                    align=self.align, peg_out_m=float(out),
                    box_disp_m=disp,
                    box_z_rise_m=float(e.box_pos()[2] - box0[2]),
                    box_tilt_deg=e.box_tilt_deg(), knob_contacts=grasp_ncon,
                    clear_start=bool(getattr(self, "clear_start", False)),
                    success=bool(out >= 0.08 and disp < 0.02
                                 and e.box_tilt_deg() < 15.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bimanual_expert.json")
    ap.add_argument("--pull", type=float, default=0.14)
    a = ap.parse_args()
    t0 = time.time(); rows = []
    for two in (True, False):
        print(f"\n=== {'TWO-HANDED (expert)' if two else 'ONE-HANDED (control)'} ===")
        ex = Expert(two_handed=two)
        r = ex.run(pull=a.pull)
        rows.append(r)
        print(f"   peg out {r['peg_out_m']*100:6.2f} cm | base moved "
              f"{r['box_disp_m']*100:5.2f} cm (z {r['box_z_rise_m']*100:+5.2f}) | "
              f"tilt {r['box_tilt_deg']:5.1f} deg | peg contacts "
              f"{r['knob_contacts']} | success={r['success']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2, default=float))
    two, one = rows
    print("\n" + "=" * 70)
    print(f"{'':22}{'peg out(cm)':>13}{'base moved(cm)':>16}{'base z(cm)':>12}{'ok':>7}")
    for r in rows:
        n = "two-handed expert" if r["two_handed"] else "one-handed control"
        print(f"{n:22}{r['peg_out_m']*100:>13.2f}{r['box_disp_m']*100:>16.2f}"
              f"{r['box_z_rise_m']*100:>12.2f}{str(r['success']):>7}")
    print("=" * 70)
    if two["success"] and not one["success"]:
        print("The task requires two hands: the expert extracts the peg, the "
              "one-handed control does not.")
    elif one["success"]:
        print("WARNING: the one-handed control ALSO succeeds. The task does not "
              "require two hands and the design is wrong.")
    else:
        print("The expert does not solve the task yet; nothing may be learned "
              "against it until it does.")
    print(f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
