"""Bimanual environment: two LEAP hands, one hinged box, a task that needs both.

Design rules this env exists to satisfy:

  - HANDS THAT CAN GRASP. Two Menagerie LEAP hands, opposition floor 0.00 cm,
    attached with `MjSpec.attach` so their names are prefixed rh_/lh_ and nothing
    collides. Not f5d6, whose 3.08 cm floor makes grasping impossible, and not on
    the Vega arms, whose reachable-orientation set is tight enough (90 deg
    reorientations unreachable, no vertical headroom) to confound any result with
    a limitation of the robot rather than of the hands.

  - A TASK THAT CANNOT BE DONE ONE-HANDED, by construction rather than by
    assertion. A PEG sits in a socket in a free-standing base, held by joint
    friction. Extracting it needs an upward force GREATER than the weight of the
    base, so pulling one-handed lifts the whole base off the table instead of
    extracting anything. Hold the base down with the other hand and the peg comes
    out. The one-handed run is a control that gets run, not a thought experiment.

    This replaced a hinged lid. The lid was equivalent in principle but its arc
    made the scripted reach fragile -- the hand had to track a rotating target
    and kept pressing the lid closed (85 N on the knob's top face, lid going
    5.9 deg -> 2.2 deg) instead of lifting it. A peg is a pure vertical pull, and
    a vertical post is the most reliable thing this hand grasps.

  - ROLE ASYMMETRY. The two hands do different things: one stabilises with a
    palm press, the other grasps and lifts. Two hands squeezing opposite faces
    of a block is a gripper with extra steps, which is what the Vega result was.

Success, v2, fixed before the run that uses it:
    peg extracted >= 8 cm, base LIFT < 2 cm, base tilt < 15 deg.

v1 used base lateral DISPLACEMENT < 2 cm. That clause was discarded because it
is not a usable discriminator: CPU and warp differ by 1.2 cm of lateral
displacement on an identical open-loop control sequence, so a 2 cm threshold
sits inside the backend noise. Base lift is what the task is actually built
around -- the socket friction exceeds the base weight, so a one-handed pull
lifts the base -- and it separates the two conditions by ~6x in BOTH backends
(0.59 vs 6.08 cm on CPU, 0.63 vs 3.95 cm on warp). Lateral displacement is still
reported, it just no longer gates.
"""
from __future__ import annotations

import numpy as np
import mujoco

from oppdef.paths import MENAGERIE
from pathlib import Path


# --- task geometry (metres, kg) ---
BASE_HALF = (0.060, 0.110, 0.025)     # wide in y so the stabilising palm has room:
                                      # at 2.5 cm strips the two hands collided
# The peg: 5.0 x 6.0 x 14 cm, standing proud of the base. The 6.0 cm y-width is
# the graspable axis -- LEAP's closure gap spans 5.32-19.26 cm, so anything under
# 5.3 cm is infeasible for this hand. Tall so the fingers can reach alongside it
# without bottoming out on the base.
PEG_HALF = (0.025, 0.030, 0.070)
KNOB_HALF = PEG_HALF                  # kept for callers that import the old name
BASE_MASS = 0.08
PEG_MASS = 0.05
SOCKET_FRICTION = 1.20                # N, must exceed the base weight
HINGE_FRICTION = SOCKET_FRICTION      # old name, same knob
BASE_FRICTION = "0.4 0.02 0.001"

BASE_POS = (0.0, 0.0, BASE_HALF[2])
PEG_REST_Z = 2 * BASE_HALF[2] + PEG_HALF[2] - 0.03   # 3 cm seated in the socket

# Homes are derived from measured offsets, not guessed. With the attach frame
# flipped, each palm sits 10 cm below its frame; the right hand's CLOSED grasp
# centre is at palm + (-0.025, +0.016, -0.078), so the frame that puts that
# centre on the knob is knob + (0.025, -0.016, 0.078) + (0, 0, 0.10). Both homes
# are then raised so the hands start clear and descend under actuator control.
#
# The stabilising hand goes to -y, not +y: the right hand's thumb points toward
# +y from its palm, and with the left hand there the two collided at reset
# (rh_palm vs lh_if_ds, rh_th_px vs lh_mf_tip).
RH_HOME = (0.063, -0.016, 0.40)       # right hand: above the knob
LH_HOME = (0.000, -0.085, 0.30)       # left hand: above the exposed -y strip

FINGERS = ["if", "mf", "rf"]
FLEX = ["mcp", "pip", "dip"]
THUMB = ["th_cmc", "th_axl", "th_mcp", "th_ipl"]
BASE_DOF = ["x", "y", "z", "rx", "ry", "rz"]


def _add_base_dof(spec, palm, prefix, home):
    """Six position-controlled DoF so the hand floats: 3 slides + 3 hinges."""
    body = spec.body(palm)
    axes = dict(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1],
                rx=[1, 0, 0], ry=[0, 1, 0], rz=[0, 0, 1])
    for d in BASE_DOF:
        slide = d in ("x", "y", "z")
        body.add_joint(
            name=f"{prefix}{d}",
            type=mujoco.mjtJoint.mjJNT_SLIDE if slide else mujoco.mjtJoint.mjJNT_HINGE,
            axis=axes[d],
            range=[-0.6, 0.6] if slide else [-3.2, 3.2],
            damping=8.0 if slide else 1.0,
            armature=0.02)
    return body


def build(hinge_friction=HINGE_FRICTION, base_mass=BASE_MASS, two_handed=True,
          peg_friction=1.0):
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0

    wb = spec.worldbody
    wb.add_light(pos=[0.2, -0.2, 1.2], dir=[-0.2, 0.2, -1.0],
                 type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
                 diffuse=[0.8, 0.8, 0.8], ambient=[0.45, 0.45, 0.45])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[1.0, 1.0, 0.05], pos=[0, 0, 0],
                rgba=[0.55, 0.55, 0.58, 1], friction=[0.4, 0.02, 0.001])

    # ---- the two hands ----
    for prefix, xml, home in (("rh_", MENAGERIE / "leap_hand" / "right_hand.xml", RH_HOME),
                              ("lh_", MENAGERIE / "leap_hand" / "left_hand.xml", LH_HOME)):
        child = mujoco.MjSpec.from_file(str(xml))
        frame = wb.add_frame(pos=list(home), quat=[0, 1, 0, 0])   # palms down
        spec.attach(child, prefix=prefix, frame=frame)

    # ---- the base and its peg ----
    box = wb.add_body(name="box", pos=list(BASE_POS))
    box.add_freejoint(name="box_free")
    box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=list(BASE_HALF), mass=base_mass,
                 rgba=[0.35, 0.42, 0.62, 1],
                 friction=[float(v) for v in BASE_FRICTION.split()])

    peg = box.add_body(name="peg", pos=[0.0, 0.0, PEG_REST_Z - BASE_POS[2]])
    peg.add_joint(name="peg_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                  axis=[0, 0, 1], range=[-0.01, 0.30],
                  frictionloss=hinge_friction, damping=0.05, armature=0.002)
    peg.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=list(PEG_HALF), mass=PEG_MASS,
                 rgba=[0.85, 0.75, 0.30, 1],
                 friction=[peg_friction, 0.02, 0.001])

    # ---- floating base DoF + actuators ----
    for prefix, home in (("rh_", RH_HOME), ("lh_", LH_HOME)):
        _add_base_dof(spec, f"{prefix}palm", prefix, home)
        for d in BASE_DOF:
            slide = d in ("x", "y", "z")
            kp = 4000.0 if slide else 200.0
            kv = 200.0 if slide else 20.0
            gp = [0.0] * 10; gp[0] = kp                 # gainprm is length 10
            bp = [0.0] * 10; bp[1], bp[2] = -kp, -kv    # biasprm likewise
            spec.add_actuator(
                name=f"{prefix}{d}_act", target=f"{prefix}{d}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gainprm=gp, biasprm=bp,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                ctrllimited=1,
                ctrlrange=[-0.6, 0.6] if slide else [-3.2, 3.2])

    model = spec.compile()
    return model, spec


class BimanualBox:
    """Handles for the scripted expert."""

    def __init__(self, hinge_friction=HINGE_FRICTION, base_mass=BASE_MASS,
                 peg_friction=1.0):
        self.m, self.spec = build(hinge_friction, base_mass,
                                  peg_friction=peg_friction)
        self.d = mujoco.MjData(self.m)
        n2 = lambda t, s: mujoco.mj_name2id(self.m, t, s)
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a
                    for a in range(self.m.nu)}
        self.box_bid = n2(mujoco.mjtObj.mjOBJ_BODY, "box")
        self.peg_bid = n2(mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.box_gid = n2(mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        self.knob_gid = n2(mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        self.box_q = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "box_free")]
        self.hinge_q = self.m.jnt_qposadr[n2(mujoco.mjtObj.mjOBJ_JOINT, "peg_slide")]
        self.palms = {p: n2(mujoco.mjtObj.mjOBJ_BODY, f"{p}palm")
                      for p in ("rh_", "lh_")}

    # ---- state ----
    def peg_out(self):
        """Peg extraction along the socket axis, metres."""
        return float(self.d.qpos[self.hinge_q])

    def lid_angle(self):          # old name; now the extraction distance
        return self.peg_out()

    def box_pos(self):
        return self.d.qpos[self.box_q:self.box_q + 3].copy()

    def box_quat(self):
        return self.d.qpos[self.box_q + 3:self.box_q + 7].copy()

    def box_tilt_deg(self):
        q = self.box_quat()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        up = R.reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))

    def knob_pos(self):
        return self.d.geom_xpos[self.knob_gid].copy()

    # ---- control ----
    def ctrl_vec(self, rh_pose=None, lh_pose=None, rh_close=None, lh_close=None):
        c = self.d.ctrl.copy()
        for prefix, pose, close in (("rh_", rh_pose, rh_close),
                                    ("lh_", lh_pose, lh_close)):
            if pose is not None:
                for d, v in zip(BASE_DOF, pose):
                    c[self.act[f"{prefix}{d}_act"]] = v
            if close is not None:
                for f in FINGERS:
                    for j, v in zip(FLEX, close[:3]):
                        c[self.act[f"{prefix}{f}_{j}_act"]] = v
                for n, v in zip(THUMB, close[3:]):
                    c[self.act[f"{prefix}{n}_act"]] = v
        return c

    def step(self, c, n):
        for _ in range(n):
            self.d.ctrl[:] = c
            mujoco.mj_step(self.m, self.d)
