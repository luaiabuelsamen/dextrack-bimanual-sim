"""Embodiment registry: every hand and arm on this machine, and how to compose them.

Until now each experiment reached for models its own way -- `hand_specs` knew
about hands, `envs.bimanual` hard-coded two LEAP paths, `envs.hand_bench` built
its own XML. Adding a hand meant editing three files and adding an arm was not
possible at all.

One registry, two tables, one builder:

    make(hand="leap")                     a floating hand
    make(hand="leap", arm="ur5e")         that hand on that arm
    make(hand="leap", count=2)            two of them, prefixed rh_/lh_

Everything mechanical about a hand -- how it closes, where its grasp centre is --
is still DERIVED, never tabled (see `oppdef.hands.specs`). What lives here is
naming and geometry: which file, which body is the palm, which bodies are the
fingertips, where an arm accepts a tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from oppdef.paths import MENAGERIE, VEGA_URDF, compile_urdf, menagerie_xml


@dataclass(frozen=True)
class Hand:
    key: str
    source: str                 # "menagerie" | "urdf"
    path: str                   # relative to MENAGERIE, or absolute for urdf
    palm: str
    tips: tuple                 # non-thumb fingertips
    thumb: str
    dof: int
    note: str
    joint_prefix: str = ""      # restrict to these joints (f5d6 lives inside a robot)

    @property
    def tip_names(self):
        return tuple(self.tips) + (self.thumb,)


@dataclass(frozen=True)
class Arm:
    key: str
    path: str                   # relative to MENAGERIE
    mount: str                  # site or body a tool attaches to
    dof: int
    note: str


HANDS = {
    "leap": Hand("leap", "menagerie", "leap_hand/right_hand.xml", "palm",
                 ("if_ds", "mf_ds", "rf_ds"), "th_ds", 16, "LEAP, 16 DoF"),
    "leap_left": Hand("leap_left", "menagerie", "leap_hand/left_hand.xml", "palm",
                      ("if_ds", "mf_ds", "rf_ds"), "th_ds", 16, "LEAP left, 16 DoF"),
    "allegro": Hand("allegro", "menagerie", "wonik_allegro/right_hand.xml", "palm",
                    ("ff_tip", "mf_tip", "rf_tip"), "th_tip", 16,
                    "Wonik Allegro, 16 DoF"),
    "shadow": Hand("shadow", "menagerie", "shadow_hand/right_hand.xml", "rh_palm",
                   ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"),
                   "rh_thdistal", 24, "Shadow Hand, 24 DoF"),
    # tips are the URDF's real tip frames, not the distal joint origins that
    # were tracked before (those do not move when the distal joint moves).
    # Six INDEPENDENT joints; the other five follow by mimic. See hands/f5d6.py.
    "f5d6": Hand("f5d6", "urdf", str(VEGA_URDF), "R_arm_l7",
                 ("R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"), "R_th_tip",
                 6, "Dexmate f5d6, 6 independent joints (5 mimic-coupled)",
                 joint_prefix="R_"),
}

ARMS = {
    "ur5e": Arm("ur5e", "universal_robots_ur5e/ur5e.xml", "attachment_site", 6,
                "Universal Robots UR5e"),
    "panda": Arm("panda", "franka_emika_panda/panda.xml", "attachment", 8,
                 "Franka Emika Panda (ships its own gripper)"),
    "xarm7": Arm("xarm7", "ufactory_xarm7/xarm7.xml", "link_tcp", 8,
                 "UFactory xArm7"),
    "so_arm100": Arm("so_arm100", "trs_so_arm100/so_arm100.xml", "", 6,
                     "TRS SO-ARM100 -- the SO-101 family, the lab's real hardware"),
}


def hand_xml(h: Hand) -> str:
    """XML text for a hand. URDF sources go through the glb-stripping compiler,
    which wants a Path -- `Hand.path` is a str so the dataclass stays hashable."""
    if h.source == "menagerie":
        return menagerie_xml(h.path)
    from pathlib import Path
    return compile_urdf(Path(h.path))


def hand_spec(h: Hand):
    return mujoco.MjSpec.from_string(hand_xml(h))


@dataclass
class Embodiment:
    """A compiled model plus the handles every downstream module asks for."""
    model: mujoco.MjModel
    spec: mujoco.MjSpec
    hands: dict = field(default_factory=dict)     # prefix -> Hand
    palm_bid: dict = field(default_factory=dict)  # prefix -> body id
    tip_bid: dict = field(default_factory=dict)   # prefix -> [body ids]
    arm: Arm | None = None

    def joint_ids(self, prefix=""):
        m = self.model
        out = []
        for j in range(m.njnt):
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                continue
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if n.startswith(prefix):
                out.append(j)
        return out

    def actuated(self):
        m = self.model
        return [a for a in range(m.nu)
                if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]


def _mount_dof(body, prefix):
    """Three slides and three hinges on a carrier body, so a hand can float.

    Ranges are written in DEGREES for the hinges because that is the unit these
    specs compile angles in -- writing radians there silently limited the
    bimanual env's wrist to 3.2 degrees. See NOTES 2026-09-12.
    """
    # A moving body needs mass, and MuJoCo derives it from the subtree. The
    # f5d6 hand comes from a URDF whose bodies carry no inertials, so the
    # carrier compiled at zero mass and was rejected. Stating a small inertia
    # explicitly makes the carrier self-sufficient for every hand.
    body.mass = 1e-3
    body.inertia = [1e-6, 1e-6, 1e-6]
    body.explicitinertial = True
    axes = dict(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1],
                rx=[1, 0, 0], ry=[0, 1, 0], rz=[0, 0, 1])
    for d, axis in axes.items():
        slide = d in ("x", "y", "z")
        body.add_joint(
            name=f"{prefix}{d}", axis=axis,
            type=mujoco.mjtJoint.mjJNT_SLIDE if slide else mujoco.mjtJoint.mjJNT_HINGE,
            range=[-0.6, 0.6] if slide else [-180.0, 180.0],
            damping=8.0 if slide else 1.0, armature=0.02)
    return body


def make(hand="leap", arm=None, count=1, prefixes=None, positions=None,
         quats=None, free_base=False, add_actuators=True):
    """Compose an embodiment.

    count=1            one hand, prefix ""
    count=2            two hands, prefixes rh_/lh_ (override with `prefixes`)
    arm="ur5e"         attach the hand at the arm's mount site
    free_base=True     give each palm 6 position-controlled DoF (a floating hand)
    """
    h = HANDS[hand] if isinstance(hand, str) else hand
    a = ARMS[arm] if isinstance(arm, str) else arm
    prefixes = prefixes or (("",) if count == 1 else ("rh_", "lh_")[:count])
    positions = positions or [[0, 0, 0]] * len(prefixes)
    quats = quats or [None] * len(prefixes)

    if a is not None:
        parent = mujoco.MjSpec.from_string(menagerie_xml(a.path))
    else:
        parent = mujoco.MjSpec()
        parent.worldbody.add_light(
            pos=[0.2, -0.2, 1.2], dir=[-0.2, 0.2, -1.0],
            type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
            diffuse=[0.8, 0.8, 0.8], ambient=[0.45, 0.45, 0.45])

    # Where the hand goes. With an arm, attach at its declared mount site so
    # `arm=` means MOUNTED rather than "both in the same scene"; without one,
    # attach to the world.
    mount_site = None
    if a is not None and a.mount:
        for st in parent.sites:
            if st.name == a.mount:
                mount_site = st
                break

    attached_prefix = []
    floating = free_base and a is None
    for pfx, pos, quat in zip(prefixes, positions, quats):
        child = hand_spec(h)
        kw = dict(pos=list(pos))
        if quat is not None:
            kw["quat"] = list(quat)
        # A prefix is required whenever anything else is in the scene: MuJoCo
        # namespaces bodies and joints on attach but NOT assets, so a hand and an
        # arm that both define a material called "black" collide. An empty
        # prefix is only safe for a lone hand.
        eff = pfx or ("hand_" if (a is not None or len(prefixes) > 1) else "")
        if mount_site is not None:
            parent.attach(child, prefix=eff, site=mount_site)
        elif floating:
            # The six base DoF get their OWN body, with the hand as its child.
            # Adding them to the palm works only for a palm that has no joints
            # of its own: MuJoCo allows at most 6 DoF per body, and Shadow's
            # palm already carries a 2-DoF wrist while f5d6's sits inside a
            # whole robot, so both failed to compile with "more than 6 dofs".
            mount = parent.worldbody.add_body(name=f"{eff}base", **kw)
            _mount_dof(mount, eff)
            parent.attach(child, prefix=eff, frame=mount.add_frame())
        else:
            parent.attach(child, prefix=eff, frame=parent.worldbody.add_frame(**kw))
        attached_prefix.append(eff)

    prefixes = tuple(attached_prefix)
    emb = Embodiment(model=None, spec=parent, arm=a)
    emb.hands = {p: h for p in prefixes}

    if free_base:
        from oppdef.envs.bimanual import _add_base_dof, BASE_DOF
        for pfx in prefixes:
            if not floating:
                _add_base_dof(parent, f"{pfx}{h.palm}", pfx, None)
            if add_actuators:
                for d in BASE_DOF:
                    slide = d in ("x", "y", "z")
                    kp = 4000.0 if slide else 200.0
                    kv = 200.0 if slide else 20.0
                    gp = [0.0] * 10; gp[0] = kp
                    bp = [0.0] * 10; bp[1], bp[2] = -kp, -kv
                    parent.add_actuator(
                        name=f"{pfx}{d}_act", target=f"{pfx}{d}",
                        trntype=mujoco.mjtTrn.mjTRN_JOINT, gainprm=gp, biasprm=bp,
                        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                        biastype=mujoco.mjtBias.mjBIAS_AFFINE, ctrllimited=1,
                        ctrlrange=[-0.6, 0.6] if slide else [-3.2, 3.2])

    emb.model = parent.compile()
    m = emb.model
    for pfx in prefixes:
        emb.palm_bid[pfx] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                              f"{pfx}{h.palm}")
        emb.tip_bid[pfx] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{t}")
                            for t in h.tip_names]
    return emb


def inventory():
    """What this machine can actually build, checked by loading each one."""
    rows = []
    for k, h in HANDS.items():
        try:
            m = mujoco.MjModel.from_xml_string(hand_xml(h))
            rows.append(("hand", k, m.nq, m.nu, h.note, "ok"))
        except Exception as e:
            rows.append(("hand", k, 0, 0, h.note, f"{type(e).__name__}"))
    for k, a in ARMS.items():
        try:
            m = mujoco.MjModel.from_xml_string(menagerie_xml(a.path))
            rows.append(("arm", k, m.nq, m.nu, a.note, "ok"))
        except Exception as e:
            rows.append(("arm", k, 0, 0, a.note, f"{type(e).__name__}"))
    return rows
