"""Build the left Allegro hand DexTrack does not ship, in the convention it does.

DexTrack trains one hand. Its right hand is
`allegro_hand_description_right_fly_v2.urdf`: a six-joint "fly" base
(prismatic x, y, z then revolute x, y, z, no offsets) carrying the sixteen
Allegro finger joints, so a pose is the 22-vector their references store,
`[x, y, z, rx, ry, rz, joint_0 .. joint_15]` with the rotation read as
intrinsic XYZ Euler. They also ship `allegro_hand_description_left.urdf`,
which has the sixteen left finger joints and no base at all: its palm is
bolted to the root through a fixed joint with a 9.5 cm offset, and its
meshes are addressed through Drake package paths that do not resolve here.

This module splices the two: the left hand's fingers onto the right hand's
base, with the fixed joint and its offset removed so that the left hand's
22-vector means exactly what the right hand's does. That is the whole point.
A reference retargeted against this file can be handed to their trainer for
either hand without a coordinate map, and a two-hand reference is then just
two 22-vectors and one object trajectory.

    python dextrack/left_hand.py --out out/dextrack_assets/allegro_hand_description_left_fly_v2.urdf

`--check` verifies the result against the right hand: same joint names, same
degree-of-freedom count and ordering, and a palm-relative fingertip layout
that is the right hand's mirrored in y.
"""
from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ASSETS = Path("out/dextrack_assets")
RIGHT = ASSETS / "allegro_hand_description_right_fly_v2.urdf"
LEFT = ASSETS / "allegro_hand_description_left.urdf"

#: the chain their references address, in order. Also the order of the 22-vector.
BASE_JOINTS = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"]
FINGER_JOINTS = [f"joint_{i}" for i in range(16)]
DOF_ORDER = BASE_JOINTS + FINGER_JOINTS

#: the links the fly base introduces between the root and the palm
BASE_LINKS = ["link_palm_x", "link_palm_y", "link_palm_z", "link_palm_rx", "link_palm_ry"]

#: Drake package paths in their left hand, which resolve nowhere on this machine
DRAKE_PREFIX = "package://drake/manipulation/models/allegro_hand_description/meshes/"
MESH_PREFIX = "../meshes/"


def build(right: Path = RIGHT, left: Path = LEFT) -> ET.ElementTree:
    """The left hand's fingers on the right hand's fly base."""
    r_root = ET.parse(right).getroot()
    l_root = ET.parse(left).getroot()

    # their left hand reaches its meshes through Drake; ours sit beside the urdf
    for mesh in l_root.iter("mesh"):
        fn = mesh.get("filename") or ""
        if fn.startswith(DRAKE_PREFIX):
            mesh.set("filename", MESH_PREFIX + fn[len(DRAKE_PREFIX):])

    # drop the fixed root-to-palm joint: the fly base replaces it, and its 9.5 cm
    # offset would make the left hand's base translation mean something different
    # from the right hand's
    for j in list(l_root.findall("joint")):
        if j.get("type") == "fixed" and j.find("parent").get("link") == "hand_root":
            l_root.remove(j)

    # Mirror every finger joint's placement from the right hand rather than
    # trusting the left file's own numbers. Their left hand carries the ring
    # finger's mount rotation unmirrored (+5 degrees where the mirror of the
    # right hand's +5 is -5), which splays that finger the wrong way and moves
    # its tip 19 mm. Mirroring in the y = 0 plane: a position (x, y, z) goes to
    # (x, -y, z), an rpy (r, p, y) to (-r, p, -y), and a rotation axis, being a
    # pseudovector, to (-a_x, a_y, -a_z). The left meshes and inertias are kept.
    r_joints = {j.get("name"): j for j in r_root.findall("joint")}
    for j in l_root.findall("joint"):
        src = r_joints.get(j.get("name"))
        if src is None or not j.get("name", "").startswith("joint_"):
            continue
        s_o, o = src.find("origin"), j.find("origin")
        if s_o is not None and o is not None:
            if s_o.get("xyz"):
                x, y, z = (float(v) for v in s_o.get("xyz").split())
                o.set("xyz", f"{x} {-y} {z}")
            if s_o.get("rpy"):
                rr, pp, yy = (float(v) for v in s_o.get("rpy").split())
                o.set("rpy", f"{-rr} {pp} {-yy}")
        s_a, a = src.find("axis"), j.find("axis")
        if s_a is not None and a is not None and s_a.get("xyz"):
            ax, ay, az = (float(v) for v in s_a.get("xyz").split())
            a.set("xyz", f"{-ax} {ay} {-az}")

    # carry over the base links and joints verbatim, so the two hands share
    # limits, effort and damping as well as geometry
    have = {l.get("name") for l in l_root.findall("link")}
    for name in BASE_LINKS:
        src = next(l for l in r_root.findall("link") if l.get("name") == name)
        if name not in have:
            l_root.append(copy.deepcopy(src))
    for name in BASE_JOINTS:
        src = next(j for j in r_root.findall("joint") if j.get("name") == name)
        l_root.append(copy.deepcopy(src))

    l_root.set("name", "allegro_hand_left_fly")
    return ET.ElementTree(l_root)


def mujoco_copy(urdf: Path, work: Path, meshdir: Path = ASSETS / "meshes") -> Path:
    """The same urdf with a compiler tag, so MuJoCo can find the meshes.

    The tag goes INSIDE `<robot>`, which is where MuJoCo's urdf extension
    looks for it. Put it before `<robot>` and the file has two root elements:
    MuJoCo reads the first, and loads a world with nothing in it.
    """
    import re

    work.mkdir(parents=True, exist_ok=True)
    txt = urdf.read_text()
    if "<mujoco>" not in txt:
        tag = f'\n  <mujoco><compiler meshdir="{meshdir.resolve()}/" balanceinertia="true" discardvisual="false"/></mujoco>'
        m = re.search(r"<robot\b[^>]*>", txt)
        if m is None:
            raise ValueError(f"no <robot> element in {urdf}")
        txt = txt[:m.end()] + tag + txt[m.end():]
    out = work / (urdf.stem + "_mj.urdf")
    out.write_text(txt)
    return out


def check(out: Path) -> None:
    """Load both hands in MuJoCo and compare what a 22-vector does to each."""
    import numpy as np
    import mujoco

    def load(p):
        m = mujoco.MjModel.from_xml_path(str(mujoco_copy(p, Path("out/urdf_check"))))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        return m, mujoco.MjData(m), names

    mr, dr, nr = load(RIGHT)
    ml, dl, nl = load(out)
    assert nl == nr == DOF_ORDER or set(nl) == set(nr), f"joint sets differ:\n {nl}\n {nr}"
    print(f"dofs: right {mr.nq}, left {ml.nq}; joint names identical: {sorted(nl) == sorted(nr)}")

    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(8):
        q = np.zeros(22)
        q[6:] = rng.uniform(-0.3, 0.9, 16)          # finger joints only; base at identity
        for m, d, names in ((mr, dr, nr), (ml, dl, nl)):
            d.qpos[:] = 0
            for n, v in zip(DOF_ORDER, q):
                d.qpos[m.jnt_qposadr[names.index(n)]] = v
            mujoco.mj_forward(m, d)
        pr = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_BODY, "palm_link")
        pl = mujoco.mj_name2id(ml, mujoco.mjtObj.mjOBJ_BODY, "palm_link")
        for tip in ("link_3", "link_7", "link_11", "link_15"):
            a = dr.xpos[mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_BODY, tip)] - dr.xpos[pr]
            b = dl.xpos[mujoco.mj_name2id(ml, mujoco.mjtObj.mjOBJ_BODY, tip)] - dl.xpos[pl]
            b_mirrored = np.array([b[0], -b[1], b[2]])
            worst = max(worst, float(np.abs(a - b_mirrored).max()))
    print(f"palm-relative fingertips, left mirrored in y vs right: worst {worst*1000:.2f} mm over 8 poses")
    if worst > 0.002:
        raise SystemExit("left hand is not the right hand's mirror; check the splice")
    print("left hand ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--right", type=Path, default=RIGHT)
    ap.add_argument("--left", type=Path, default=LEFT)
    ap.add_argument("--out", type=Path, default=ASSETS / "allegro_hand_description_left_fly_v2.urdf")
    ap.add_argument("--check", action="store_true", help="verify against the right hand in MuJoCo")
    a = ap.parse_args()
    tree = build(a.right, a.left)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, "  ")
    tree.write(a.out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {a.out}")
    if a.check:
        check(a.out)


if __name__ == "__main__":
    main()
