"""A MuJoCo scene holding a GRAB object and a retargeted robot hand.

Built in the OBJECT frame, which is where the retargeting solves: the object
sits at the origin with identity orientation and the hand moves around it.
That makes a rigid hold a nearly constant pose, and it means the same scene
serves every frame of every sequence that uses this object.

This exists to check the retarget in physics rather than in the residual that
produced it.  A fingertip residual of 6 mm is compatible with the middle
phalanges being 20 mm inside the object -- the fit only ever looked at tips --
and that failure has already happened once in this repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

from oppdef import paths
from oppdef.human import grab as grab_mod

#: Decimating the collision mesh is not cosmetic: MuJoCo's convex-hull
#: collision is O(vertices) per pair, and GRAB's contact meshes carry up to
#: 30k. mjMAXCONPAIR also caps contacts per geom pair, so the hull is what
#: matters, not the triangle count.
MAX_HULL_VERTS = 400


@dataclass
class ObjectScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    spec: mujoco.MjSpec
    tip_bids: list[int]
    wrist_bid: int
    q_closure: np.ndarray
    obj_gids: list[int]
    hand_gids: list[int]
    tip_gids: list[int]
    qadr: np.ndarray
    jids: list[int]
    base_kind: str = "hinges"
    mocap_bid: int = -1
    base_bid: int = -1
    obj_bid: int = -1

    def set_q(self, q):
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)

    def penetration(self):
        """(max_depth_m, n_contacts) between the hand and the object."""
        d = self.data
        obj, hand = set(self.obj_gids), set(self.hand_gids)
        depth, n = 0.0, 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in obj and g2 in hand) or (g2 in obj and g1 in hand):
                n += 1
                depth = max(depth, -float(c.dist))
        return depth, n


def _decimate(v: np.ndarray, f: np.ndarray, n: int = MAX_HULL_VERTS):
    """Keep at most n vertices, chosen by a uniform stride over the hull."""
    if len(v) <= n:
        return v, f
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(v)
        idx = np.unique(h.vertices)
        if len(idx) > n:
            idx = idx[np.linspace(0, len(idx) - 1, n).astype(int)]
        return v[idx], None
    except Exception:                                    # noqa: BLE001
        step = int(np.ceil(len(v) / n))
        return v[::step], None


def _mocap_parent(hand: str):
    """A hand on a FREE joint, driven by a welded mocap body.

    The default floating base is three slides and three hinges in series. Three
    serial hinges do not merely parameterise SO(3) awkwardly -- they lose a
    degree of freedom at gimbal lock, where two axes align, and that is a
    property of the mechanism rather than of the maths. Measured on
    `mug_drink_1`, the base command jumped 0.70 rad between frames while the
    object turned 0.231, and constraining the solve to stay nearby cost up to
    293 mm of placement, because no nearby solution exists.

    A free joint has no such configuration. The hand is then driven by welding
    it to a mocap body whose pose is commanded directly as SE(3), which is also
    the representation a tracking controller should be producing.
    """
    from oppdef.embodiment import HANDS, hand_spec

    h = HANDS[hand]
    parent = mujoco.MjSpec()
    parent.worldbody.add_light(
        pos=[0.2, -0.2, 1.2], dir=[-0.2, 0.2, -1.0],
        type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
        diffuse=[0.8, 0.8, 0.8], ambient=[0.45, 0.45, 0.45])

    base = parent.worldbody.add_body(name="hand_base")
    base.add_freejoint()
    # a massless-ish stub so the free joint is well conditioned on its own
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0],
                  rgba=[0, 0, 0, 0], contype=0, conaffinity=0, mass=0.01)
    parent.attach(hand_spec(h), prefix="hand_", frame=base.add_frame())

    mocap = parent.worldbody.add_body(name="hand_mocap", mocap=True)
    mocap.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01],
                   rgba=[0.9, 0.2, 0.2, 0.0], contype=0, conaffinity=0)
    eq = parent.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2 = "hand_mocap", "hand_base"
    eq.data[:7] = [0, 0, 0, 1, 0, 0, 0]
    # finite stiffness: an infinitely rigid wrist can push the object with
    # unbounded force, which is not a wrist
    eq.solref[:2] = [0.01, 1.0]
    return parent, "hand_"


def build(hand: str = "shadow", obj: str = "mug", free_base: bool = True,
          obj_static: bool = True, convex_parts: bool = True,
          base: str = "hinges") -> ObjectScene:
    """Hand + GRAB object, in the object frame.

    `base` is "hinges" (three slides and three hinges, position-actuated) or
    "mocap" (a free joint welded to a mocap body, commanded as SE(3)). Use
    hinges for static fitting and mocap for anything that follows a trajectory.
    """
    from oppdef.embodiment import make

    if base == "mocap":
        spec, pfx_m = _mocap_parent(hand)
        emb = None
    else:
        emb = make(hand=hand, free_base=free_base)
        spec = emb.spec
        pfx_m = list(emb.palm_bid)[0]
    pfx = pfx_m
    # the default offscreen buffer is 640x480; renders here are square
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1024)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 1024)

    # One body, one geom per convex part. A single mesh geom would be
    # collided as its convex hull, which for these objects encloses up to 3.5x
    # the true volume -- see oppdef.human.decompose.
    from oppdef.human.decompose import decompose

    parts = decompose(obj) if convex_parts else None
    if parts is None:
        v, f = grab_mod._ply(
            paths.GRAB / "tools" / "object_meshes" / "contact_meshes" / f"{obj}.ply")
        parts = [_decimate(v, f)[0]]

    body = spec.worldbody.add_body()
    body.name = f"obj_{obj}"
    if not obj_static:
        body.add_freejoint()
    for i, pv in enumerate(parts):
        if len(pv) < 4:
            continue
        mesh = spec.add_mesh()
        mesh.name = f"grab_{obj}_{i}"
        mesh.uservert = np.asarray(pv, float).flatten().tolist()
        g = body.add_geom()
        g.name = f"objgeom_{obj}_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = f"grab_{obj}_{i}"
        g.rgba = [0.75, 0.75, 0.78, 1.0]
        g.condim = 4

    model = spec.compile()
    data = mujoco.MjData(model)

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{obj}")
    obj_gids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == obj_bid]

    # the hand's geoms, by body descent from the palm -- f5d6's file is a whole
    # robot, so anything name-based picks up the torso and the other arm
    from oppdef.embodiment import HANDS
    h = HANDS[hand]
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 f"{pfx}{h.palm}")
    mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_base")

    def descends(b):
        while b > 0:
            if b == palm_bid:
                return True
            b = int(model.body_parentid[b])
        return False

    hand_gids = [i for i in range(model.ngeom)
                 if descends(int(model.geom_bodyid[i]))]

    base_names = {f"{pfx}{d}" for d in ("x", "y", "z", "rx", "ry", "rz")}
    jids = [j for j in range(model.njnt)
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_FREE,
                                         mujoco.mjtJoint.mjJNT_BALL)
            and (descends(int(model.jnt_bodyid[j]))
                 or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
                 in base_names)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])

    tip_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{t}")
                for t in h.tip_names]
    wrist_bid = palm_bid

    # The closing posture is DERIVED (specs.derive_flex solves it over all
    # joints at once); here it is only re-indexed into this model's order so it
    # can seed the fit.
    from oppdef.hands.specs import derive_flex
    _m, _cfg, flex, _t, _g = derive_flex(hand)
    q_closure = np.zeros(len(jids))
    for i, j in enumerate(jids):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        key = n[len(pfx):] if pfx and n.startswith(pfx) else n
        if key in flex:
            q_closure[i] = flex[key]

    # Fingertip geoms are excluded from the penetration penalty: a fingertip
    # resting on the surface is the grasp, not an error, and penalising it
    # fights the contact targets directly -- doing so cost 27 mm of contact
    # accuracy to buy 4 mm of penetration.
    tipset = set(tip_bids)

    def under_tip(b):
        while b > 0:
            if b in tipset:
                return True
            b = int(model.body_parentid[b])
        return False

    tip_gids = [i for i in hand_gids if under_tip(int(model.geom_bodyid[i]))]

    return ObjectScene(model=model, data=data, spec=spec, tip_bids=tip_bids,
                       wrist_bid=wrist_bid, q_closure=q_closure,
                       obj_gids=obj_gids, hand_gids=hand_gids,
                       tip_gids=tip_gids, qadr=qadr, jids=jids,
                       base_kind=base, mocap_bid=mocap_bid, base_bid=base_bid,
                       obj_bid=obj_bid)
