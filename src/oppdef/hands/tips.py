"""Where a fingertip actually is, derived from the finger's own geometry.

Every hand in this project tracked a BODY ORIGIN as its fingertip, and for all
four the most distal joint rotates about that very point: moving it displaced
the tracked position by 0.000 mm on shadow, leap and allegro, and on f5d6
before its tip frames were restored. The opposition floor -- the measurement
this project is named after -- was therefore taken at a point that is not the
part of the finger that touches anything, and that is blind to the last joint
of every finger.

The fix has to be general, because the defect is. A fingertip is not a frame
someone remembered to name; it is the far end of the distal link, and the model
already says where that is: the collision geometry. So the tip is derived as
the point of the distal body's own collision geoms lying furthest from that
body's origin, in the body frame. No hand-authored offsets, nothing to keep in
sync with an asset update, and it works for a capsule, a box or a mesh alike.

Derived, not tabled -- the same rule this project applies to closure poses, and
for the same reason: a table is a place for an error to hide.
"""
from __future__ import annotations

import numpy as np
import mujoco


def _quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _geom_extremes(model, gid):
    """Candidate extreme points of one geom, in its parent body's frame."""
    t = int(model.geom_type[gid])
    pos = np.array(model.geom_pos[gid], float)
    R = _quat_mat(model.geom_quat[gid])
    s = np.array(model.geom_size[gid], float)
    G = mujoco.mjtGeom
    out = []
    if t == G.mjGEOM_SPHERE:
        for ax in np.eye(3):
            out += [pos + R @ (ax * s[0]), pos - R @ (ax * s[0])]
    elif t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        half = s[1] + (s[0] if t == G.mjGEOM_CAPSULE else 0.0)
        z = R @ np.array([0.0, 0, 1.0])
        out += [pos + z * half, pos - z * half]
    elif t == G.mjGEOM_BOX:
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    out.append(pos + R @ (np.array([sx, sy, sz]) * s))
    elif t == G.mjGEOM_MESH:
        mid = int(model.geom_dataid[gid])
        if mid >= 0:
            a = int(model.mesh_vertadr[mid])
            n = int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + n].reshape(-1, 3)
            out += list(pos + v @ R.T)
    else:                                  # ellipsoid, plane, sdf: bound it
        r = float(model.geom_rbound[gid])
        for ax in np.eye(3):
            out += [pos + ax * r, pos - ax * r]
    return out


def tip_offset(model, body_id, collision_only=True):
    """Offset of the fingertip from the body origin, in the body frame.

    The furthest point of the body's geometry from its own origin. Returns a
    zero vector when the body carries no usable geom, which keeps the old
    behaviour rather than inventing a location.
    """
    best, best_d = np.zeros(3), -1.0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != int(body_id):
            continue
        if collision_only and not (model.geom_contype[g]
                                   or model.geom_conaffinity[g]):
            continue
        for p in _geom_extremes(model, g):
            d = float(np.linalg.norm(p))
            if d > best_d:
                best, best_d = p, d
    return best


def tip_points(model, data, body_ids, offsets=None):
    """World positions of the fingertips for the current kinematic state."""
    offs = offsets if offsets is not None else [
        tip_offset(model, b) for b in body_ids]
    out = []
    for b, off in zip(body_ids, offs):
        R = data.xmat[b].reshape(3, 3)
        out.append(data.xpos[b] + R @ off)
    return np.array(out)
