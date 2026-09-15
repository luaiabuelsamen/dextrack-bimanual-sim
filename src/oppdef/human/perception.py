"""Stage 7: close the loop on depth instead of on the simulator's ground truth.

Every controller in this project reads the object's pose straight out of MuJoCo.
A robot cannot. The question this stage exists to answer is narrow and testable:

    does a tracking controller that was tuned against ground-truth object pose
    still work when it is given an ESTIMATE from depth?

So the tracker is left exactly as it is -- frozen -- and only its object-pose
input is replaced. Anything that changes about the result is attributable to
perception, which is the only way to separate "the controller is brittle" from
"the estimator is bad".

The estimator here is deliberately not learned. A depth camera sees the object,
the object's mesh is known (it is the same mesh the scene was built from), and
registering one to the other is a solved problem: back-project the segmented
depth to a point cloud and run point-to-point ICP from the previous frame's
pose. A learned pose regressor would add a second thing that can be wrong, and
the question above is about the controller.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mujoco

os.environ.setdefault("MUJOCO_GL", "egl")


@dataclass
class Camera:
    """A depth camera looking at the scene, with its intrinsics."""
    width: int
    height: int
    fovy_deg: float
    pos: np.ndarray
    lookat: np.ndarray

    @property
    def K(self) -> np.ndarray:
        f = 0.5 * self.height / np.tan(np.radians(self.fovy_deg) / 2.0)
        return np.array([[f, 0.0, self.width / 2.0],
                         [0.0, f, self.height / 2.0],
                         [0.0, 0.0, 1.0]])

    def mjv(self) -> mujoco.MjvCamera:
        """MuJoCo places the eye at `lookat - distance * forward`.

        So the azimuth and elevation describe the direction the camera LOOKS,
        not where it sits. Deriving them from (pos - lookat) puts the camera on
        the opposite side of the scene: asking for (0.3, -0.3, 0.3) produced an
        eye at (-0.28, 0.32, -0.30).
        """
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        d = np.asarray(self.pos, float) - np.asarray(self.lookat, float)
        n = float(np.linalg.norm(d))
        u = -d / max(n, 1e-9)                       # the viewing direction
        c.lookat[:] = self.lookat
        c.distance = n
        c.azimuth = float(np.degrees(np.arctan2(u[1], u[0])))
        c.elevation = float(np.degrees(np.arcsin(np.clip(u[2], -1.0, 1.0))))
        return c


class DepthSensor:
    """Renders depth + segmentation and returns the object's point cloud.

    One renderer is kept for the life of the sensor: creating a second EGL
    context after closing the first raises EGL_NOT_INITIALIZED on this machine
    (see viz/floor_poses.py, which hit the same thing).
    """

    def __init__(self, model, cam: Camera, obj_gids):
        self.model = model
        self.cam = cam
        self.obj = set(int(g) for g in obj_gids)
        self._d = mujoco.Renderer(model, height=cam.height, width=cam.width)
        self._s = mujoco.Renderer(model, height=cam.height, width=cam.width)
        self._d.enable_depth_rendering()
        self._s.enable_segmentation_rendering()
        self._mjv = cam.mjv()

    def close(self):
        for r in (self._d, self._s):
            try:
                r.close()
            except Exception:                              # noqa: BLE001
                pass

    def cloud(self, data, lookat=None, noise_m: float = 0.0,
              rng=None) -> np.ndarray:
        """(N,3) world points on the object's visible surface."""
        if lookat is not None:
            self._mjv.lookat[:] = lookat
        self._d.update_scene(data, camera=self._mjv)
        self._s.update_scene(data, camera=self._mjv)
        depth = self._d.render()
        seg = self._s.render()[:, :, 0]          # geom id per pixel, -1 empty

        mask = np.isin(seg, list(self.obj)) & np.isfinite(depth) & (depth > 0)
        if mask.sum() == 0:
            return np.zeros((0, 3))
        ys, xs = np.nonzero(mask)
        z = depth[ys, xs].astype(np.float64)
        if noise_m > 0:
            rng = rng or np.random.default_rng(0)
            z = z + rng.normal(scale=noise_m, size=z.shape)

        K = self.cam.K
        x = (xs - K[0, 2]) * z / K[0, 0]
        y = (ys - K[1, 2]) * z / K[1, 1]
        # MuJoCo's camera looks down -Z with +Y up; image rows run downward
        pts_cam = np.stack([x, -y, -z], axis=1)

        # camera pose, from the scene MuJoCo actually rendered
        scn_cam = self._d.scene.camera[0]
        fwd = np.array(scn_cam.forward, float)
        up = np.array(scn_cam.up, float)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(right, fwd)
        R = np.stack([right, up, -fwd], axis=1)
        eye = np.array(scn_cam.pos, float)
        return pts_cam @ R.T + eye


def icp(model_pts: np.ndarray, model_tree, cloud: np.ndarray,
        R0: np.ndarray, t0: np.ndarray, iters: int = 25, trim: float = 0.9):
    """Register a known mesh to a PARTIAL depth cloud.

    The correspondence runs cloud -> model, not model -> cloud. A depth camera
    sees one side of the object, so most model points have no true match;
    matching the full model onto a one-sided cloud drags the mesh until it
    straddles the visible face. Measured, that direction took a 26.9 mm seed to
    43.7 mm. Going the other way, every observed point does have a match.

    Trimmed: the worst `1 - trim` of correspondences are dropped each iteration,
    which is what keeps a few points bled from the hand or the background from
    dominating the fit.
    """
    R, t = np.array(R0, float), np.array(t0, float)
    for _ in range(iters):
        # bring the observation into the model's frame
        local = (cloud - t) @ R
        d, idx = model_tree.query(local, k=1)
        keep = np.argsort(d)[:max(3, int(trim * len(d)))]
        Q = cloud[keep]                      # observed, world
        P = model_pts[idx[keep]]             # corresponding model points
        pc, qc = P.mean(0), Q.mean(0)
        H = (P - pc).T @ (Q - qc)
        U, _S, Vt = np.linalg.svd(H)
        Rn = Vt.T @ U.T
        if np.linalg.det(Rn) < 0:
            Vt[-1] *= -1
            Rn = Vt.T @ U.T
        tn = qc - Rn @ pc
        step = np.linalg.norm(tn - t) + np.linalg.norm(Rn - R)
        R, t = Rn, tn
        if step < 1e-7:
            break
    return R, t


def object_model_points(model, obj_gids, max_pts: int = 2000,
                        seed: int = 0) -> np.ndarray:
    """The object's surface points in its BODY frame, from the compiled model.

    Not from the source .ply. MuJoCo re-expresses a mesh asset in its own
    inertial frame on compile, so the geom's vertices differ from the file's by
    a rigid offset -- for the GRAB mug, enough that ICP converged tidily to a
    stable answer 26 mm from the truth. Reading the geometry that is actually
    simulated and rendered removes the mismatch instead of calibrating it away.

    Sampled over the mesh FACES, not just its vertices. A convex decomposition
    part carries at most 64 vertices, so the vertex set alone is ~12 mm apart on
    these objects, and a nearest-VERTEX match cannot resolve a pose better than
    that: ICP converged tidily and stably to an answer 28 mm out.
    """
    rng = np.random.default_rng(seed)
    tris = []
    for g in obj_gids:
        g = int(g)
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        v = model.mesh_vert[va:va + vn].astype(np.float64)
        f = model.mesh_face[fa:fa + fn].astype(int)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        v = v @ R.reshape(3, 3).T + model.geom_pos[g]
        tris.append(v[f])
    if not tris:
        return np.zeros((0, 3))
    T = np.concatenate(tris)                       # (F, 3, 3)
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    if area.sum() <= 0:
        return T.reshape(-1, 3)
    pick = rng.choice(len(T), size=max_pts, p=area / area.sum())
    u = rng.random((max_pts, 1))
    w = rng.random((max_pts, 1))
    flip = (u + w) > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return T[pick, 0] + u * e1[pick] + w * e2[pick]


class PoseEstimator:
    """Track an object's 6-DoF pose from depth, given its mesh.

    Seeded from the previous estimate, which is what a real tracker does and
    what makes the failure mode honest: once it loses the object it stays lost,
    rather than being silently re-seeded from the ground truth it is meant to
    replace.
    """

    def __init__(self, mesh_verts: np.ndarray, max_pts: int = 1500, seed: int = 0):
        from scipy.spatial import cKDTree
        v = np.asarray(mesh_verts, float)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(v), size=min(max_pts, len(v)), replace=False)
        self.model_pts = v[idx]
        self.model_tree = cKDTree(self.model_pts)
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.lost = False

    def reset(self, R, t):
        self.R, self.t, self.lost = np.array(R, float), np.array(t, float), False

    def update(self, cloud: np.ndarray, iters: int = 25):
        if len(cloud) < 20:
            self.lost = True
            return self.R, self.t
        self.R, self.t = icp(self.model_pts, self.model_tree, np.asarray(cloud),
                             self.R, self.t, iters=iters)
        return self.R, self.t

    def quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.R.flatten())
        return q
