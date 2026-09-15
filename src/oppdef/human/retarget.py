"""Retarget a GRAB sequence onto a robot hand, frame by frame.

Single-pose retargeting (`oppdef.retarget`) answers "what joint angles make
this hand look like that grasp".  A tracking reference needs something else: a
CONTINUOUS trajectory whose contacts land on the object in the same places the
human's did, because that -- not the hand's silhouette -- is what determines
whether the object can be carried.

So the target here is contact-centric.  Each human fingertip close to the
object is projected onto the object surface, and the robot fingertip is asked
to reach that surface point.  The object is the same size for the human and the
robot, so this removes the hand-scale problem entirely instead of fitting a
scale factor: a smaller robot hand simply moves its wrist in.  Fingertips that
were NOT near the object keep the human's own position at a low weight, which
holds the posture plausible without pretending a non-contact is a contact.

The solve is Gauss-Newton on MuJoCo's analytic body Jacobians, warm-started
from the previous frame.  Finite differences would need ~30 forward-kinematics
calls per iteration per frame; the analytic Jacobian is one, which is what
makes 291 sequences tractable on this machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco
from scipy.spatial import cKDTree

from oppdef.human import grab as grab_mod
from oppdef.retarget import correspond, retargeter_for

#: MANO tips come out thumb-first; this repository orders tips fingers-then-
#: thumb, and `correspond` relies on that to pair thumb with thumb. Getting
#: this backwards silently pairs the human thumb with the robot index finger.
MANO_TO_REPO = [1, 2, 3, 4, 0]          # index, middle, ring, pinky, thumb
REPO_TIP_NAMES = ("index", "middle", "ring", "pinky", "thumb")

CONTACT_TOL = 0.015     # a human tip within 15 mm of the surface was reaching for it
W_CONTACT = 1.0
W_FREE = 0.25
W_WRIST = 0.15
W_SMOOTH = 0.05
W_PEN = 2.0             # penetration outweighs tip error: a pose 14 mm inside
                        # the object is not a grasp, whatever its tips score
PEN_MARGIN = 0.0        # only true penetration is penalised


@dataclass
class RobotTrack:
    """A robot hand trajectory retargeted from one human hand."""
    hand: str
    side: str
    seq: str
    q: np.ndarray                 # (T, nq) joint values, base DoF included
    joint_names: list[str]
    tip_err: np.ndarray           # (T,) mean fingertip residual, metres
    contact_err: np.ndarray       # (T,) residual on CONTACTING tips only
    n_contact: np.ndarray         # (T,) how many tips were asked to touch
    frames: np.ndarray            # (T,) index into the source sequence
    meta: dict = field(default_factory=dict)


def _dof_of(m, jids) -> np.ndarray:
    return np.array([m.jnt_dofadr[j] for j in jids], dtype=int)


class _Solver:
    """Gauss-Newton fit of one frame, solved against the collision scene.

    The model carries the object, so penetration is read from MuJoCo's own
    narrowphase rather than estimated: every hand-object contact with negative
    distance contributes a residual that pushes that link out along the contact
    normal.  Constraining fingertips alone is not enough -- it leaves the
    middle phalanges 14 mm inside a mug while the tips score 7 mm.
    """

    def __init__(self, sc, tips, wrist_bid, lo, hi, w_smooth=W_SMOOTH,
                 w_pen=W_PEN):
        self.sc = sc
        self.m, self.d = sc.model, sc.data
        self.tips = list(tips)
        self.wrist = wrist_bid
        self.lo, self.hi = lo, hi
        self.dofs = np.array([sc.model.jnt_dofadr[j] for j in sc.jids], dtype=int)
        self.n = len(sc.jids)
        self.w_smooth, self.w_pen = w_smooth, w_pen
        self.obj = set(sc.obj_gids)
        self.hand = set(sc.hand_gids) - set(sc.tip_gids)
        self._jp = np.zeros((3, sc.model.nv))
        self._jr = np.zeros((3, sc.model.nv))

    def _fk(self, q, collide=True):
        self.d.qpos[:] = 0.0
        self.d.qpos[self.sc.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        # mj_jacBody reads d.cdof, which mj_kinematics does NOT fill. Without
        # this the Jacobian is identically zero, every Gauss-Newton step is
        # zero, and the fit silently returns its seed pose.
        mujoco.mj_comPos(self.m, self.d)
        if collide:
            mujoco.mj_collision(self.m, self.d)

    def _jac_body(self, bid):
        self._jp[:] = 0.0
        mujoco.mj_jacBody(self.m, self.d, self._jp, self._jr, bid)
        return self._jp[:, self.dofs].copy()

    def _jac_point(self, point, bid):
        self._jp[:] = 0.0
        mujoco.mj_jac(self.m, self.d, self._jp, self._jr,
                      np.asarray(point, float), bid)
        return self._jp[:, self.dofs].copy()

    def _pen_rows(self):
        """One scalar residual per penetrating hand-object contact."""
        rows, res = [], []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.obj and g2 in self.hand:
                hand_g, sign = g2, +1.0
            elif g2 in self.obj and g1 in self.hand:
                hand_g, sign = g1, -1.0
            else:
                continue
            depth = PEN_MARGIN - float(c.dist)
            if depth <= 0:
                continue
            # contact frame row 0 is the normal, pointing geom1 -> geom2
            n = np.array(c.frame[:3]) * sign
            bid = int(self.m.geom_bodyid[hand_g])
            J = self._jac_point(np.array(c.pos), bid)
            rows.append(self.w_pen * (-(n @ J))[None, :])
            res.append(self.w_pen * np.array([depth]))
        return rows, res

    def solve(self, q0, targets, weights, wrist_target, q_prev,
              iters=12, damp=1e-3):
        """targets/weights are per ROBOT tip (already corresponded)."""
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        for _ in range(iters):
            self._fk(q, collide=self.w_pen > 0)
            rows, res = [], []
            for i, bid in enumerate(self.tips):
                w = weights[i]
                if w <= 0:
                    continue
                rows.append(w * self._jac_body(bid))
                res.append(w * (self.d.xpos[bid] - targets[i]))
            if wrist_target is not None:
                rows.append(W_WRIST * self._jac_body(self.wrist))
                res.append(W_WRIST * (self.d.xpos[self.wrist] - wrist_target))
            if self.w_pen > 0:
                pr, pe = self._pen_rows()
                rows += pr
                res += pe
            if q_prev is not None and self.w_smooth > 0:
                rows.append(self.w_smooth * np.eye(self.n))
                res.append(self.w_smooth * (q - q_prev))

            J = np.vstack(rows)
            r = np.concatenate(res)
            # Levenberg-Marquardt step; the damping also keeps the redundant
            # base DoF from wandering when the fingers alone could satisfy the
            # residual.
            H = J.T @ J + damp * np.eye(self.n)
            try:
                dq = np.linalg.solve(H, -J.T @ r)
            except np.linalg.LinAlgError:
                break
            step = np.linalg.norm(dq)
            if step > 0.25:                       # trust region, radians/metres
                dq *= 0.25 / step
            q = np.clip(q + dq, self.lo, self.hi)
            if step < 1e-5:
                break
        self._fk(q)
        return q, self.d.xpos[np.array(self.tips)].copy()


def retarget_sequence(seq, side: str = "rhand", hand: str = "shadow",
                      window: tuple[int, int] | None = None,
                      contact_tol: float = CONTACT_TOL,
                      sc=None, tree=None, iters: int = 80,
                      w_pen: float = W_PEN,
                      w_smooth: float = W_SMOOTH) -> RobotTrack:
    """Fit `hand` to the human hand `side` over a GRAB sequence.

    `window` is (start, length) in sequence frames -- normally the hold window
    from the inventory, because retargeting the reach adds nothing a tracking
    controller can use and the free-flying hand there has no contacts to
    constrain it.

    `sc` is a built `human.scene.ObjectScene`; pass one in to reuse it across
    the sequences that share an object, since compiling the model and loading
    the convex decomposition dominate the cost of a short clip.

    `iters` is 80 rather than a dozen because the hand is redundant: five
    fingertip targets are fifteen constraints on twenty-nine degrees of
    freedom, leaving a fourteen-dimensional null space. Stopped early, each
    frame halts wherever its trust-region path happened to reach, and
    consecutive frames land in different parts of that null space -- the fitted
    palm then jumped 157 mm and 0.715 rad between frames while the human's own
    wrist moved at most 52.3 mm. Converged, the same fit moves the palm 52.6 mm,
    which is the human's motion, at no cost in accuracy (16.7 mm vs 16.6 mm).
    Intermediate values are NOT monotone -- iters=60 was worse than either --
    because the trust region makes the path, not just the optimum, matter.
    """
    from oppdef.human.scene import build as build_scene

    sc = sc or build_scene(hand=hand, obj=seq.obj)
    human = seq.hands[side]
    if human.verts is None:
        raise ValueError("load the sequence with verts=True")
    ov, _ = seq.obj_mesh
    tree = tree or cKDTree(ov)

    lo_f, hi_f = (0, seq.T) if window is None else (window[0], window[0] + window[1])
    frames = np.arange(lo_f, min(hi_f, seq.T))

    m = sc.model
    jr = m.jnt_range[sc.jids]
    lo, hi = jr[:, 0].copy(), jr[:, 1].copy()
    bad = hi <= lo
    lo[bad], hi[bad] = -np.pi, np.pi
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in sc.jids]
    for i, n in enumerate(names):
        if n in ("rx", "ry", "rz"):     # the base hinges ship at +/-0.06 rad;
            lo[i], hi[i] = -np.pi, np.pi   # a retarget must be able to turn over

    ri, hidx = correspond(5, len(sc.tip_bids))
    solver = _Solver(sc, sc.tip_bids, sc.wrist_bid, lo, hi,
                     w_smooth=w_smooth, w_pen=w_pen)

    Q = np.zeros((len(frames), len(sc.jids)))
    tip_err = np.zeros(len(frames))
    con_err = np.zeros(len(frames))
    n_con = np.zeros(len(frames), int)

    q = np.clip(sc.q_closure.copy(), lo, hi)
    q_prev = None
    for t, k in enumerate(frames):
        R = grab_mod._rodrigues(seq.obj_quat_aa[k][None])[0]
        p = seq.obj_pos[k]

        # Everything is solved in the OBJECT frame, never in GRAB's world
        # frame. GRAB places the object 0.8-1.7 m up, while the floating base
        # has 0.6 m of travel, so a world-frame fit cannot reach the object at
        # all and converges to the hand pinned against its own limits. The
        # object frame is also the frame the result is used in: a rigid hold is
        # a near-constant pose here, whatever the object is doing in the world.
        tips_o = (human.joints[k, 16:, :][MANO_TO_REPO] - p) @ R
        dist, idx = tree.query(tips_o, k=1)

        touch = dist < contact_tol
        tgt_all = np.where(touch[:, None], ov[idx], tips_o)
        w_all = np.where(touch, W_CONTACT, W_FREE)

        targets = np.zeros((len(sc.tip_bids), 3))
        weights = np.zeros(len(sc.tip_bids))
        targets[hidx] = tgt_all[ri]
        weights[hidx] = w_all[ri]

        wrist_t = (human.joints[k, 0, :] - p) @ R
        q, got = solver.solve(q, targets, weights, wrist_t, q_prev, iters=iters)
        q_prev = q.copy()

        Q[t] = q
        err = np.linalg.norm(got[hidx] - targets[hidx], axis=1)
        tip_err[t] = float(err.mean())
        sel = weights[hidx] >= W_CONTACT
        con_err[t] = float(err[sel].mean()) if sel.any() else np.nan
        n_con[t] = int(sel.sum())

    return RobotTrack(
        hand=hand, side=side, seq=seq.name, q=Q, joint_names=names,
        tip_err=tip_err, contact_err=con_err, n_contact=n_con, frames=frames,
        meta={"object": seq.obj, "intent": seq.intent, "subject": seq.subject,
              "dt": seq.dt, "contact_tol": contact_tol,
              "w_pen": w_pen, "w_smooth": w_smooth},
    )
