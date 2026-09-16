"""Retarget a GRAB sequence onto a robot hand, frame by frame.

Single-pose retargeting (`oppdef.grasping.retarget_pose`) answers "what joint angles make
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
from oppdef.hands.tips import tip_offset
from oppdef.grasping.retarget_pose import correspond, retargeter_for

#: MANO tips come out thumb-first; this repository orders tips fingers-then-
#: thumb, and `correspond` relies on that to pair thumb with thumb. Getting
#: this backwards silently pairs the human thumb with the robot index finger.
MANO_TO_REPO = [1, 2, 3, 4, 0]          # index, middle, ring, pinky, thumb
REPO_TIP_NAMES = ("index", "middle", "ring", "pinky", "thumb")

#: MANO's joint chains, in the repo's finger order (index, middle, ring, pinky,
#: thumb): the proximal, middle and distal joint of each finger. Verified
#: empirically against the tip vertices rather than assumed -- chain (7,8,9) is
#: the PINKY and (10,11,12) the ring, which is not the order the numbering
#: suggests.
MANO_CHAINS = ((1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9), (13, 14, 15))

#: Weight on the intermediate (non-tip) joints. **Default 0, and that is a
#: result rather than a default.** Setting it to 0.45 makes the retarget match
#: the human far better -- contacts on the mug's handle 31.8% -> 38.8%, median
#: distance to the human's own contact points 19.6 -> 13.1 mm, and the render
#: visibly changes from gripping the body to reaching through the handle. It
#: also makes the grasp WORSE at its job:
#:
#:      W_JOINT   G5 hold rate (2790 obs, 279 clusters)
#:        0.00      0.318   CI [0.270, 0.367]
#:        0.45      0.200   CI [0.161, 0.238]
#:
#: Non-overlapping intervals. And the gap survives grasp synthesis, so it is not
#: something the repair stage absorbs: on a 20-frame subsample, 0.500 -> 0.750
#: from tips-only against 0.250 -> 0.650 from the wrap-aware fit.
#:
#: That is this project's founding claim, arriving from the other direction:
#: reproducing the human's hand pose more faithfully is not the same as
#: producing a grasp that works, and optimising for the first costs the second.
#: Set W_JOINT > 0 when contact fidelity is what you want to study.
#:
#: Lower than a contact, because
#: these are shape targets rather than contact targets, but not zero: a handle
#: grasp touches the object with the MIDDLE phalanges while the fingertip sits
#: in free space inside the hole. Measured on `mug_drink_1`, the human's
#: fingertips are 5-19 mm off the surface through the grasp and only the thumb
#: tip touches (1.3-2.4 mm), so a fingertip-only objective has no way to
#: reproduce the wrap and the retarget slid off the handle onto the body.
W_JOINT = 0.0

CONTACT_TOL = 0.015     # a human tip within 15 mm of the surface was reaching for it
W_CONTACT = 1.0
W_FREE = 0.25
W_WRIST = 0.15
W_SMOOTH = 0.05
#: Penetration weight for ARM-SIDE bodies -- forearm, wrist, palm, and the
#: knuckles. These have no business touching the object at all, and they are
#: what makes the retarget infeasible: scanned over 20 sequences and 2866
#: frames, only 23 frames are under 1 mm of penetration, and the sequences that
#: NEVER reach a clean frame are precisely those whose deepest body is a forearm
#: (binoculars, 17.15 mm on all 155 frames), a palm (flashlight_on_2, 12.82 mm
#: on all 144) or a proximal link. Sequences whose worst offender is a distal
#: link do reach clean frames. A Shadow forearm is not shaped like a human's, so
#: placing the wrist where the human's wrist was puts the arm inside the object.
#:
#: Weighted far above the finger term because it is a feasibility constraint
#: rather than a preference: no grasp is acceptable with the forearm inside the
#: object, whereas a fingertip 2 mm in is the grasp.
W_PEN_ARM = 60.0

W_PEN = 1.0             # with fingertips targeted at their DERIVED points and
                        # allowed 2 mm, a light penalty is enough; heavier ones
                        # fight the contact targets without reducing penetration
                        # the object is not a grasp, whatever its tips score
PEN_MARGIN = 0.0        # non-fingertip links: any penetration is penalised
TIP_ALLOW = 0.002       # fingertips may sink 2 mm; beyond that it is not contact


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
                 w_pen=W_PEN, tip_offsets=None, base_qpos=None):
        self.sc = sc
        self.m, self.d = sc.model, sc.data
        self.tips = list(tips)
        self.wrist = wrist_bid
        self.lo, self.hi = lo, hi
        self.dofs = np.array([sc.model.jnt_dofadr[j] for j in sc.jids], dtype=int)
        self.n = len(sc.jids)
        self.w_smooth, self.w_pen = w_smooth, w_pen
        #: When two hands share a scene, zeroing all of qpos to pose one of
        #: them also teleports the other to its zero configuration -- so the
        #: obstacle the solver is meant to avoid is not where it will be.
        self.base_qpos = None if base_qpos is None else np.asarray(base_qpos).copy()
        self.obj = set(sc.obj_gids)
        self.hand = set(sc.hand_gids)
        self.tips_g = set(sc.tip_gids)
        # Arm-side geoms: everything on the hand that is NOT strictly below a
        # knuckle, i.e. forearm, wrist, palm and the knuckles themselves.
        m = sc.model
        finger_roots = set()
        for b in sc.tip_bids:
            x = int(b)
            while x > 0 and x != sc.wrist_bid:
                finger_roots.add(x)
                x = int(m.body_parentid[x])
        self.arm_g = {g for g in sc.hand_gids
                      if int(m.geom_bodyid[g]) not in finger_roots}
        self._jp = np.zeros((3, sc.model.nv))
        self._jr = np.zeros((3, sc.model.nv))
        # A fingertip is the far end of the distal link, not that link's body
        # origin. Targeting the origin puts the CENTRE of the tip geom on the
        # object surface, which buries the geom by its own radius: measured,
        # 12 mm of tip penetration and 4469 N of contact force on a 0.2 kg mug.
        # oppdef.hands.tips derives the real point from the collision geometry;
        # this project has already been caught by exactly this once.
        self.tip_off = (tip_offsets if tip_offsets is not None else
                        [tip_offset(sc.model, b) for b in self.tips])
        # the two links proximal to each fingertip, for the shape targets
        self.chain = []
        for b in self.tips:
            up, x = [], int(sc.model.body_parentid[b])
            while x > 0 and x != sc.wrist_bid and len(up) < 2:
                up.append(x)
                x = int(sc.model.body_parentid[x])
            self.chain.append(up)              # [middle, proximal]

    def _tip_world(self, i):
        b = self.tips[i]
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.tip_off[i]

    def _fk(self, q, collide=True):
        self.d.qpos[:] = 0.0 if self.base_qpos is None else self.base_qpos
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

    def _jac_tip(self, i):
        return self._jac_point(self._tip_world(i), self.tips[i])

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
            # A fingertip ON the surface is the grasp, so tips get an
            # allowance rather than exemption. Exempting them entirely let the
            # fit bury them 12 mm in, and the constraint solver answered with
            # 4469 N of contact force on a 0.2 kg object -- which makes every
            # number downstream of it meaningless.
            arm = hand_g in self.arm_g
            allow = TIP_ALLOW if hand_g in self.tips_g else PEN_MARGIN
            depth = -float(c.dist) - allow
            if depth <= 0:
                continue
            w_this = W_PEN_ARM if arm else self.w_pen
            # contact frame row 0 is the normal, pointing geom1 -> geom2
            n = np.array(c.frame[:3]) * sign
            bid = int(self.m.geom_bodyid[hand_g])
            J = self._jac_point(np.array(c.pos), bid)
            rows.append(w_this * (-(n @ J))[None, :])
            res.append(w_this * np.array([depth]))
        return rows, res

    def solve(self, q0, targets, weights, wrist_target, q_prev,
              iters=12, damp=1e-3, joint_targets=None, weights_scale=None):
        """targets/weights are per ROBOT tip (already corresponded).

        `joint_targets[i]` is [middle, proximal] world positions for robot tip
        i, or None entries where the human has no counterpart.
        """
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        if weights_scale is None:
            weights_scale = np.ones(len(self.tips))
        for _ in range(iters):
            self._fk(q, collide=self.w_pen > 0)
            rows, res = [], []
            for i, bid in enumerate(self.tips):
                w = weights[i]
                if w <= 0:
                    continue
                rows.append(w * self._jac_tip(i))
                res.append(w * (self._tip_world(i) - targets[i]))
            for i, up in enumerate(self.chain):
                for lvl, bid in enumerate(up):
                    tg = joint_targets[i][lvl] if joint_targets is not None else None
                    if tg is None:
                        continue
                    w = W_JOINT * weights_scale[i]
                    rows.append(w * self._jac_body(bid))
                    res.append(w * (self.d.xpos[bid] - tg))
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
        return q, np.array([self._tip_world(i) for i in range(len(self.tips))])


def retarget_sequence(seq, side: str = "rhand", hand: str = "shadow",
                      window: tuple[int, int] | None = None,
                      contact_tol: float = CONTACT_TOL,
                      sc=None, tree=None, iters: int = 80,
                      w_pen: float = W_PEN,
                      w_smooth: float = W_SMOOTH,
                      base_qpos=None, q_init=None) -> RobotTrack:
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
                     w_smooth=w_smooth, w_pen=w_pen, base_qpos=base_qpos)

    Q = np.zeros((len(frames), len(sc.jids)))
    tip_err = np.zeros(len(frames))
    con_err = np.zeros(len(frames))
    n_con = np.zeros(len(frames), int)

    q = np.clip((sc.q_closure if q_init is None else np.asarray(q_init)[0]).copy(),
                lo, hi)
    q_prev = None
    for t, k in enumerate(frames):
        R = seq.obj_R[k]
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

        # shape targets: the human's own middle and proximal joints, in the
        # object frame, for each corresponded finger. These are what carry a
        # WRAP -- a handle grasp contacts with the middle phalanges while the
        # fingertip sits in the hole.
        jt = [None] * len(sc.tip_bids)
        for a, b in zip(ri, hidx):
            ch = MANO_CHAINS[a]
            mid = (human.joints[k, ch[2], :] - p) @ R
            prox = (human.joints[k, ch[1], :] - p) @ R
            jt[b] = [mid, prox]

        wrist_t = (human.joints[k, 0, :] - p) @ R
        q, got = solver.solve(q, targets, weights, wrist_t, q_prev, iters=iters,
                              joint_targets=jt)
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


def retarget_bimanual(seq, window, hand_r="shadow", hand_l="shadow_left",
                      rounds: int = 2, sc=None, **kw):
    """Fit BOTH hands, each seeing the other.

    Fitted independently in separate single-hand scenes, neither solver knows
    the other hand exists: measured on `gamecontroller_play_1`, the two poses
    interpenetrated by 11.7 mm across 42 contacts and the left hand pushed the
    right off the object entirely. Separating them afterwards with a rigid shift
    removes the overlap and makes tracking worse (105 mm -> 255 m), because
    clearing the other hand also breaks the grasp -- the poses are mutually
    inconsistent, not merely overlapping.

    So they are fitted in one shared scene, alternately: each pass re-solves one
    hand with the other posed at its current solution and folded into the
    obstacle set, for `rounds` sweeps. The scene must use the HINGE base -- a
    free joint gives the solver no wrist DoF to move, which is why this could
    not be done before.
    """
    from oppdef.human.scene import build_bimanual

    sc = sc or build_bimanual(hand_r, hand_l, seq.obj, obj_static=True,
                              base="hinges")
    if sc.base_kind != "hinges":
        raise ValueError("bimanual fitting needs base='hinges'")

    sides = {"r": ("rhand", hand_r), "l": ("lhand", hand_l)}
    views = {sd: sc.side_view(sd, hand=hk) for sd, (_s, hk) in sides.items()}
    tracks, poses = {}, {}

    for rnd in range(rounds):
        for sd, (side, _hk) in sides.items():
            other = "l" if sd == "r" else "r"
            base = np.zeros(sc.model.nq)
            if other in poses:
                # hold the other hand at its own first-frame solution, so the
                # obstacle is where it will actually be
                base[sc.qadr[other]] = poses[other]
            tracks[sd] = retarget_sequence(
                seq, side, _hk, window=window, sc=views[sd],
                base_qpos=base,
                q_init=None if sd not in tracks else tracks[sd].q, **kw)
            poses[sd] = tracks[sd].q[0]
    return tracks, sc, views
