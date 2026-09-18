"""Retarget a GRAB sequence onto a robot hand, frame by frame.

Single-pose retargeting (`handsim.grasping.retarget_pose`) answers "what joint angles make
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

from handsim.human import grab as grab_mod
from handsim.hands.tips import tip_offset
from handsim.grasping.retarget_pose import correspond, retargeter_for

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

#: Weight on a middle-phalanx CONTACT target. Distinct from `W_JOINT` above,
#: and the distinction is the whole point. `W_JOINT` asks the robot's middle
#: joint to be WHERE THE HUMAN'S BONE WAS, and that measurably costs hold rate
#: (0.318 -> 0.200) because a Shadow phalanx is not a human one: the human's
#: joint position is the wrong place for the robot's joint even when the
#: contact is right. This term instead asks the robot's middle phalanx to TOUCH
#: THE OBJECT where the human's middle phalanx touched it -- a contact target,
#: which is morphology-independent in the way a joint-position target is not.
#:
#: It fires only on fingers whose own human middle joint is inside
#: `CONTACT_TOL` of the surface, which is not a rare case. Measured, gaps in mm,
#: against the 15 mm tolerance:
#:
#:      reference          TIP gaps                   MID gaps            fires
#:      binoculars_see_1   3.1  7.8  6.5 10.8  0.9    5.8 10.3 18.4 24.5 16.1  2/5
#:      flashlight_on_2   10.3 15.4 12.8 23.3 27.9    7.8  2.9  5.4 24.0 13.1  4/5
#:      cup_lift           6.8 21.8 11.6 18.8  5.5    4.8  7.7  6.3  5.3  0.5  5/5
#:      mug_drink_1       12.8 13.1 13.6 16.4  1.1    7.6  8.2  9.8 26.0 27.7  3/5
#:
#: On `cup_lift` every middle phalanx is CLOSER to the object than the
#: fingertips are -- 0.5-7.7 mm against 5.5-21.8 mm. The human holds that cup
#: with the middles of its fingers, and a tips-only objective has no way to ask
#: for that. This is the term that makes the fit optimise the right links.
W_MID = 0.0

#: **Default 0, and like `W_JOINT` above that is a result rather than a
#: default.** At 0.6 the term does mechanically what it was designed to do --
#: binoculars engages 4 -> 5 distinct links, mug 5 -> 6, and the mug's contact
#: one-sidedness improves 0.669 -> 0.390 -- but it pays for every bit of that
#: in penetration, which is the one currency `wrap_score` refuses:
#:
#:      reference          w_mid  links  sided  worst pen mm
#:      binoculars_see_1    0.0     4    0.380     3.19
#:      binoculars_see_1    0.6     5    0.423     5.86
#:      flashlight_on_2     0.0     6    0.474     9.04
#:      flashlight_on_2     0.6     6    0.512    13.03
#:      mug_drink_1         0.0     5    0.669     5.02
#:      mug_drink_1         0.6     6    0.390     8.21
#:
#: Arm penetration stays at 0.00 throughout, so `W_PEN_ARM` is holding and the
#: extra burial is entirely finger-side. Split by link, the middle phalanx
#: itself stays inside the 3 mm allowance (0.0-2.2 mm); the depth lands on the
#: TIP (binoculars 3.2 -> 5.9, mug 5.0 -> 8.2) and the PROXIMAL link
#: (flashlight 1.0 -> 12.5). That is an over-constraint, not a weight to tune:
#: the fit still pins the fingertip to a surface vertex at `W_CONTACT` while
#: this term pins the middle phalanx too, and a Shadow finger's link lengths
#: cannot reach both points on the object's curvature the way the human's did.
#: `MID_FREES_TIP` below tests the obvious remedy and does not rescue it.

#: On a finger where the human contacts with BOTH tip and middle phalanx, drop
#: the fingertip from a surface target to a free one. See `w_mid` above: a
#: human finger reaches both contacts by wrapping around the object's
#: curvature, and a robot finger whose links are a different length cannot, so
#: demanding both is infeasible and the solver pays for it in penetration.
#:
#: It RELOCATES the burial rather than removing it. Worst penetration by link,
#: at w_mid=0.6, without and with the release:
#:
#:      reference            tip   mid  prox        tip   mid  prox
#:      binoculars_see_1    5.86  0.00  0.00  ->   6.47  4.32  0.00
#:      flashlight_on_2     5.02  1.74 12.54  ->   4.20  5.59  6.80
#:      mug_drink_1         8.21  2.19  0.00  ->   3.99  5.95  0.00
#:
#: The tip comes out (mug 8.2 -> 4.0, below even the w_mid=0 baseline of 5.0)
#: and the middle phalanx goes in. Worst-link depth improves past baseline only
#: on flashlight (8.1 -> 6.8); mug is neutral and binoculars is worse. Total
#: burial is roughly conserved, which is the finding: the middle phalanx cannot
#: reach this surface without burying SOMETHING.
#:
#: Nor is the wrist anchor to blame -- the obvious next suspect, since a Shadow
#: hand is bigger than the human hand that produced the demonstration and might
#: simply be unable to reach a second contact from the human's wrist. Sweeping
#: `w_wrist` down refutes it: the hand drifts off the object instead of
#: wrapping it, at w_mid=0.6,
#:
#:      reference          w_wrist  links  sided  pen mm
#:      binoculars_see_1     0.15      5   0.538    6.47
#:      binoculars_see_1     0.00      1   0.711    1.61
#:      mug_drink_1          0.15      6   0.380    6.06
#:      mug_drink_1          0.00      1   0.487    5.80
#:
#: One link at high one-sidedness and low penetration is a hand touching the
#: object with a fingertip, not holding it.
#:
#: Taken together these three sweeps say the same thing from three directions:
#: tighten the middle target and links are bought with burial, release the tip
#: and the burial moves, release the wrist and the contact degenerates. Every
#: knob in this objective is a position, and a position objective has no term
#: that says HOLD THE OBJECT -- it can only say put these points there, which
#: one buried finger or one grazing fingertip both satisfy. That is this
#: project's founding claim reaching the retarget: the next term to add needs
#: force content (closure, a wrench the contact set can resist), not another
#: point target. It is also the fourth independent route to the same empty
#: neighbourhood, after a peer's 283-candidate sweep, the w_pen sweep, and the
#: wrap search.
MID_FREES_TIP = True

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
                 w_pen=W_PEN, tip_offsets=None, base_qpos=None,
                 w_wrist=W_WRIST):
        self.sc = sc
        self.m, self.d = sc.model, sc.data
        self.tips = list(tips)
        self.wrist = wrist_bid
        self.lo, self.hi = lo, hi
        self.dofs = np.array([sc.model.jnt_dofadr[j] for j in sc.jids], dtype=int)
        self.n = len(sc.jids)
        self.w_smooth, self.w_pen, self.w_wrist = w_smooth, w_pen, w_wrist
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
        # handsim.hands.tips derives the real point from the collision geometry;
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
        # Radius of each middle phalanx, so a contact target can be offset off
        # the surface by it. Targeting the BODY ORIGIN at the surface point
        # would bury the link by its own radius -- the identical defect that
        # `tip_offset` exists to prevent, and that cost this project 4469 N of
        # contact force once already.
        self.mid_rad = []
        for up in self.chain:
            r = 0.01
            if up:
                gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == up[0]]
                if gs:
                    r = float(max(m.geom_size[g][0] for g in gs))
            self.mid_rad.append(r)

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
              iters=12, damp=1e-3, joint_targets=None, weights_scale=None,
              joint_weights=None):
        """targets/weights are per ROBOT tip (already corresponded).

        `joint_targets[i]` is [middle, proximal] world positions for robot tip
        i, or None entries where the human has no counterpart.
        """
        q = np.clip(np.asarray(q0, float), self.lo, self.hi)
        self.n_joint_rows = 0        # so a caller can assert the term FIRED
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
                    w = (W_JOINT * weights_scale[i] if joint_weights is None
                         else float(joint_weights[i][lvl]))
                    if w <= 0:
                        continue
                    self.n_joint_rows += 1
                    rows.append(w * self._jac_body(bid))
                    res.append(w * (self.d.xpos[bid] - tg))
            if wrist_target is not None and self.w_wrist > 0:
                rows.append(self.w_wrist * self._jac_body(self.wrist))
                res.append(self.w_wrist * (self.d.xpos[self.wrist] - wrist_target))
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
                      w_mid: float = W_MID,
                      mid_frees_tip: bool = MID_FREES_TIP,
                      w_wrist: float = W_WRIST,
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
    from handsim.human.scene import build as build_scene

    sc = sc or build_scene(hand=hand, obj=seq.obj)
    human = seq.hands[side]
    if human.verts is None:
        raise ValueError("load the sequence with verts=True")
    ov, of = seq.obj_mesh
    tree = tree or cKDTree(ov)
    # Outward vertex normals, to offset a middle-phalanx contact target off the
    # surface rather than onto it.
    if w_mid > 0:
        import trimesh
        onrm = np.asarray(trimesh.Trimesh(ov, of, process=False).vertex_normals)
    else:
        onrm = None

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
                     w_smooth=w_smooth, w_pen=w_pen, base_qpos=base_qpos,
                     w_wrist=w_wrist)

    Q = np.zeros((len(frames), len(sc.jids)))
    tip_err = np.zeros(len(frames))
    con_err = np.zeros(len(frames))
    n_con = np.zeros(len(frames), int)
    n_mid = np.zeros(len(frames), int)
    rows_fired = 0

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
        jw = np.zeros((len(sc.tip_bids), 2))
        for a, b in zip(ri, hidx):
            ch = MANO_CHAINS[a]
            mid = (human.joints[k, ch[2], :] - p) @ R
            prox = (human.joints[k, ch[1], :] - p) @ R
            jt[b] = [mid, prox]
            jw[b] = (W_JOINT, W_JOINT)
            if w_mid > 0:
                # Did the HUMAN's own middle phalanx engage this object? If so,
                # ask the robot's to touch the surface there -- offset outward
                # by the link radius, since the target drives the body origin.
                dm, im = tree.query(mid[None], k=1)
                if dm[0] < contact_tol:
                    jt[b][0] = ov[im[0]] + onrm[im[0]] * solver.mid_rad[b]
                    jw[b, 0] = w_mid
                    n_mid[t] += 1
                    if mid_frees_tip and weights[b] >= W_CONTACT:
                        # This finger contacts at BOTH tip and middle. A human
                        # satisfies that by wrapping around curvature; a Shadow
                        # finger has different link lengths, so pinning both to
                        # the surface is over-constrained and least squares
                        # puts the residual into penetration. Release the tip
                        # to its free-space target and let the middle lead.
                        weights[b] = W_FREE
                        targets[b] = tips_o[a]

        wrist_t = (human.joints[k, 0, :] - p) @ R
        q, got = solver.solve(q, targets, weights, wrist_t, q_prev, iters=iters,
                              joint_targets=jt, joint_weights=jw)
        rows_fired += solver.n_joint_rows
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
              "w_pen": w_pen, "w_smooth": w_smooth, "w_mid": w_mid,
              "n_mid": n_mid, "mid_rows_fired": int(rows_fired)},
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
    from handsim.human.scene import build_bimanual

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


def _arm_gap(sc):
    """(worst arm-side penetration, min fingertip-to-surface gap) in metres."""
    m, d = sc.model, sc.data
    objset = set(sc.obj_gids)
    armset = set(sc.hand_gids) - set(sc.tip_gids)
    pen = 0.0
    for c in range(d.ncon):
        pair = {int(d.contact[c].geom1), int(d.contact[c].geom2)}
        if pair & objset and pair & armset:
            pen = max(pen, -float(d.contact[c].dist))
    gap = float("inf")
    for tb in sc.tip_bids:
        for g in sc.obj_gids:
            gap = min(gap, float(np.linalg.norm(d.xpos[tb] - d.geom_xpos[g]))
                      - float(m.geom_rbound[g]))
    return pen, gap


def orient_to_clear(sc, q, rot_deg=40.0, n_rot=7, advances_mm=(-20.0, 0.0, 20.0),
                    w_pen=10.0):
    """Rotate the wrist so the FINGERS reach the object before the arm does.

    The fit solves fingertip positions and wrist POSITION; `W_PEN_ARM`
    constrains where the arm may be. Wrist ORIENTATION is inherited from the
    human and never searched -- and it is what decides which part of the hand
    arrives first. Measured on `mug_drink_1`, the arm penetrates at 0.48 mm
    while the fingertips are still 42.9 mm away, so the arm is already the
    binding contact and no translation helps: `advance_to_contact` moves 0.0 mm
    on every reference tried, because advancing drives the arm deeper first.

    Sweeping rotation changes that. Over +/-40 deg on each wrist axis crossed
    with a small advance, `binoculars_see_1` and `flashlight_on_2` both reach
    **0.00 mm** of arm penetration with the fingertips in contact (47/1029 and
    21/1029 poses qualify), at rotations of 12-40 deg -- within 40 deg of where
    the fit was already putting the wrist. The advance matters but only once
    rotation has stopped the arm leading: +20 mm is chosen in all three, yet
    translation alone buys nothing.

    Done in the FITTING scene, which drives the hand through six hinge/slide
    joints with no mocap body and no weld, so there is no qpos/mocap/weld
    consistency to get wrong -- set the joints, call mj_forward, read contacts.

    Scored by `|gap| + w_pen * max(0, pen - 1 mm)`: bring the fingertips TO the
    surface and pay heavily for arm penetration past the 1 mm tolerance the soft
    arm constraint cannot go below anyway. The absolute value is load-bearing --
    scoring the signed gap rewards driving the fingers deeper into the object,
    which is the defect this whole stage exists to remove. Measured: with the
    signed form, `camera_takepicture_2` went from 6.3 mm of fingertip
    penetration to 23.8 mm and scored it an improvement.

    Returns (q_best, pen_m, gap_m). Leaves `sc` at the returned configuration.
    """
    m, d = sc.model, sc.data
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in sc.jids]
    o = {n: i for i, n in enumerate(names) if n in ("x", "y", "z", "rx", "ry", "rz")}
    q = np.array(q, float)
    if len(o) < 6:
        sc.set_q(q)
        pen, gap = _arm_gap(sc)
        return q, pen, gap

    def score(pen, gap):
        # abs(gap): touching is the target, not burial. See the docstring.
        return abs(gap) + w_pen * max(0.0, pen - 0.001)

    sc.set_q(q)
    pen, gap = _arm_gap(sc)
    best = (score(pen, gap), q.copy(), pen, gap)

    grid = np.deg2rad(np.linspace(-rot_deg, rot_deg, n_rot))
    for rx in grid:
        for ry in grid:
            for rz in grid:
                base = q.copy()
                base[o["rx"]] += rx
                base[o["ry"]] += ry
                base[o["rz"]] += rz
                sc.set_q(base)
                u = -d.xpos[sc.wrist_bid].copy()      # object sits at the origin
                n = float(np.linalg.norm(u))
                if n < 1e-6:
                    continue
                u /= n
                for adv in advances_mm:
                    cand = base.copy()
                    cand[o["x"]] += u[0] * adv / 1000.0
                    cand[o["y"]] += u[1] * adv / 1000.0
                    cand[o["z"]] += u[2] * adv / 1000.0
                    sc.set_q(cand)
                    pen, gap = _arm_gap(sc)
                    sc_ = score(pen, gap)
                    if sc_ < best[0]:
                        best = (sc_, cand.copy(), pen, gap)
    sc.set_q(best[1])
    return best[1], best[2], best[3]


def advance_to_contact(sc, q, max_mm=90.0, step_mm=2.0, clear_mm=0.5):
    """Slide the hand along its approach axis until an ARM-side body would touch.

    Done in the FITTING scene, which drives the hand through six hinge/slide
    joints with no mocap body and no weld. That matters: three earlier attempts
    at this in the simulation scene all died on keeping `qpos`, the mocap target
    and the weld consistent, and none of that exists here -- set the joints, call
    mj_forward, read the contacts.

    Why it is needed: the arm-side feasibility constraint and the fingertip
    targets share one wrist, so pushing the forearm out of the object pushes the
    fingers out with it. Measured, the tips end 50-62 mm from the surface while
    the closing routine moves them 30-40 mm. The wrist has to be PLACED, and the
    binding constraint -- the arm touching -- is the right thing to terminate on,
    because advancing as far as it allows is exactly what minimises the gap the
    fingers must close.

    Returns (q_advanced, distance_m).
    """
    m, d = sc.model, sc.data
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in sc.jids]
    xyz = [i for i, n in enumerate(names) if n in ("x", "y", "z")]
    if len(xyz) != 3:
        return np.asarray(q), 0.0
    order = {names[i]: i for i in xyz}
    idx = [order["x"], order["y"], order["z"]]

    objset = set(sc.obj_gids)
    armset = set(sc.hand_gids) - set(sc.tip_gids)

    q = np.array(q, float)
    sc.set_q(q)
    palm = d.xpos[sc.wrist_bid].copy()
    # the object is static at the origin in the fitting scene
    u = -palm
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return q, 0.0
    u /= n

    base = q[idx].copy()
    step = step_mm / 1000.0
    best = 0.0
    for i in range(1, int(max_mm / step_mm) + 1):
        q[idx] = base + u * (i * step)
        sc.set_q(q)
        pen = 0.0
        for c in range(d.ncon):
            pair = {int(d.contact[c].geom1), int(d.contact[c].geom2)}
            if pair & objset and pair & armset:
                pen = max(pen, -float(d.contact[c].dist))
        if pen > clear_mm / 1000.0:
            break
        best = i * step
    q[idx] = base + u * best
    sc.set_q(q)
    return q, best


def approach_opposition(sc, q, tip_offsets=None, max_gap=0.030):
    """How OPPOSED the fingertips are about the object, with no contact needed.

    The orientation search that fixed arm clearance scores minimum
    fingertip-to-surface distance, and distance cannot tell a hand poised to
    grasp from one merely adjacent: it took `flashlight_on_2` and `cup_lift` to
    tips 1.5-3.1 mm off the surface with ZERO contacts and exact free fall, and
    closing from there engages nothing because the fingers are near the surface
    and not facing it.

    This measures facing. For each fingertip, take the nearest point on the
    object and the outward surface normal there; a hand that can grasp has those
    normals pointing in opposed directions, one that is merely adjacent has them
    all pointing the same way. Returns |mean unit normal|, so 0 is fully opposed
    and 1 is every finger on the same side -- the same convention as
    `one_sidedness`, which needs contacts and therefore cannot be used before
    the hand is touching.
    """
    from handsim.hands.tips import tip_offset
    from scipy.spatial import cKDTree

    m, d = sc.model, sc.data
    if tip_offsets is None:
        tip_offsets = [tip_offset(m, b) for b in sc.tip_bids]
    sc.set_q(q)

    pts = []
    for g in sc.obj_gids:
        g = int(g)
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = m.mesh_vert[a:a + n].astype(np.float64)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        pts.append(v @ R.reshape(3, 3).T + m.geom_pos[g])
    if not pts:
        return 1.0
    P = np.concatenate(pts)
    tree = cKDTree(P)
    com = P.mean(0)

    acc, used = np.zeros(3), 0
    for b, off in zip(sc.tip_bids, tip_offsets):
        tip = d.xpos[b] + d.xmat[b].reshape(3, 3) @ off
        dist, idx = tree.query(tip, k=1)
        if dist > max_gap:
            continue
        # outward normal approximated from the object's centroid; exact enough
        # for a direction test and free of a normal-buffer lookup
        nrm = P[idx] - com
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        acc += nrm / nn
        used += 1
    if used < 2:
        return 1.0
    return float(np.linalg.norm(acc) / used)
