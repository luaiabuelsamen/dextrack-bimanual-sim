"""Retargeting a human grasp onto a robot hand, under a choice of objective.

This is the project's central tool and it did not exist. The thesis is that the
objective everyone uses -- match the human's keypoints -- is the wrong one for a
hand that cannot oppose, and that the right one is stated in the object's wrench
space. Both are implemented here so the claim is a measurement rather than an
argument.

    KEYPOINT   minimise |wrist->fingertip vectors - the human's|
               what DexPilot / AnyTeleop / dex-retargeting optimise, and what
               DexTrack's MANO->Shadow step optimises

    EPSILON    maximise the Ferrari-Canny epsilon of the contact set the pose
               produces, ignoring what the human's hand looked like

    BLEND      (1-w)*keypoint + w*epsilon, so the trade-off can be swept rather
               than argued about

Contacts are estimated GEOMETRICALLY -- nearest point on the object surface to
each fingertip, with that surface's normal -- rather than by stepping physics
inside the optimiser. That is the same approximation grasp synthesis uses (BODex
optimises a differentiable force-closure energy on predicted contacts); the
result is a pose to be VALIDATED in simulation afterwards, never a claim on its
own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco
from scipy.optimize import minimize

from oppdef.grasping.epsilon import epsilon_from_wrenches, wrench_set

KEYPOINT, EPSILON, BLEND = "keypoint", "epsilon", "blend"


# --------------------------------------------------------------------------
# geometric contact estimate
# --------------------------------------------------------------------------
def box_surface(p, half):
    """Nearest point on a box surface to p, and that face's outward normal.

    Both defined in the box frame. For a point inside the box the nearest face
    is used, which is what a fingertip pressing in should see.
    """
    half = np.asarray(half, float)
    clamped = np.clip(p, -half, half)
    inside = np.all(np.abs(p) <= half)
    if not inside:
        n = p - clamped
        d = np.linalg.norm(n)
        return clamped, (n / d if d > 1e-12 else np.array([0.0, 0, 1.0])), d
    # inside: push out through the closest face
    slack = half - np.abs(p)
    ax = int(np.argmin(slack))
    n = np.zeros(3)
    n[ax] = np.sign(p[ax]) or 1.0
    q = p.copy()
    q[ax] = np.sign(p[ax]) * half[ax] if p[ax] != 0 else half[ax]
    return q, n, -float(slack[ax])


def contacts_from_tips(tips, obj_half, obj_pos=None, tol=0.008):
    """Which fingertips are touching the box, where, and with what normal.

    Returns (points, normals-into-the-object, mask). A fingertip beyond `tol`
    from the surface is not a contact, which is what makes a pose that merely
    resembles the human's -- without reaching the object -- score zero.
    """
    obj_pos = np.zeros(3) if obj_pos is None else np.asarray(obj_pos, float)
    pts, nrm, mask = [], [], []
    for t in np.asarray(tips, float):
        local = t - obj_pos
        q, n_out, dist = box_surface(local, obj_half)
        touching = dist <= tol
        mask.append(touching)
        if touching:
            pts.append(q + obj_pos)
            nrm.append(-n_out)          # into the object: the push direction
    return (np.array(pts).reshape(-1, 3), np.array(nrm).reshape(-1, 3),
            np.array(mask, bool))


def geometric_epsilon(tips, obj_half, mu=1.0, obj_pos=None, tol=0.008):
    pts, nrm, mask = contacts_from_tips(tips, obj_half, obj_pos, tol)
    if len(pts) < 2:
        return 0.0, int(mask.sum())
    lam = float(np.linalg.norm(obj_half)) or 1.0
    com = np.zeros(3) if obj_pos is None else np.asarray(obj_pos, float)
    W = wrench_set(pts, nrm, [mu] * len(pts), com, lam)
    return epsilon_from_wrenches(W), int(mask.sum())


def reach_energy(tips, obj_half, obj_pos=None, tol=0.008, pen=0.004):
    """Smooth distance-to-object energy, and a penetration penalty.

    Epsilon alone cannot be optimised by a gradient method: with no fingertip
    touching, epsilon is EXACTLY zero in every direction, so a numerical
    gradient is zero and the search terminates where it started.  That is not a
    tuning problem, it is the shape of the function -- the first run of this
    module reported 0 contacts for every hand for precisely this reason.

    So the objective carries the same three terms grasp synthesis uses
    (DexGraspNet / BODex: E_fc + E_dis + E_pen): epsilon supplies the quality,
    `reach` supplies a gradient that pulls tips onto the surface before any
    contact exists, and `pierce` keeps them from being driven through it.
    """
    obj_pos = np.zeros(3) if obj_pos is None else np.asarray(obj_pos, float)
    reach = pierce = 0.0
    for t in np.asarray(tips, float):
        _q, _n, d = box_surface(t - obj_pos, obj_half)
        reach += max(d - tol, 0.0) ** 2
        pierce += max(-d - pen, 0.0) ** 2
    n = max(len(tips), 1)
    return reach / n, pierce / n


def correspond(n_ref, n_hand):
    """Index arrays pairing reference tips with hand tips, thumb to thumb.

    Both are ordered fingers-then-thumb, so the pairing is the first k fingers
    plus the thumb -- NOT the first k entries, which silently pairs a five-
    fingered reference's little finger with a four-fingered hand's thumb.
    Hands differ in finger count (LEAP 4, Shadow 5), so this cannot be skipped.
    """
    k = min(n_ref - 1, n_hand - 1)
    ri = list(range(k)) + [n_ref - 1]
    hi = list(range(k)) + [n_hand - 1]
    return np.array(ri), np.array(hi)


def tip_graph(tips):
    """Vectors BETWEEN fingertips: thumb->each finger, and finger->neighbour.

    Keypoint retargeting is usually written against wrist->fingertip vectors,
    and for a single hand that is fine. Across hands it is not: the body a model
    calls the wrist is a modelling choice, and these disagree by more than the
    hand -- Shadow's palm body sits 25 cm from its own fingertips (it includes
    the forearm), LEAP's 8 cm. Scaling a wrist-relative offset therefore places
    the object somewhere it was never demonstrated; rendered, the box came out
    buried inside LEAP's palm.

    Inter-fingertip vectors have no origin to disagree about, which is why
    DexPilot and AnyTeleop optimise these rather than wrist-relative ones.
    """
    tips = np.asarray(tips, float)
    n = len(tips)
    pairs = [(i, n - 1) for i in range(n - 1)] + [(i, i + 1) for i in range(n - 2)]
    return np.stack([tips[b] - tips[a] for a, b in pairs]), pairs


def align_reference(ref_tips, hand_tips, ref_wrist=None, hand_wrist=None,
                    w_palm=2.0):
    """Rotation carrying the human's hand shape onto this robot's. Rotation only.

    An earlier version also fitted a scale, and every way of fitting one was
    wrong. Least squares collapsed toward zero when the two shapes differed
    (0.50 for LEAP, whose fingers are ~0.8x the reference's, which buried the
    object 2 cm inside the palm). A span ratio gave 3.26, because it compared a
    CLOSED human grasp against the robot at its zero posture, fully open.

    There is no need to fit one: the demonstration grasped a box of known width,
    so asking the hand for a box of width w scales the reference by exactly
    w / w_ref. Scale becomes a swept experimental variable rather than a fudge
    factor -- see `transform_ref`.

    The rotation is fitted on inter-fingertip vectors normalised to unit mean
    length, at the hand's DERIVED closure rather than its open posture, so a
    closed human grasp is compared against a closed robot one.
    """
    ri, hi = correspond(len(ref_tips), len(hand_tips))
    A, _ = tip_graph(np.asarray(ref_tips, float)[ri])
    B, _ = tip_graph(np.asarray(hand_tips, float)[hi])
    A = A / max(np.linalg.norm(A, axis=1).mean(), 1e-9)
    B = B / max(np.linalg.norm(B, axis=1).mean(), 1e-9)
    if ref_wrist is not None and hand_wrist is not None:
        # Fingertips lie close to a plane, and inter-tip vectors are blind to
        # which SIDE of that plane the palm is on -- the one degree of freedom
        # that decides whether the fingers point at the object or away from it.
        # Fitted without this, LEAP's palm was placed above the box with its
        # fingers extending further above, and every keypoint fit hovered.
        # Each palm direction is unit-normalised, so it votes on orientation
        # without reintroducing a scale.
        a = np.asarray(ref_tips, float)[ri].mean(0) - np.asarray(ref_wrist, float)
        b = np.asarray(hand_tips, float)[hi].mean(0) - np.asarray(hand_wrist, float)
        a = a / max(np.linalg.norm(a), 1e-9)
        b = b / max(np.linalg.norm(b), 1e-9)
        A = np.vstack([A, w_palm * a[None, :]])
        B = np.vstack([B, w_palm * b[None, :]])
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# --------------------------------------------------------------------------
# the retargeter
# --------------------------------------------------------------------------
@dataclass
class Result:
    q: np.ndarray
    objective: str
    keypoint_err_m: float
    epsilon: float
    n_contacts: int
    weight: float = 0.0


class Retargeter:
    """Fits one robot hand's joints to one demonstrated grasp."""

    def __init__(self, model, tip_bids, joint_ids, wrist_bid=None):
        self.m, self.d = model, mujoco.MjData(model)
        self.tips = list(tip_bids)
        self.wrist = wrist_bid if wrist_bid is not None else self.tips[0]
        self.jids = list(joint_ids)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        r = model.jnt_range[self.jids]
        lo, hi = r[:, 0].copy(), r[:, 1].copy()
        bad = hi <= lo
        lo[bad], hi[bad] = -np.pi, np.pi
        self.lo, self.hi = lo, hi
        #: a posture known to close the hand, used as an optimiser seed; set by
        #: `retargeter_for` from the DERIVED closure, never from a table
        self.q_closure = None
        self._align = None
        #: rebuilds this hand's spec, so a fitted pose can be RENDERED with the
        #: object added to the same model the epsilon was computed against
        self.spec_fn = None
        self.hand_key = None
        self.prefix = ""

    # -- reference frame calibration ---------------------------------------
    def calibrate(self, ref_tips, ref_wrist=None):
        """Fit (R, s) mapping the reference's hand shape onto this hand's.

        `ref_tips` may be positions or wrist-relative vectors: the fit runs on
        inter-fingertip vectors, which are translation-invariant either way.
        """
        q = self.q_closure if self.q_closure is not None \
            else np.zeros(len(self.jids))
        tips, wrist = self.forward(np.clip(q, self.lo, self.hi))
        self._align = align_reference(ref_tips, tips, ref_wrist=ref_wrist,
                                      hand_wrist=wrist)
        return self._align

    def target_graph(self, ref_tips, scale=1.0):
        """The inter-fingertip vectors this hand is asked to reproduce."""
        if self._align is None:
            self.calibrate(ref_tips)
        ri, _hi = correspond(len(ref_tips), len(self.tips))
        A, _ = tip_graph(np.asarray(ref_tips, float)[ri])
        return scale * (A @ self._align)

    def graph_error(self, tips, ref_tips, scale=1.0):
        """Mean inter-fingertip vector error, in metres."""
        _ri, hi = correspond(len(ref_tips), len(tips))
        B, _ = tip_graph(np.asarray(tips, float)[hi])
        return float(np.linalg.norm(B - self.target_graph(ref_tips, scale),
                                    axis=1).mean())

    def forward(self, q):
        self.d.qpos[:] = 0.0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.tips].copy(), self.d.xpos[self.wrist].copy()

    def keypoint_error(self, q, ref_tips, scale=1.0):
        tips, _wrist = self.forward(q)
        return self.graph_error(tips, ref_tips, scale)

    def fit(self, ref_vectors, obj_half, objective=KEYPOINT, weight=0.5, mu=1.0,
            obj_pos=None, restarts=6, seed=0, maxiter=400, tol=0.008,
            w_reach=40.0, w_pen=200.0, scale=1.0, wrist_target=None,
            w_wrist=1.0, seed_q=None):
        """Fit joints to one grasp under `objective`.

        `w_reach`/`w_pen` only shape the SEARCH; the epsilon and contact count
        returned are measured on the final pose by the same hard geometric test
        used everywhere else, so a pose that merely gets close scores nothing.

        `wrist_target` is where the demonstration puts the palm relative to the
        object. The keypoint objective NEEDS it: inter-fingertip vectors are
        origin-free, so on their own they reproduce the human's finger shape
        while leaving the hand free to float anywhere -- a fit that matched the
        demonstrated shape to 3 mm made just one contact, because nothing had
        told it where the object was. Supplying it is also what the real
        pipelines do (DexPilot takes the wrist pose from the human and
        retargets only the fingers).
        """
        rng = np.random.default_rng(seed)
        ref = np.asarray(ref_vectors, float)
        _ri, hi = correspond(len(ref), len(self.tips))
        A_ref = self.target_graph(ref, scale)

        wt = None if wrist_target is None else np.asarray(wrist_target, float)

        def cost(q):
            tips, wrist = self.forward(q)
            B, _ = tip_graph(tips[hi])
            kp = float(np.linalg.norm(B - A_ref, axis=1).mean())
            if wt is not None:
                kp += w_wrist * float(np.linalg.norm(wrist - wt))
            if objective == KEYPOINT:
                return kp
            eps, _ = geometric_epsilon(tips, obj_half, mu, obj_pos, tol)
            reach, pierce = reach_energy(tips, obj_half, obj_pos, tol)
            shaped = -eps + w_reach * reach + w_pen * pierce
            if objective == EPSILON:
                return shaped
            return (1 - weight) * kp + weight * shaped

        # Seeds: the open hand, the derived closure (the only posture known a
        # priori to bring the fingers together), anything the caller supplies,
        # and random draws in range.
        #
        # `seed_q` matters more than it looks. Epsilon's landscape is flat
        # wherever nothing touches and ridged where things do, and from cold
        # starts the search kept losing to the BLEND objective -- a strict
        # impossibility if it were converging, since blend optimises epsilon
        # with a competing term attached. Handing it the keypoint solution also
        # makes the comparison the one worth running: not two unrelated
        # searches, but what refining a RETARGETED grasp for epsilon buys.
        seeds = [np.zeros(len(self.jids))]
        if seed_q is not None:
            for q0 in np.atleast_2d(seed_q):
                seeds.append(np.clip(np.asarray(q0, float), self.lo, self.hi))
        if self.q_closure is not None:
            seeds.append(np.clip(self.q_closure, self.lo, self.hi))
            seeds.append(np.clip(0.5 * self.q_closure, self.lo, self.hi))
        while len(seeds) < restarts:
            seeds.append(self.lo + rng.random(len(self.jids)) * (self.hi - self.lo))

        best = None
        for x0 in seeds[:max(restarts, len(seeds))]:
            r = minimize(cost, x0, method="L-BFGS-B",
                         bounds=list(zip(self.lo, self.hi)),
                         options=dict(maxiter=maxiter))
            if best is None or r.fun < best.fun:
                best = r
        q = np.clip(best.x, self.lo, self.hi)
        tips, _ = self.forward(q)
        eps, n = geometric_epsilon(tips, obj_half, mu, obj_pos, tol)
        return Result(q=q, objective=objective,
                      keypoint_err_m=self.keypoint_error(q, ref_vectors, scale),
                      epsilon=eps, n_contacts=n, weight=weight)


def transform_ref(rt, ref, width=None):
    """Carry a demonstration into this hand's frame, with the object placed.

    The alignment is rotation-only and origin-free, so it fixes the
    demonstration's SHAPE and ORIENTATION but not where it sits. Two things
    supply the rest:

    position -- a hand-intrinsic point, the midpoint of this hand's opposition
        axis at its derived closure (`grasp_centre`). It does not depend on the
        objective being fitted, so keypoint and epsilon are scored against the
        same object in the same place, which is the only way the comparison
        means anything.

    scale -- the demonstration held a box of a known width, so testing a box of
        width `width` scales the reference by exactly width / width_ref. Sweep
        `width` and the size relationship becomes a measurement rather than an
        assumption.

    Returns (reference fingertips, object position, object half-extents, the
    object scale, the demonstration's fingertips placed around that object, and
    its wrist placed at this hand's own finger length.).
    """
    V = np.asarray(ref.fingertips, float)
    half_ref = np.asarray(ref.obj_half, float)
    w_ref = 2.0 * float(half_ref[0])
    s = 1.0 if width is None else float(width) / w_ref
    if rt._align is None:
        rt.calibrate(V, ref_wrist=ref.wrist)
    obj = grasp_centre(rt)

    # Two scales, because two different quantities are being carried over and
    # they do not share a unit. How far apart the fingertips sit is set by the
    # OBJECT (s above): a 5 cm box needs the same span whoever holds it. How far
    # the palm sits behind them is set by the HAND -- its finger length -- and
    # using the object scale for it put LEAP's palm 10 cm from a box its 8 cm
    # fingers then could not reach, leaving the hand hovering above the object
    # in every keypoint fit.
    q_c = rt.q_closure if rt.q_closure is not None else np.zeros(len(rt.jids))
    tips_c, wrist_c = rt.forward(np.clip(q_c, rt.lo, rt.hi))
    reach_robot = float(np.linalg.norm(tips_c - wrist_c[None, :], axis=1).mean())
    reach_human = float(np.linalg.norm(V - np.asarray(ref.wrist, float)[None, :],
                                       axis=1).mean())
    s_hand = reach_robot / reach_human if reach_human > 1e-9 else 1.0

    o0 = np.asarray(ref.obj_pos, float)
    tips = obj + s * ((V - o0) @ rt._align)
    wrist = obj + s_hand * ((np.asarray(ref.wrist, float) - o0) @ rt._align)
    return V, obj, s * half_ref, s, tips, wrist


def squeeze(rt, q, obj_half, obj_pos, mu=1.0, tol=0.008, n=60):
    """Close the fingers from a fitted pose until they grip, and report the best.

    Without this the keypoint comparison is unfair, and obviously so: matching a
    human's fingertip geometry positions the fingers, it does not press them
    into anything, so a keypoint pose can match the demonstration to 5 mm and
    still touch nothing. No practitioner ships that pose -- Dexonomy's data is
    stored as a pre-grasp / grasp / SQUEEZE triple precisely because the squeeze
    is a separate step, and every closure written by hand in this project failed
    until it was structured the same way.

    So the fingers are driven from `q` toward the hand's DERIVED closure and the
    best epsilon along that path is returned. Only finger joints move; the base
    stays where the fit put it, because sliding the whole hand would be solving
    a different problem.
    """
    q = np.asarray(q, float).copy()
    if rt.q_closure is None:
        return q, *geometric_epsilon(rt.forward(q)[0], obj_half, mu, obj_pos, tol)
    base = {f"{rt.prefix}{d}" for d in ("x", "y", "z", "rx", "ry", "rz")}
    moving = np.array([
        (mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j) or "") not in base
        for j in rt.jids])
    target = np.clip(rt.q_closure, rt.lo, rt.hi)
    # Seeded with the pose as GIVEN, scored the same way, so a squeeze can only
    # ever improve it. Starting from (q, 0, 0) meant that when every candidate
    # along the path was rejected the function returned zero for a pose that
    # already had four contacts -- reporting a loss the squeeze did not cause.
    e0, n0 = geometric_epsilon(rt.forward(q)[0], obj_half, mu, obj_pos, tol)
    best = (q, e0, n0)
    for a in np.linspace(0.0, 1.0, n):
        qa = q.copy()
        qa[moving] = (1 - a) * q[moving] + a * target[moving]
        qa = np.clip(qa, rt.lo, rt.hi)
        tips, _w = rt.forward(qa)
        _pts, _nrm, mask = contacts_from_tips(tips, obj_half, obj_pos, tol)
        # reject poses that drive a fingertip deep into the object: a "grasp"
        # holding a box by passing through it is the reward hacking this
        # project has already had to fix once (NOTES, MPPI)
        deep = any(box_surface(t - np.asarray(obj_pos, float), obj_half)[2]
                   < -0.006 for t in tips)
        if deep:
            continue
        eps, nc = geometric_epsilon(tips, obj_half, mu, obj_pos, tol)
        if eps > best[1] or (eps == best[1] and nc > best[2]):
            best = (qa, eps, nc)
    return best


def grasp_centre(rt, q=None):
    """Where an object has to sit for this hand to be able to touch it.

    A retargeter moves JOINTS only -- it cannot walk the hand over to the
    object.  Leave the object at the world origin and every hand records zero
    contacts and the comparison measures nothing, which is exactly what the
    first run of this module did.

    The point returned is the midpoint of the hand's opposition axis at its
    closed posture: halfway between the thumb tip and the centroid of the
    other fingertips.  That is the same axis `oppdef.hands.axis` measures the
    opposition floor along, so object placement and floor stay consistent.
    """
    q = rt.q_closure if q is None else q
    if q is None:
        q = np.zeros(len(rt.jids))
    tips, _wrist = rt.forward(np.clip(q, rt.lo, rt.hi))
    if len(tips) < 2:
        return tips.mean(0)
    return 0.5 * (tips[-1] + tips[:-1].mean(0))


def retargeter_for(hand_key, prefix="", free_base=True):
    """Build a Retargeter for a registered hand, seeded with its derived closure.

    `free_base` gives the palm six position DoF. Without it the hand is bolted
    to the origin and the only way to reach the object is to contort the
    fingers, which scores the objectives on a task neither is meant to solve --
    on a robot the wrist pose comes from the arm.
    """
    from oppdef.embodiment import make, HANDS
    from oppdef.hands.specs import derive_flex
    emb = make(hand=hand_key, free_base=free_base)
    m = emb.model
    pfx = list(emb.palm_bid)[0]
    # Only the hand's own joints, plus the floating base. Selected by BODY
    # DESCENT rather than by name: f5d6's file is the whole Vega robot, and a
    # name-blind sweep handed the optimiser 47 joints including the head, the
    # torso lift and the LEFT hand. Descent also survives every hand's own
    # naming scheme, which no prefix rule does.
    palm = emb.palm_bid[pfx]
    base = {f"{pfx}{d}" for d in ("x", "y", "z", "rx", "ry", "rz")}

    def in_hand(j):
        b = int(m.jnt_bodyid[j])
        while b > 0:
            if b == palm:
                return True
            b = int(m.body_parentid[b])
        return False

    jids = [j for j in range(m.njnt)
            if m.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_FREE,
                                     mujoco.mjtJoint.mjJNT_BALL)
            and (in_hand(j)
                 or (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
                 in base)]
    rt = Retargeter(m, emb.tip_bid[pfx], jids, wrist_bid=emb.palm_bid[pfx])

    # The closing posture is DERIVED (specs.derive_flex solves the closure over
    # all joints at once); here it is only re-indexed into this model's joint
    # order so it can seed the optimiser.
    _m, _cfg, flex, _tips, _gap = derive_flex(hand_key)
    q = np.zeros(len(jids))
    for i, j in enumerate(jids):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        base = n[len(pfx):] if pfx and n.startswith(pfx) else n
        if base in flex:
            q[i] = flex[base]
    rt.q_closure = q
    rt.hand_key = hand_key
    if free_base:
        # `_add_base_dof` ranges come from the bimanual env, where the hand
        # starts already pointed at the task: the base HINGES are limited to
        # +/-0.06 rad, about three degrees. A retargeter has to be able to turn
        # the hand over to face an object, so the optimiser's bounds are opened
        # here. Only the search bounds change -- the model is untouched, so the
        # env keeps the limits its results were measured under.
        for i, j in enumerate(rt.jids):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if n in ("rx", "ry", "rz"):
                rt.lo[i], rt.hi[i] = -np.pi, np.pi
    rt.spec_fn = lambda: make(hand=hand_key, free_base=free_base).spec
    return rt


class Penetration:
    """How far the WHOLE hand passes through the object at a pose.

    `geometric_epsilon` knows only about fingertips. It reads their positions,
    projects them onto the object surface and builds a wrench set -- and is
    blind to every other link. Optimised against it, the search happily drives
    the proximal and middle phalanges straight through the box: poses scoring
    epsilon = 0.36 were found, on inspection in MuJoCo, to be interpenetrating
    the object by 19 mm at 44 contact points, and to fling it away the instant
    physics is switched on.

    Fingertip-only penetration terms do not catch this, which is why the energy
    used by DexGraspNet and BODex has a whole-hand E_pen and not a fingertip
    one. This class supplies it, using MuJoCo's own collision detection against
    a static copy of the object, so the penalty is exact rather than a proxy.
    """

    def __init__(self, rt, obj_half, obj_pos, hand_key=None):
        from oppdef.embodiment import make, HANDS
        hand_key = hand_key or rt.hand_key
        emb = make(hand=hand_key, free_base=rt.prefix != "" or False)
        spec = emb.spec
        b = spec.worldbody.add_body(name="pen_obj",
                                    pos=[float(x) for x in obj_pos])
        b.add_geom(name="pen_obj_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[float(x) for x in obj_half], mass=0.05)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                     "pen_obj_geom")
        h = HANDS[hand_key]
        self.tip_bodies = {
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                              f"{rt.prefix}{t}") for t in h.tip_names}
        self.qadr = np.array([
            self.m.jnt_qposadr[mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT,
                mujoco.mj_id2name(rt.m, mujoco.mjtObj.mjOBJ_JOINT, j))]
            for j in rt.jids])

    def depths(self, q):
        """(deepest non-fingertip penetration, deepest fingertip), metres >= 0."""
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        body_pen = tip_pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != self.gid and c.geom2 != self.gid:
                continue
            other = c.geom2 if c.geom1 == self.gid else c.geom1
            depth = max(-float(c.dist), 0.0)
            if int(self.m.geom_bodyid[other]) in self.tip_bodies:
                tip_pen = max(tip_pen, depth)
            else:
                body_pen = max(body_pen, depth)
        return body_pen, tip_pen
