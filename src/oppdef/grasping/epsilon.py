"""Ferrari-Canny epsilon-metric over MuJoCo contact states.

epsilon = radius of the largest origin-centred ball inscribed in the convex hull
of the grasp wrench space (Ferrari & Canny, ICRA 1992). epsilon > 0 iff the
contact set can resist a wrench in every direction, i.e. force closure.

Conventions fixed here and not changed later without a NOTES.md entry:

- Friction cones are linearised at `n_edges` = 8 edges, Coulomb coefficient taken
  from the contact's own `friction[0]` (so it inherits the scene, not a constant).
- Each cone edge is scaled to UNIT NORMAL COMPONENT, not unit magnitude. This is
  the convention that makes epsilon monotone in mu (a larger friction cone can
  only ever help) and that makes `epsilon * F_total` meaningful when F_total is a
  sum of measured normal forces. Normalising edges to unit *length* instead makes
  epsilon non-monotone in mu, because the normal component then shrinks as the
  cone widens -- checked, and it is why that convention is not used here.
- Torques are divided by a characteristic length `lam` so the 6-vector has
  commensurate units. `lam` is the object's bounding radius, which depends on the
  object only -- NOT on the contact set -- so epsilon stays comparable across
  configurations of the same object.
- L1 form: the wrench space is the convex hull of the union of the primitive
  wrenches, i.e. unit total contact force distributed over the contacts. This is
  the standard epsilon_L1.

Because force is normalised out, epsilon alone does not say whether a given mass
can be held. That takes the force the hand actually applies, which we measure
from the simulator rather than model -- see `grasp_metrics`.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial import ConvexHull, QhullError

N_EDGES = 8
# Below this, the inscribed ball is qhull noise, not force closure.
EPS_TOL = 1e-9


def _cone_edges(normal, mu, n_edges=N_EDGES):
    """Unit force directions spanning the linearised friction cone."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    # any tangent basis
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    th = np.arange(n_edges) * 2.0 * np.pi / n_edges
    # unit NORMAL component (see module docstring), not unit magnitude
    return n[None, :] + mu * (np.cos(th)[:, None] * t1[None, :]
                              + np.sin(th)[:, None] * t2[None, :])


def wrench_set(points, normals, mus, com, lam, n_edges=N_EDGES):
    """Primitive wrenches for a contact set. points/normals in world frame;
    `normals` point INTO the object (the direction the contact can push it)."""
    W = []
    for p, n, mu in zip(points, normals, mus):
        for f in _cone_edges(n, mu, n_edges):
            tau = np.cross(np.asarray(p) - com, f) / lam
            W.append(np.concatenate([f, tau]))
    return np.asarray(W)


def epsilon_from_wrenches(W):
    """Radius of the largest origin-centred ball inside conv(W union {0}).

    Returns 0.0 when the origin is not interior, which is the force-closure
    test. Degenerate (lower-dimensional) hulls also return 0.0: a contact set
    whose wrench space does not span R^6 cannot contain a 6-ball.
    """
    if W is None or len(W) < 7:
        return 0.0
    pts = np.vstack([W, np.zeros((1, 6))])
    for opts in (None, "QJ"):
        try:
            hull = ConvexHull(pts, qhull_options=opts)
            break
        except QhullError:
            hull = None
    if hull is None:
        return 0.0
    # hull.equations rows are [a (unit normal) | b] with a.x + b <= 0 inside.
    a, b = hull.equations[:, :6], hull.equations[:, 6]
    na = np.linalg.norm(a, axis=1)
    ok = na > 1e-12
    if not ok.any():
        return 0.0
    dist = -b[ok] / na[ok]          # signed distance origin -> facet plane
    eps = float(dist.min())
    return eps if eps > EPS_TOL else 0.0


def wrench_hull(W):
    """Facets of conv(W u {0}) as (a, b), with a.x + b <= 0 inside, or None."""
    if W is None or len(W) < 7:
        return None
    pts = np.vstack([W, np.zeros((1, 6))])
    for opts in (None, "QJ"):
        try:
            hull = ConvexHull(pts, qhull_options=opts)
            break
        except QhullError:
            hull = None
    if hull is None:
        return None
    return hull.equations[:, :6], hull.equations[:, 6]


def support_along(hull, d):
    """Largest magnitude of `d` the contact set can resist, per unit contact force.

    Epsilon is the radius of the largest ball inside the wrench hull -- the
    worst case over EVERY direction. This is the same hull queried along ONE
    direction, which is what a task needs: a grasp only has to resist the
    wrenches its task actually demands, and those may lie in a narrow cone that
    the worst-case radius says nothing about.
    """
    if hull is None:
        return 0.0
    a, b = hull
    d = np.asarray(d, float)
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        return float("inf")
    u = d / nd
    proj = a @ u
    ok = proj > 1e-12
    if not ok.any():
        return float("inf")          # unbounded in this direction
    lam = (-b[ok] / proj[ok]).min()
    return float(max(lam, 0.0))


def task_margin(W, required, f_total=1.0):
    """How much headroom the contact set has against a required wrench sequence.

    Returns min over the sequence of (resistible / required). Below 1 means the
    task demands a wrench this grasp cannot supply. `required` rows are 6-D
    wrenches in the same frame and scaling as `wrench_set` produced W.
    """
    hull = wrench_hull(W)
    if hull is None:
        return 0.0
    worst = float("inf")
    for w in np.atleast_2d(required):
        need = float(np.linalg.norm(w))
        if need < 1e-9:
            continue
        # the contacts must resist w, i.e. supply -w
        cap = support_along(hull, -w) * float(f_total)
        worst = min(worst, cap / need)
    return 0.0 if worst == float("inf") else float(worst)


# --------------------------------------------------------------------------
# MuJoCo extraction
# --------------------------------------------------------------------------

def object_contacts(model, data, obj_geom_id, exclude_geoms=()):
    """Contacts on the object, with normals oriented INTO the object.

    Returns (points, normals, mus, normal_forces). `exclude_geoms` drops
    contacts with e.g. the table, so the grasp is scored on the hand alone.
    """
    P, N, MU, F = [], [], [], []
    ex = set(exclude_geoms)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if obj_geom_id not in (g1, g2):
            continue
        other = g2 if g1 == obj_geom_id else g1
        if other in ex:
            continue
        # MuJoCo contact frame: first row of the 3x3 is the normal, pointing
        # from geom1 toward geom2.
        n = np.array(c.frame[:3], dtype=float)
        if g2 == obj_geom_id:
            pass            # already points into the object
        else:
            n = -n          # object is geom1; flip so it points into it
        fc = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, fc)
        P.append(np.array(c.pos, dtype=float))
        N.append(n)
        MU.append(float(c.friction[0]))
        F.append(abs(float(fc[0])))
    return (np.asarray(P).reshape(-1, 3), np.asarray(N).reshape(-1, 3),
            np.asarray(MU), np.asarray(F))


def grasp_metrics(model, data, obj_geom_id, obj_body_id, exclude_geoms=(),
                  n_edges=N_EDGES):
    """epsilon plus the measured quantities needed to turn it into a deficit.

    delta = m*g / (epsilon * F_total): the gravity wrench divided by the wrench
    this grasp can actually resist. delta < 1 predicts the object is held.
    F_total is MEASURED (summed contact normal forces), not modelled.
    """
    P, N, MU, F = object_contacts(model, data, obj_geom_id, exclude_geoms)
    com = np.array(data.xipos[obj_body_id], dtype=float)
    lam = float(np.linalg.norm(model.geom_size[obj_geom_id]))
    lam = lam if lam > 1e-6 else 1.0
    out = dict(n_contacts=int(len(P)), epsilon=0.0, f_total=float(F.sum()) if len(F) else 0.0,
               lam=lam, mass=float(model.body_mass[obj_body_id]))
    if len(P) >= 2:
        W = wrench_set(P, N, MU, com, lam, n_edges)
        out["epsilon"] = epsilon_from_wrenches(W)
    w_req = out["mass"] * 9.81
    resist = out["epsilon"] * out["f_total"]
    out["w_required"] = w_req
    out["resistible"] = resist
    out["delta"] = (w_req / resist) if resist > 1e-9 else float("inf")
    return out


# --------------------------------------------------------------------------
# Self-test: analytic cases with known answers.
# --------------------------------------------------------------------------

def _selftest():
    lam = 0.05
    com = np.zeros(3)
    cases = []

    # 1. Two antipodal contacts, mu=1, on +-x faces. Force closure in the plane
    #    but NOT in R^6: no contact can resist a torque about the line joining
    #    them, so epsilon must be 0.
    P = [[0.04, 0, 0], [-0.04, 0, 0]]
    N = [[-1, 0, 0], [1, 0, 0]]
    cases.append(("2 antipodal point contacts", epsilon_from_wrenches(
        wrench_set(P, N, [1.0, 1.0], com, lam)), "0"))

    # 2. The same two faces but each contact spread into 3 points (a pad, like a
    #    palm) -- now torque about the x-axis IS resisted. Expect epsilon > 0.
    P, N = [], []
    for sx in (+1, -1):
        for dy, dz in ((0, 0.03), (0.03, -0.02), (-0.03, -0.02)):
            P.append([sx * 0.04, dy, dz]); N.append([-sx, 0, 0])
    cases.append(("2 spread pads, mu=1", epsilon_from_wrenches(
        wrench_set(P, N, [1.0] * 6, com, lam)), ">0"))

    # 3. Same geometry, frictionless. Two opposing normal directions only ->
    #    cannot span R^6 -> 0.
    cases.append(("same pads, mu=0", epsilon_from_wrenches(
        wrench_set(P, N, [0.0] * 6, com, lam)), "0"))

    # 4. Non-opposing contacts: all normals point the same way (what an
    #    opposition-deficient hand produces). Must be 0 at any friction.
    P = [[0, 0, -0.04], [0.03, 0, -0.04], [-0.03, 0, -0.04], [0, 0.03, -0.04]]
    N = [[0, 0, 1]] * 4
    cases.append(("4 co-directional contacts, mu=1", epsilon_from_wrenches(
        wrench_set(P, N, [1.0] * 4, com, lam)), "0"))

    # 5. A full enclosing cage: 6 face-centre contacts. Strongest case.
    P, N = [], []
    for ax in range(3):
        for s in (+1, -1):
            p = np.zeros(3); p[ax] = s * 0.04
            n = np.zeros(3); n[ax] = -s
            P.append(p); N.append(n)
    cases.append(("6 face contacts, mu=1", epsilon_from_wrenches(
        wrench_set(P, N, [1.0] * 6, com, lam)), ">0"))

    ok = True
    print(f"{'case':<34} {'epsilon':>10}   expect")
    for name, val, exp in cases:
        good = (val < 1e-6) if exp == "0" else (val > 1e-6)
        ok &= good
        print(f"{name:<34} {val:>10.5f}   {exp:<4} {'ok' if good else 'FAIL'}")
    # monotonicity: more friction cannot reduce epsilon
    P, N = [], []
    for sx in (+1, -1):
        for dy, dz in ((0, 0.03), (0.03, -0.02), (-0.03, -0.02)):
            P.append([sx * 0.04, dy, dz]); N.append([-sx, 0, 0])
    es = [epsilon_from_wrenches(wrench_set(P, N, [mu] * 6, com, lam))
          for mu in (0.2, 0.5, 1.0, 2.0)]
    mono = all(b >= a - 1e-9 for a, b in zip(es, es[1:]))
    ok &= mono
    print(f"{'epsilon monotone in mu':<34} {str([round(e,4) for e in es]):>10}"
          f"        {'ok' if mono else 'FAIL'}")
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
