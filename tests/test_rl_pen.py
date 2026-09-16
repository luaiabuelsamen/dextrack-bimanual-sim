"""The penetration cost and the hold proxy in the PPO reward (`oppdef.human.rl`).

Needs GRAB; skipped without it. Builds the smallest tracker the pipeline has
-- an 8-frame window of s1/cubesmall_lift at stride 8 -- once per module
(~20 s), with FOUR MuJoCo substeps per control step instead of 33: the
retarget's own overlap (20.9 mm at reset, the stage 2 defect) then decays
18.8 -> 14.6 -> 10.4 mm over the first steps instead of the cube being flung
clear within one control frame, so the buried state the cost must see is
present, and one substep is too few -- `d.contact` after a single `mj_step`
is the collision pass at the PRE-integration pose, identical in every env.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import mujoco

from oppdef import paths
from oppdef.human import grab, track, rl

CLIP = "s1/cubesmall_lift.npz"
BASE = "3216767"        # the commit before the cost existed
REPO = Path(__file__).resolve().parents[1]
N_ENVS = 2

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (paths.GRAB / "grab" / CLIP).exists(),
                       reason="GRAB data not present"),
]


@pytest.fixture(scope="module")
def rt():
    if str(REPO) not in sys.path:          # track.py itself imports experiments.*
        sys.path.insert(0, str(REPO))
    from experiments.tracking.grab_inventory import contact_mask, longest_run, _tree
    seq = grab.load(CLIP, verts=True, stride=8)
    w = longest_run(contact_mask(seq, _tree(seq, {}), "rhand"))
    # The REAL control rate is ~33 substeps per control step (one reference
    # frame, see ReferenceTracker.ctrl_every). 4 is a test convenience that
    # keeps the buried state in view; nothing here is evidence about what the
    # policy or the object does at the real rate.
    return track.ReferenceTracker(seq, window=(w[0], 8), ctrl_every=4)


def load_base_rl():
    """`rl.py` exactly as committed at BASE, as its own module."""
    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE}:src/oppdef/human/rl.py"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:      # noqa: PERF203
        pytest.skip(f"cannot read rl.py at {BASE}: {e}")
    spec = importlib.util.spec_from_loader(f"rl_{BASE}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod       # dataclasses resolves string annotations here
    exec(compile(src, f"rl.py@{BASE}", "exec"), mod.__dict__)
    return mod


def run(pool_cls, rt, cfg, n_steps=4, seed=0, random_actions=True, prep=None,
        measure=False):
    """A seeded step sequence; returns rewards, dones, obs and per-step stats.

    `measure` asks the branch's step to read the contact set (the base
    commit's step has no such argument, so it is passed only when set)."""
    rng = np.random.default_rng(seed)
    pool = pool_cls(rt, N_ENVS, [0])
    try:
        pool.reset_all(rng)
        R, D, O, S = [], [], [], []
        for t in range(n_steps):
            if prep is not None:
                prep(pool, t)
            a = (rng.normal(size=(N_ENVS, rt.n_action)) if random_actions
                 else np.zeros((N_ENVS, rt.n_action)))
            o, r, d = pool.step(a, cfg, **({"measure": True} if measure else {}))
            R.append(r.copy()), D.append(d.copy()), O.append(o.copy())
            S.append({k: getattr(pool, k).copy() for k in
                      ("pen", "ncon", "nbody", "grip", "r_pen", "r_hold")}
                     if hasattr(pool, "pen") else None)
        return np.array(R), np.array(D), np.array(O), S
    finally:
        pool.restore()
        pool.close()


def stat(S, key):
    return np.array([s[key] for s in S])


# -- (a) defaults leave the reward bit-for-bit as it was --------------------
def test_defaults_match_base_commit_bit_for_bit(rt):
    base = load_base_rl()
    cfg = rl.RLConfig()
    assert cfg.w_pen == 0 and cfg.w_force == 0 and cfg.w_hold == 0
    R_a, D_a, O_a, _ = run(rl.Pool, rt, cfg)
    R_b, D_b, O_b, _ = run(rl.Pool, rt, cfg)
    assert np.array_equal(R_a, R_b), "the step sequence itself is not deterministic"
    R0, D0, O0, _ = run(base.Pool, rt, base.RLConfig())
    assert np.array_equal(R0, R_a)                # bit-for-bit, no tolerance
    assert np.array_equal(D0, D_a)
    assert np.array_equal(O0, O_a)
    # measuring the contact set for the log changes nothing either
    R_m, D_m, O_m, S_m = run(rl.Pool, rt, cfg, measure=True)
    assert np.array_equal(R0, R_m) and np.array_equal(O0, O_m)
    assert np.all(np.isfinite(stat(S_m, "pen")))


# -- (b) the log reads each environment's own MjData -------------------------
def test_stats_are_finite_and_read_from_the_right_data(rt):
    rng = np.random.default_rng(1)
    pool = rl.Pool(rt, N_ENVS, [0])
    try:
        pool.reset_all(rng)
        a = np.zeros((N_ENVS, rt.n_action))
        a[1, :3] = 1.0                        # env 1's palm moves, env 0's does not
        pool.step(a, rl.RLConfig(), measure=True)
        assert pool.measured
        assert np.all(np.isfinite(pool.pen)) and np.all(pool.pen >= 0)
        assert np.all(pool.ncon >= 0) and np.all(pool.grip >= 0)
        assert np.all(np.isfinite(pool.pe)) and np.all(pool.pe >= 0)
        assert pool.pen[0] != pool.pen[1], "both envs read the same data"
        for i in range(N_ENVS):
            pool._use(i)
            assert rt.sim.data is pool.datas[i]
            cs = rl.contact_summary(rt.sim)
            assert (cs.pen, cs.n) == (pool.pen[i], pool.ncon[i])
            assert (cs.pen, cs.n) == rt.sim.penetration()       # the scene's own loop
            grip_n, ncon = track.total_grip(rt.sim)
            assert cs.grip == pytest.approx(grip_n) and cs.n == ncon
            assert cs.bodies == track.wrap_score(rt.sim)[1] == pool.nbody[i]
            assert cs == rl.contact_summary(rt.sim, masks=pool._masks)
            assert pool.pe[i] == rt.error(0)[0]
        # at defaults, and not asked for, the contact set is not read at all:
        # nan across the board, position error still on
        pool.step(a, rl.RLConfig())
        assert not pool.measured
        for v in (pool.pen, pool.ncon, pool.nbody, pool.grip):
            assert np.all(np.isnan(v))
        assert np.all(np.isfinite(pool.pe))
        # a term that prices penetration reads depth, count and bodies but
        # not the force; one that prices the force reads everything
        pool.step(a, rl.RLConfig(w_pen=0.1))
        assert pool.measured and np.all(np.isfinite(pool.pen)) and np.all(np.isnan(pool.grip))
        pool.step(a, rl.RLConfig(w_hold=0.1))
        assert np.all(np.isfinite(pool.grip))
    finally:
        pool.restore()
        pool.close()


# -- (c) penetration costs, and is bounded ----------------------------------
def bury(pool, t):
    """Env 1: object centre onto the centroid of the hand's geoms -- measured
    22 mm deep on 17 contacts at ~2 kN. (Not the mocap body: that is the hand
    BASE, 30 cm from the palm, and puts the cube in empty space.)"""
    if t == 0:
        rt = pool.rt
        pool._use(1)
        d = rt.sim.data
        d.qpos[rt.obj_q:rt.obj_q + 3] = d.geom_xpos[rt.sim.hand_gids].mean(0)
        mujoco.mj_forward(rt.sim.model, d)


def test_penetration_lowers_the_reward_by_the_hinge(rt):
    cfg0, cfg1 = rl.RLConfig(), rl.RLConfig(w_pen=0.5)
    R0, _, _, S0 = run(rl.Pool, rt, cfg0, n_steps=2, random_actions=False, prep=bury,
                       measure=True)
    R1, _, _, S1 = run(rl.Pool, rt, cfg1, n_steps=2, random_actions=False, prep=bury)
    pen = stat(S1, "pen")
    assert np.array_equal(pen, stat(S0, "pen")), "the cost changed the physics"
    assert pen[0, 1] > 5 * cfg1.pen_allow, "the buried env is not buried"
    deep = pen > cfg1.pen_allow
    assert deep.any()
    assert np.all(R1[deep] < R0[deep])
    assert np.all(R1[~deep] == R0[~deep])
    hinge = 0.5 * np.maximum(0.0, pen - cfg1.pen_allow) / cfg1.pen_allow
    assert np.allclose(R0 - R1, np.minimum(hinge, cfg1.pen_max), atol=1e-12)
    assert np.allclose(stat(S1, "r_pen"), np.minimum(hinge, cfg1.pen_max))
    # the bound: no weight can make a step cost more than pen_max
    R9, _, _, S9 = run(rl.Pool, rt, rl.RLConfig(w_pen=1e6), n_steps=2,
                       random_actions=False, prep=bury)
    assert np.all(R0 - R9 <= cfg1.pen_max + 1e-12)
    assert np.all(stat(S9, "r_pen")[deep] == cfg1.pen_max)


# -- (d) the hold proxy: two bodies, one object weight, bounded --------------
def far(pool, t):
    """Env 1: object a metre above the hand -- no contact, no penetration."""
    if t == 0:
        rt = pool.rt
        pool._use(1)
        d = rt.sim.data
        d.qpos[rt.obj_q + 2] += 1.0
        mujoco.mj_forward(rt.sim.model, d)


def test_hold_bonus_pays_for_a_multi_body_contact_set_only(rt):
    cfg0, cfgh = rl.RLConfig(), rl.RLConfig(w_hold=0.7)
    R0, _, _, S0 = run(rl.Pool, rt, cfg0, n_steps=2, random_actions=False, prep=far,
                       measure=True)
    Rh, _, _, Sh = run(rl.Pool, rt, cfgh, n_steps=2, random_actions=False, prep=far)
    nbody, grip = stat(Sh, "nbody"), stat(Sh, "grip")
    assert nbody[0, 0] >= 2 and grip[0, 0] > 0, "env 0 is not in a multi-body contact"
    assert nbody[0, 1] == 0 and grip[0, 1] == 0 and stat(Sh, "pen")[0, 1] == 0
    multi = nbody >= 2
    assert np.all(Rh[multi] > R0[multi])
    assert np.all(Rh[~multi] == R0[~multi])           # letting go earns nothing
    weight = rl.object_weight(rt)
    assert weight == pytest.approx(0.2 * 9.81)
    expect = np.where(multi, 0.7 * np.minimum(1.0, grip / weight), 0.0)
    assert np.allclose(Rh - R0, expect, atol=1e-12)
    assert np.all(Rh - R0 <= 0.7 + 1e-12)
    # and the penetration cost is zero on the let-go env, whatever the weight
    Rp, _, _, Sp = run(rl.Pool, rt, rl.RLConfig(w_pen=1e6), n_steps=2,
                       random_actions=False, prep=far)
    assert stat(Sp, "r_pen")[0, 1] == 0 and Rp[0, 1] == R0[0, 1]


# -- evaluate() reports where the rollout ended -----------------------------
def test_evaluate_reports_end_state(rt, capsys):
    torch = pytest.importorskip("torch")

    class Still:
        def dist(self, x):
            mean = torch.zeros(x.shape[0], rt.n_action)
            return type("D", (), {"mean": mean, "sample": lambda self: mean})()

    errs, acts = rl.evaluate(rt, Still(), start=0)          # unchanged shape
    assert errs.shape == (rt.T,) and acts.shape == (rt.T, rt.n_action)
    errs, acts, es = rl.evaluate(rt, Still(), start=0, end=True, verbose=True)
    assert isinstance(es, rl.EndState)
    assert np.isfinite(es.pen_mm) and es.pen_mm >= 0
    assert es.contacts >= 0 and es.bodies >= 0 and es.grip_n >= 0
    assert "contacts" in capsys.readouterr().out


# -- train() logs burying and held-ness every iteration ---------------------
def test_train_logs_penetration_and_heldness(rt, capsys):
    pytest.importorskip("torch")
    cfg = rl.RLConfig(n_envs=N_ENVS, horizon=4, iters=3, epochs=1, minibatch=8)
    _net, log = rl.train(rt, cfg, starts=[0], verbose=True)
    for k in ("err_mm", "r_pen", "r_hold"):
        v = getattr(log, k)
        assert len(v) == 3 and np.all(np.isfinite(v)) and np.all(np.asarray(v) >= 0)
    assert np.all(np.asarray(log.r_pen) == 0) and np.all(np.asarray(log.r_hold) == 0)
    # at defaults the contact set is SAMPLED: logged iterations only (0 and
    # the last with log_every=10), nan in between
    for k in ("pen_mm", "frac_held", "grip_n"):
        v = getattr(log, k)
        assert len(v) == 3
        assert np.isfinite(v[0]) and np.isfinite(v[2]) and v[0] >= 0
        assert np.isnan(v[1])
    out = capsys.readouterr().out
    assert "pen" in out and "held" in out and "grip" in out and "err" in out
    assert out.count("iter") == 2
    # a term that prices the force turns the measurement on everywhere
    _net, log = rl.train(rt, rl.RLConfig(n_envs=N_ENVS, horizon=4, iters=3, epochs=1,
                                         minibatch=8, w_hold=0.5), starts=[0], verbose=False)
    for k in ("pen_mm", "frac_held", "grip_n"):
        assert np.all(np.isfinite(getattr(log, k)))
