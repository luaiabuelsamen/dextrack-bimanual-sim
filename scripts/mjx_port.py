"""MJX port of the bimanual scene, done without degrading the physics.

The approach carried over from May was: delete the collision meshes, replace
fingertips with spheres, swap Newton for CG, convert cylinders to boxes, and add
armature until it stopped diverging. That produces a scene whose CONTACT PHYSICS
differs from the one the expert was validated in, so anything learned in it is
about a different world -- which is why that effort could only claim "parity
within primitive mode". That is not a port, it is a substitution.

The distinction it missed: this model's 252k mesh vertices are almost entirely
VISUAL geoms -- contype=0, conaffinity=0, density=0 in Menagerie's `visual`
class. They generate no contacts and carry no mass, so deleting them is provably
a no-op on the dynamics. The COLLISION meshes are the eight fingertips at 52
vertices each, which MJX handles natively.

So this module does exactly two things:
  1. strips visual-only geoms and any mesh assets left unreferenced,
  2. proves the strip is physics-neutral by replaying an identical control
     sequence through full and stripped models on CPU and comparing states.

Only then does it hand the model to MJX, and the MJX-vs-MuJoCo difference is
reported as a measured number rather than tuned away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def is_visual(g):
    """Visual-only: collides with nothing and contributes no mass.

    Both conditions matter. A geom with contype=0 still contributes inertia
    unless its density (or explicit mass) is zero, so dropping on contype alone
    would change the dynamics -- exactly the kind of silent physics edit this
    module exists to avoid.
    """
    massless = float(g.density) == 0.0 or float(g.mass) == 0.0
    return int(g.contype) == 0 and int(g.conaffinity) == 0 and massless


def strip_visual(spec):
    """Delete visual-only geoms and the mesh assets that become unreferenced."""
    removed = []
    for g in list(spec.geoms):
        if is_visual(g):
            removed.append(g.name)
            spec.delete(g)
    used = set()
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname:
            used.add(g.meshname)
    dropped = []
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            dropped.append(mesh.name)
            spec.delete(mesh)
    return removed, dropped


def rollout_cpu(model, ctrl_seq, seed_qpos=None):
    """Deterministic replay; returns the full state trace."""
    d = mujoco.MjData(model)
    if seed_qpos is not None:
        d.qpos[:] = seed_qpos
    mujoco.mj_forward(model, d)
    out = []
    for c in ctrl_seq:
        d.ctrl[:] = c
        mujoco.mj_step(model, d)
        out.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
    return np.array(out)


if __name__ == "__main__":
    from bimanual_env import build

    full, spec_full = build()
    stripped, spec_strip = build()
    removed, dropped = strip_visual(spec_strip)
    stripped = spec_strip.compile()

    print(f"full     : nbody {full.nbody:3} ngeom {full.ngeom:3} nmesh {full.nmesh:3} "
          f"meshvert {full.nmeshvert:7} nq {full.nq} nv {full.nv}")
    print(f"stripped : nbody {stripped.nbody:3} ngeom {stripped.ngeom:3} "
          f"nmesh {stripped.nmesh:3} meshvert {stripped.nmeshvert:7} "
          f"nq {stripped.nq} nv {stripped.nv}")
    print(f"removed {len(removed)} visual geoms, {len(dropped)} unreferenced meshes")

    assert full.nq == stripped.nq and full.nv == stripped.nv, \
        "strip changed the state dimension -- it is not physics-neutral"

    # identical pseudo-random control sequence through both models
    rng = np.random.default_rng(0)
    n = 400
    ctrl = np.clip(rng.normal(0, 0.25, (n, full.nu)), -0.5, 0.5)
    for i in range(full.nu):
        lo, hi = full.actuator_ctrlrange[i]
        if full.actuator_ctrllimited[i]:
            ctrl[:, i] = np.clip(ctrl[:, i], lo, hi)

    a = rollout_cpu(full, ctrl)
    b = rollout_cpu(stripped, ctrl)
    err = np.abs(a - b)
    print(f"\nPHYSICS-NEUTRALITY CHECK  ({n} steps, identical controls)")
    print(f"  max |state difference| : {err.max():.3e}")
    print(f"  mean |state difference|: {err.mean():.3e}")
    print("  ->", "IDENTICAL, the strip is a no-op on the dynamics"
          if err.max() < 1e-9 else
          f"NOT identical: the strip changed the physics (max {err.max():.2e})")


def to_mjx(model):
    """Hand the stripped model to MJX. No solver or geometry substitutions."""
    import mujoco.mjx as mjx
    return mjx.put_model(model)


def parity(model, mjx_model, ctrl_seq, n_report=6):
    """CPU MuJoCo vs MJX on the same controls. The difference is MEASURED, not
    engineered away: whatever it is, it is the transfer gap for anything trained
    in MJX and evaluated on CPU."""
    import jax, jax.numpy as jp
    import mujoco.mjx as mjx

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    dx = mjx.put_data(model, d)
    step = jax.jit(mjx.step)

    cpu, gpu = [], []
    for c in ctrl_seq:
        d.ctrl[:] = c
        mujoco.mj_step(model, d)
        cpu.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
        dx = dx.replace(ctrl=jp.asarray(c))
        dx = step(mjx_model, dx)
        gpu.append(np.concatenate([np.asarray(dx.qpos), np.asarray(dx.qvel)]))
    cpu, gpu = np.array(cpu), np.array(gpu)
    err = np.abs(cpu - gpu)
    return cpu, gpu, err
