"""Sensing: what a policy is allowed to know, declared rather than assumed.

The first observation in this project was 52 numbers assembled inline in the BC
script. Eleven of them were ground-truth object state -- including `peg_out`,
which IS the success metric -- there were no velocities, and there was no contact
sensing at all, in a project whose subject is contact. None of that was visible
without reading the function.

So an observation here is a declared `ObsSpec`, and every channel says whether it
is ONBOARD (a real robot could measure it) or PRIVILEGED (only a simulator knows
it). `ObsSpec.onboard_only()` drops the privileged half, which makes the
sim-to-real question an ablation you run rather than a caveat you write.

Channels
--------
proprio_pos   joint positions of the actuated chain            onboard
proprio_vel   joint velocities                                 onboard
tactile       per-fingertip normal force, from touch sensors   onboard
object_pose   object position + quaternion                     PRIVILEGED
object_vel    object linear + angular velocity                 PRIVILEGED
task          scalar task readouts (e.g. peg extraction)       PRIVILEGED
phase         normalised episode time (+ sin/cos)              onboard-ish*

*phase is onboard in the trivial sense that a controller knows its own clock,
but leaning on it means cloning a time-indexed demonstrator rather than a
state-driven one; `ObsSpec` records it separately so that dependence stays
visible. See NOTES 2026-09-12.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import mujoco

ONBOARD = ("proprio_pos", "proprio_vel", "tactile", "phase")
PRIVILEGED = ("object_pose", "object_vel", "task")


@dataclass(frozen=True)
class ObsSpec:
    """Which channels a policy gets. Order is fixed by `CHANNELS`."""
    proprio_pos: bool = True
    proprio_vel: bool = True
    tactile: bool = True
    object_pose: bool = False
    object_vel: bool = False
    task: bool = False
    phase: bool = False

    CHANNELS = ("proprio_pos", "proprio_vel", "tactile",
                "object_pose", "object_vel", "task", "phase")

    def enabled(self):
        return [c for c in self.CHANNELS if getattr(self, c)]

    def onboard_only(self) -> "ObsSpec":
        """The same spec with every privileged channel removed."""
        return replace(self, **{c: False for c in PRIVILEGED})

    def uses_privileged(self) -> bool:
        return any(getattr(self, c) for c in PRIVILEGED)

    def describe(self, sizes: dict) -> str:
        parts = [f"{c}={sizes.get(c, 0)}"
                 f"{'*' if c in PRIVILEGED else ''}" for c in self.enabled()]
        return f"{sum(sizes.get(c, 0) for c in self.enabled())}d  [" + \
               " ".join(parts) + "]   (* = privileged)"


# --------------------------------------------------------------------------
# model surgery: add the sensors the scene never had
# --------------------------------------------------------------------------
def add_touch_sensors(spec, geom_names, prefix="touch_"):
    """Attach a MuJoCo `touch` sensor to each named geom's body.

    A touch sensor reports the total normal force on the sites within its
    zone -- the closest thing to a fingertip pressure pad, and the channel this
    project most conspicuously lacked. Returns the sensor names created.
    """
    made = []
    for g in geom_names:
        geom = None
        for cand in spec.geoms:
            if cand.name == g:
                geom = cand
                break
        if geom is None:
            continue
        body = geom.parent
        site_name = f"{prefix}site_{g}"
        site = body.add_site(name=site_name, pos=geom.pos,
                             size=[max(float(np.max(geom.size)) * 1.35, 0.008)] * 3,
                             type=mujoco.mjtGeom.mjGEOM_SPHERE, group=4)
        spec.add_sensor(name=f"{prefix}{g}",
                        type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
        made.append(f"{prefix}{g}")
    return made


def fingertip_geoms(model, tip_bodies):
    """Collision geoms belonging to the given fingertip bodies."""
    out = []
    for b in tip_bodies:
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == b and (model.geom_contype[g]
                                              or model.geom_conaffinity[g]):
                n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                if n:
                    out.append(n)
    return out


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
class Observer:
    """Builds observation vectors for one model against one ObsSpec."""

    def __init__(self, model, data, spec: ObsSpec, joint_qadr, joint_dofadr,
                 touch_sensors=(), object_qadr=None, object_dofadr=None,
                 task_fn=None):
        self.m, self.d, self.spec = model, data, spec
        self.jq = np.asarray(joint_qadr, int)
        self.jv = np.asarray(joint_dofadr, int)
        self.oq, self.ov = object_qadr, object_dofadr
        self.task_fn = task_fn
        self.touch_adr = []
        for name in touch_sensors:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                self.touch_adr.append(int(model.sensor_adr[sid]))
        self.sizes = {
            "proprio_pos": len(self.jq) if spec.proprio_pos else 0,
            "proprio_vel": len(self.jv) if spec.proprio_vel else 0,
            "tactile": len(self.touch_adr) if spec.tactile else 0,
            "object_pose": 7 if (spec.object_pose and object_qadr is not None) else 0,
            "object_vel": 6 if (spec.object_vel and object_dofadr is not None) else 0,
            "task": (len(np.atleast_1d(task_fn())) if (spec.task and task_fn) else 0),
            "phase": 3 if spec.phase else 0,
        }
        self.dim = sum(self.sizes.values())

    def __call__(self, phase=0.0):
        d, s = self.d, self.spec
        parts = []
        if s.proprio_pos:
            parts.append(d.qpos[self.jq])
        if s.proprio_vel:
            parts.append(d.qvel[self.jv])
        if s.tactile and self.touch_adr:
            parts.append(np.array([d.sensordata[a] for a in self.touch_adr]))
        if s.object_pose and self.oq is not None:
            parts.append(d.qpos[self.oq:self.oq + 7])
        if s.object_vel and self.ov is not None:
            parts.append(d.qvel[self.ov:self.ov + 6])
        if s.task and self.task_fn:
            parts.append(np.atleast_1d(self.task_fn()))
        if s.phase:
            parts.append([phase, np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        return np.concatenate([np.asarray(p, float).ravel()
                               for p in parts]).astype(np.float32)


#: what the BC baseline used, for reference and regression
LEGACY = ObsSpec(proprio_pos=True, proprio_vel=False, tactile=False,
                 object_pose=True, object_vel=False, task=True, phase=True)

#: everything a simulator can give
FULL = ObsSpec(proprio_pos=True, proprio_vel=True, tactile=True,
               object_pose=True, object_vel=True, task=True, phase=True)

#: what a real robot could actually measure
ONBOARD_ONLY = FULL.onboard_only()
