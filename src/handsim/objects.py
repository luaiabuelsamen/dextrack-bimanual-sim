"""The object set benchmarks are run against.

Every bench so far invented its own object inline -- a 6.5 cm box here, a peg
there -- so no two results were comparable and "which objects" had no answer.
This is one parameterised set with a standard sweep, so a hand's score means
the same thing across experiments.

Sizes are chosen against the measured hand range: LEAP's closure gap spans
5.3-19.3 cm, f5d6's 3.4-7.8 cm, so the sweep has to straddle both or it cannot
separate them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass(frozen=True)
class Obj:
    key: str
    shape: str                  # box | cylinder | sphere | ellipsoid
    half: tuple                 # mujoco size semantics for the shape
    mass: float
    friction: tuple = (1.0, 0.02, 0.001)

    @property
    def grasp_width(self) -> float:
        """The dimension a thumb-versus-fingers pinch has to span."""
        if self.shape == "sphere":
            return 2 * self.half[0]
        if self.shape == "cylinder":
            return 2 * self.half[0]
        return 2 * min(self.half[0], self.half[1])

    def geom_kwargs(self, name="object_geom"):
        t = {"box": mujoco.mjtGeom.mjGEOM_BOX,
             "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
             "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
             "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[self.shape]
        return dict(name=name, type=t, size=list(self.half), mass=self.mass,
                    rgba=[0.85, 0.3, 0.2, 1], friction=list(self.friction))


def _box(w, h=0.05, m=0.05):
    return Obj(f"box{int(w*100)}", "box", (w / 2, h / 2, h / 2), m)


#: the standard width sweep, straddling every hand's closure range
WIDTH_SWEEP = tuple(_box(w) for w in
                    (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.11))

#: shape variation at one width, for "is it only boxes?"
SHAPE_SET = (
    Obj("box5", "box", (0.025, 0.025, 0.025), 0.05),
    Obj("cyl5", "cylinder", (0.025, 0.05), 0.05),
    Obj("sph5", "sphere", (0.025,), 0.05),
    Obj("ell5", "ellipsoid", (0.025, 0.02, 0.035), 0.05),
)

#: mass variation at one geometry, for "how heavy before it slips?"
MASS_SET = tuple(Obj(f"box5m{int(m*1000)}", "box", (0.025, 0.025, 0.025), m)
                 for m in (0.02, 0.05, 0.1, 0.2, 0.4))

SETS = {"width": WIDTH_SWEEP, "shape": SHAPE_SET, "mass": MASS_SET}


def describe(objs=WIDTH_SWEEP):
    return "\n".join(
        f"  {o.key:10} {o.shape:9} grasp width {o.grasp_width*100:5.1f} cm  "
        f"{o.mass*1000:5.0f} g" for o in objs)
