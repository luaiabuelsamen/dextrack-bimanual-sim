"""Where hand-object demonstrations come from.

The project's thesis needs human grasp data, and the registration for it has
been pending since day one. That is latency, not code -- but the code that
consumes the data can exist now, so the day it lands nothing has to be
designed under time pressure.

A source yields `GraspRef`s: a hand pose, an object pose, and optionally contact
state, in a common frame. `SyntheticSource` implements that contract today with
analytically-placed human-like grasps, which is what the M1.5 probe used. The
real sources are declared with the format they will produce and fail with
instructions rather than silently returning nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass
class GraspRef:
    """One demonstrated hand-object interaction, source-agnostic.

    Poses are in a common frame whose origin is the object's centre, so a
    retargeter never has to know where the data came from.
    """
    wrist: np.ndarray             # (3,) wrist position
    wrist_quat: np.ndarray        # (4,) wrist orientation, wxyz
    fingertips: np.ndarray        # (n, 3) fingertips: FINGERS index->little,
                                  #        then the THUMB LAST
    obj_pos: np.ndarray           # (3,)
    obj_quat: np.ndarray          # (4,)
    obj_half: np.ndarray = field(   # (3,) box half-extents
        default_factory=lambda: np.full(3, 0.025))
    contacts: np.ndarray | None = None    # (n,) bool, per fingertip
    label: str = ""
    source: str = ""

    def vectors(self) -> np.ndarray:
        """Wrist->fingertip vectors: what keypoint retargeting actually matches.

        Ordered fingers-then-thumb, matching `embodiment.Hand.tip_names`. The
        convention is load-bearing: stacked thumb-first, the frame fit pairs the
        human's thumb with the robot's index finger and drops the robot's thumb,
        which is silent -- it returns a plausible number for a correspondence
        that is wrong.
        """
        return self.fingertips - self.wrist[None, :]


class GraspSource:
    """Interface. Anything that can produce GraspRefs plugs in here."""
    name = "abstract"

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[GraspRef]:
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        """(is it usable here, why not)."""
        return False, "abstract source"


class SyntheticSource(GraspSource):
    """Analytically placed human-like grasps on a box.

    Thumb pad on one face, four fingertips spread down the opposite face,
    opposition across the narrow dimension -- what a person does with a box this
    size. Explicitly NOT MANO: it is a stand-in with the same interface, so the
    retargeting code is exercised before the corpus arrives, and every result
    from it must say so.
    """
    name = "synthetic"

    def __init__(self, widths=(0.04, 0.05, 0.06, 0.07), height=None, n_per=3,
                 seed=0, cube=True):
        self.widths, self.n_per, self.cube = widths, n_per, bool(cube)
        self.height = height
        self.rng = np.random.default_rng(seed)

    def available(self):
        return True, "always -- it is generated, not loaded"

    def __len__(self):
        return len(self.widths) * self.n_per

    def __iter__(self):
        for w in self.widths:
            for k in range(self.n_per):
                jitter = self.rng.normal(0, 0.004, 3)
                hx = w / 2
                # The fingers are spread over the box's ACTUAL height. They
                # used to sit at fixed z = +/-0.035 and +/-0.012 regardless of
                # the object, so on the cube the comparison actually tests, two
                # of the five "contacts" were 12.56 and 7.44 mm OUTSIDE the
                # box -- a reference grasp that does not touch what it claims
                # to hold.
                hz = hx if (self.cube or self.height is None) \
                    else min(self.height / 2, hx)
                th = np.array([+hx, 0.0, 0.4 * hz]) + jitter
                fingers = np.stack([
                    np.array([-hx, 0.0, z * hz]) + jitter
                    for z in (0.85, 0.28, -0.28, -0.85)])
                yield GraspRef(
                    wrist=np.array([0.0, -0.10, 0.02]),
                    wrist_quat=np.array([1.0, 0, 0, 0]),
                    fingertips=np.vstack([fingers, th[None, :]]),
                    obj_pos=np.zeros(3), obj_quat=np.array([1.0, 0, 0, 0]),
                    obj_half=np.array([hx, hz, hz]),
                    contacts=np.ones(5, bool),
                    label=f"box{int(w*1000)}mm_{k}", source=self.name)


class DexonomySource(GraspSource):
    """Dexonomy (RSS 2025): 9.5M grasps, pre-grasp / grasp / SQUEEZE triples.

    The triple is the part that matters -- every closure written by hand in this
    project failed until it was structured that way. Released hands include
    Shadow; the HF release ships `Dexonomy_GRASP_shadow.tar.gz` (2.9 GB) and does
    NOT currently include LEAP.
    """
    name = "dexonomy"
    URL = "https://huggingface.co/datasets/JiayiChenPKU/Dexonomy"

    def __init__(self, root=None):
        from pathlib import Path
        self.root = Path(root) if root else None

    def available(self):
        if self.root is None or not self.root.exists():
            return False, (f"not downloaded. See {self.URL} "
                           f"(Dexonomy_GRASP_shadow.tar.gz, ~2.9 GB), then pass "
                           f"root=<extracted dir>")
        return True, "ok"

    def __len__(self):
        ok, why = self.available()
        if not ok:
            raise FileNotFoundError(why)
        raise NotImplementedError("reader lands with the data")

    def __iter__(self):
        raise FileNotFoundError(self.available()[1])


class ArcticSource(GraspSource):
    """ARCTIC: bimanual, articulated objects, ground-truth MANO + object pose.

    The right corpus for this project: it is bimanual, its objects articulate,
    and it is the benchmark StableHand is evaluated on. Requires registration --
    1-2 weeks -- which is why it is declared here before it exists.
    """
    name = "arctic"
    URL = "https://arctic.is.tue.mpg.de/"

    def __init__(self, root=None):
        from pathlib import Path
        self.root = Path(root) if root else None

    def available(self):
        if self.root is None or not self.root.exists():
            return False, f"not downloaded; register at {self.URL}"
        return True, "ok"

    def __iter__(self):
        raise FileNotFoundError(self.available()[1])


SOURCES = {"synthetic": SyntheticSource, "dexonomy": DexonomySource,
           "arctic": ArcticSource}


def status():
    rows = []
    for k, cls in SOURCES.items():
        src = cls()
        ok, why = src.available()
        rows.append((k, ok, len(src) if ok else 0, why))
    return rows
