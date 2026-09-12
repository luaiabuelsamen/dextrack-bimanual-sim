"""Every external path the package needs, in one place.

These were hard-coded in five different modules, each with its own
`sys.path.insert` to reach them. Anything that must be found on this machine
rather than inside the package belongs here.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: vendored MuJoCo Menagerie (sparse clone: leap_hand, wonik_allegro, shadow_hand)
MENAGERIE = Path(os.environ.get("OPPDEF_MENAGERIE",
                                REPO / "vendor" / "mujoco_menagerie"))

#: Dexmate Vega upper body with two f5d6 hands
VEGA_URDF = Path(os.environ.get(
    "OPPDEF_VEGA_URDF",
    "/home/jetson3/projects/dexmate/dexmate-urdf/robots/humanoid/vega_1u/"
    "vega_1u_f5d6-obj.urdf"))

#: dextrack_vega, for its URDF->MJCF compiler (it strips the .glb visual meshes
#: MuJoCo cannot decode; the local compiler choked on them)
DEXTRACK = Path(os.environ.get("OPPDEF_DEXTRACK",
                               "/home/jetson3/projects/dextrack_vega"))

RESULTS = REPO / "results"
FIGURES = REPO / "figures"


def compile_urdf(urdf: Path) -> str:
    """URDF -> MJCF text, via dextrack_vega's glb-stripping compiler."""
    import sys
    if str(DEXTRACK) not in sys.path:
        sys.path.insert(0, str(DEXTRACK))
    from dextrack_vega.assets import _compile_to_mjcf
    return _compile_to_mjcf(urdf)


def menagerie_xml(rel: str) -> str:
    """Read a Menagerie model and absolutise its meshdir."""
    p = MENAGERIE / rel
    xml = p.read_text()
    assets = (p.parent / "assets").as_posix()
    return (xml.replace('meshdir="./assets/"', f'meshdir="{assets}/"')
               .replace('meshdir="assets"', f'meshdir="{assets}"'))
