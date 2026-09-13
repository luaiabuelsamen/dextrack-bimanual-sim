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

#: Dexmate Vega upper body with two f5d6 hands. Only needed for the f5d6 rows of
#: the opposition axis -- every other hand comes from Menagerie. Not redistributed
#: here; set OPPDEF_VEGA_URDF to your own copy.
_DEV_VEGA = ("/home/jetson3/projects/dexmate/dexmate-urdf/robots/humanoid/"
             "vega_1u/vega_1u_f5d6-obj.urdf")   # the author's checkout; a default, not a requirement
VEGA_URDF = Path(os.environ.get("OPPDEF_VEGA_URDF") or _DEV_VEGA)

#: A URDF->MJCF compiler that strips the .glb visual meshes MuJoCo cannot decode.
#: Only needed alongside VEGA_URDF. Set OPPDEF_DEXTRACK to your own checkout.
_DEV_DEXTRACK = "/home/jetson3/projects/dextrack_vega"   # likewise
DEXTRACK = Path(os.environ.get("OPPDEF_DEXTRACK") or _DEV_DEXTRACK)


def require(path: Path, what: str, env: str) -> Path:
    """Fail with the fix rather than a stack trace three frames deep."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"  Set {env} to your copy, or run `make vendor` for the Menagerie models.")
    return path

RESULTS = REPO / "results"
FIGURES = REPO / "figures"


_URDF_LOCK = None


def compile_urdf(urdf: Path) -> str:
    """URDF -> MJCF text, via dextrack_vega's glb-stripping compiler.

    Serialised across processes. The upstream compiler writes fixed temporary
    filenames (`_dextrack_stripped.urdf`, `_dextrack_raw.xml`) NEXT TO the
    asset, so two processes compiling the same robot race: one deletes the
    other's temp file and the loser dies with FileNotFoundError. That killed a
    cell of the confirmatory sweep, and the review had flagged it before it
    did. A cross-process file lock beside the asset costs nothing and removes
    the race without touching the upstream compiler.
    """
    import sys, fcntl
    if str(DEXTRACK) not in sys.path:
        sys.path.insert(0, str(DEXTRACK))
    from dextrack_vega.assets import _compile_to_mjcf
    lock = Path(urdf).with_suffix(".oppdef.lock")
    try:
        fh = open(lock, "w")
    except OSError:
        return _compile_to_mjcf(urdf)
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return _compile_to_mjcf(urdf)
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def menagerie_xml(rel: str) -> str:
    """Read a Menagerie model and absolutise its meshdir.

    Models spell it several ways (`assets`, `./assets/`, `assets/`), so this
    rewrites whatever is there against the model's own directory rather than
    matching known spellings -- `trs_so_arm100` uses a form the first version
    missed and failed to open a single mesh.
    """
    import re
    p = require(MENAGERIE / rel, "MuJoCo Menagerie model", "OPPDEF_MENAGERIE")
    xml = p.read_text()

    def abso(match):
        raw = match.group(1)
        target = (p.parent / raw).resolve()
        return f'meshdir="{target.as_posix()}"'

    return re.sub(r'meshdir="([^"]+)"', abso, xml)
