"""Entry point: render each hand at its measured opposition-floor pose.

One hand per process -- creating a second EGL Renderer after the first is
closed raises EGL_NOT_INITIALIZED on this machine.
"""
import sys
from experiments.retracted.viz_floor_poses import render, CFG  # moved out
# of src/: it drew poses from the RETRACTED axis and imported `tip_ids`
# and `joint_set`, removed from oppdef.hands.axis in 8cc4b63 when that
# module stopped keeping its own hand table. It had been unimportable
# for three days -- the only import failure among 88 live modules -- and
# nothing under src/ or experiments/ referenced it but this retracted
# script, which is where the README says this lineage belongs.

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "f5d6"
    render(key, f"figures/floor_{key}.png", **CFG[key])
