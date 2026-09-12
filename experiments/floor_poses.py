"""Entry point: render each hand at its measured opposition-floor pose.

One hand per process -- creating a second EGL Renderer after the first is
closed raises EGL_NOT_INITIALIZED on this machine.
"""
import sys
from oppdef.viz.floor_poses import render, CFG

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "f5d6"
    render(key, f"figures/floor_{key}.png", **CFG[key])
