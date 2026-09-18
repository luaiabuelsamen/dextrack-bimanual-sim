"""CPU MuJoCo vs MJX on identical controls, reported as a number.

The May effort could only claim "parity within primitive mode" because the
scene had been edited until MJX would run it. This one hands MJX the same
model, stripped only of provably visual-only geoms, and prints the divergence
it finds -- whatever that is, it is the transfer gap for anything trained in
MJX and evaluated on CPU.
"""
import argparse

import numpy as np

from handsim.envs.bimanual import build
from handsim.sim.port import strip_visual
from handsim.vec import parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--backend", default="mjx")
    a = ap.parse_args()

    _full, spec = build()
    strip_visual(spec)
    m = spec.compile()
    rng = np.random.default_rng(0)
    seq = np.clip(rng.normal(0, 0.05, (a.steps, m.nu)), -0.5, 0.5)

    r = parity(m, seq, backend=a.backend, n=a.n)
    print(f"backend {r['backend']}   n={r['n']}   steps={r['steps']}")
    print(f"  max |CPU - {r['backend'].upper()}| over the run : {r['max_abs']:.3e}")
    print(f"  at the final step                  : {r['final_abs']:.3e}")
    print("  per-step (every 5th):",
          " ".join(f"{e:.1e}" for e in r["per_step"][::5]))


if __name__ == "__main__":
    main()
