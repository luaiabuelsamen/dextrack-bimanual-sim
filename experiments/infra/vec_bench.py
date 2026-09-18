"""How many steps a second, and does the batch still agree with one world?

Throughput alone is not a result. A backend that runs fast because its worlds
have silently fused, or because it drifted from single-world MuJoCo, is worse
than the slow loop it replaced -- so agreement is measured first and printed
beside the rate.
"""
import argparse

import numpy as np

from handsim.envs.bimanual import build
from handsim.sim.port import strip_visual, rollout_cpu
from handsim.vec import CpuVec, throughput


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--backend", default="cpu")
    a = ap.parse_args()

    _full, spec = build()
    strip_visual(spec)
    m = spec.compile()
    print(f"scene: nq {m.nq}  nv {m.nv}  nu {m.nu}  ngeom {m.ngeom}\n")

    c1 = np.random.default_rng(0).normal(0, 0.05, m.nu)
    v = CpuVec(m, 8)
    for _ in range(60):
        s = v.step(np.broadcast_to(c1, (8, m.nu)))
    ref = rollout_cpu(m, np.broadcast_to(c1, (60, m.nu)))
    spread = float(np.abs(s.qpos - s.qpos[0]).max())
    drift = float(np.abs(np.concatenate([s.qpos[0], s.qvel[0]]) - ref[-1]).max())
    print("AGREEMENT (8 worlds, identical controls, 60 steps)")
    print(f"  world-to-world            : {spread:.3e}")
    print(f"  vs single-world mj_step   : {drift:.3e}\n")

    print(f"THROUGHPUT ({a.backend})")
    base = None
    for n in a.sizes:
        r = throughput(m, n, backend=a.backend, steps=a.steps, warmup=5)
        base = base or r["steps_per_s"]
        print(f"  n={n:5d}: {r['steps_per_s']:9.0f} steps/s"
              f"   ({r['steps_per_s']/base:5.2f}x over n=1)")


if __name__ == "__main__":
    main()
