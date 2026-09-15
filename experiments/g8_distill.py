"""Stage 5: solve a set of references, then distil them into one policy.

Held out by OBJECT. Frames inside a clip are near-duplicates and clips of the
same object share its geometry, so a random-frame split measures interpolation
within a trajectory and would report a number that means nothing. This
repository has already shipped one behaviour-cloning result that open-loop
replay matched exactly; the split is the first line of defence against the next.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oppdef.human import distill


def main(a):
    inv = json.loads(Path("results/grab_inventory.json").read_text())["rows"]
    rows = [r for r in inv if "error" not in r and r["rhand_hold_len"] >= 25]
    # spread over objects rather than taking the alphabetical head, which would
    # be four airplanes and four alarmclocks
    seen, picked = {}, []
    for r in sorted(rows, key=lambda r: (-r["rhand_hold_len"], r["seq"])):
        c = seen.get(r["object"], 0)
        if c < a.per_object:
            picked.append(r)
            seen[r["object"]] = c + 1
        if len(picked) >= a.n:
            break
    print(f"{len(picked)} references over {len(seen)} objects", flush=True)

    eps = distill.collect(picked, hand=a.hand, out=a.out,
                          levels=(0.5, 1.0), horizon=3, samples=a.samples)
    if not eps:
        print("no episodes solved")
        return
    print(f"\n{len(eps)} episodes, "
          f"{sum(len(e.obs) for e in eps)} transitions, "
          f"{len({e.obj for e in eps})} objects")
    model, (O, A, is_test) = distill.train(a.out, holdout_frac=a.holdout,
                                           epochs=a.epochs)
    from oppdef.learning.bc import policy_fn
    act = policy_fn(model)
    for name, sel in (("train", ~is_test), ("HELD-OUT OBJECTS", is_test)):
        if sel.sum() == 0:
            continue
        pred = np.stack([act(o) for o in O[sel]])
        err = np.abs(pred - A[sel]).mean()
        base = np.abs(A[sel] - A[~is_test].mean(0)).mean()
        print(f"  {name:18s} n={int(sel.sum()):5d}  MAE {err:.5f}  "
              f"(predict-the-mean {base:.5f}, ratio {err/max(base,1e-9):.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--per-object", type=int, default=2)
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out", default="results/distill_episodes.npz")
    main(ap.parse_args())
