"""G4 Part A analysis, exactly as pre-registered in docs/G4_PREREGISTRATION.md.

Primary: unanimity rate of the R perturbed repeats per (grasp, task), with a
Wilson interval and a grasp-clustered bootstrap, since each grasp contributes
two pairs. Failed repeats are counted in the denominator, not dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

THRESHOLD = 0.80


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        blob = json.loads(pathlib.Path(f).read_text())
        rows += blob["rows"] if isinstance(blob, dict) else blob
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def strict_unanimous(r):
    """All DECLARED repeats agree. A repeat whose grasp failed to re-form is a
    disagreement with the run that produced it, not a missing datum -- dropping
    those silently is how the raw field reads 456/456 where the honest count is
    438/456."""
    return bool(r["unanimous"]) and r["n_valid_repeats"] == len(r["repeats"])


def cluster_ci(rows, fn, n_boot=3000, seed=0):
    """Bootstrap the rate, resampling GRASPS (each contributes two pairs)."""
    by_grasp = {}
    for r in rows:
        by_grasp.setdefault((r["hand"], r["shape"], r["grasp"]), []).append(r)
    groups = list(by_grasp.values())
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sel = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        flat = [r for g in sel for r in g]
        out.append(np.mean([bool(fn(r)) for r in flat]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/g4_*.json")
    a = ap.parse_args()
    rows = load(a.glob)
    if not rows:
        raise SystemExit("no rows")
    n = len(rows)
    grasps = len({(r["hand"], r["shape"], r["grasp"]) for r in rows})

    # every declared repeat must count: a repeat whose grasp failed to re-form
    # is a disagreement with the run that produced it, not a missing datum
    full = [r for r in rows if r["n_valid_repeats"] == len(r["repeats"])]
    unan = np.array([strict_unanimous(r) for r in rows])
    det = np.array([bool(r["determinism_ok"]) for r in rows])

    print(f"G4 Part A -- pre-registered analysis")
    print(f"{n} (grasp, task) pairs from {grasps} grasps; "
          f"{len(full)} had all repeats re-form\n")

    k = int(det.sum())
    lo, hi = wilson(k, n)
    print(f"DETERMINISM CHECK (unperturbed repeat vs the saved G3 outcome)")
    print(f"  agree {k}/{n} = {k/n:.3f}   Wilson 95% [{lo:.3f}, {hi:.3f}]")
    if k / n < 0.99:
        print("  ** the pipeline is NOT deterministic; the pre-registration")
        print("     says fix this before interpreting anything else **")

    k = int(unan.sum())
    lo, hi = wilson(k, n)
    clo, chi = cluster_ci(rows, strict_unanimous)
    print(f"\nPRIMARY -- unanimity of the {len(rows[0]['repeats'])} perturbed repeats")
    print(f"  unanimous {k}/{n} = {k/n:.3f}")
    print(f"  Wilson 95%           [{lo:.3f}, {hi:.3f}]")
    print(f"  grasp-clustered 95%  [{clo:.3f}, {chi:.3f}]")

    maj = [r for r in rows if r["majority"] is not None]
    agree = sum(1 for r in maj if bool(r["majority"]) == bool(r["original"]))
    print(f"\nSECONDARY -- majority vote vs the original G3 outcome: "
          f"{agree}/{len(maj)} = {agree/max(len(maj),1):.3f}")

    print(f"\nunanimity by hand:")
    for h in sorted({r["hand"] for r in rows}):
        sel = [r for r in rows if r["hand"] == h]
        u = sum(1 for r in sel if strict_unanimous(r))
        print(f"  {h:8s} {u:3d}/{len(sel):<3d} = {u/len(sel):.3f}")
    print(f"unanimity by shape:")
    for s in sorted({r["shape"] for r in rows}):
        sel = [r for r in rows if r["shape"] == s]
        u = sum(1 for r in sel if strict_unanimous(r))
        print(f"  {s:9s} {u:3d}/{len(sel):<3d} = {u/len(sel):.3f}")
    print(f"unanimity by original outcome:")
    for val, lbl in ((True, "success"), (False, "failure")):
        sel = [r for r in rows if bool(r["original"]) is val]
        if not sel:
            continue
        u = sum(1 for r in sel if strict_unanimous(r))
        print(f"  original {lbl:8s} {u:3d}/{len(sel):<3d} = {u/len(sel):.3f}")

    print("\n=== PRE-DECLARED DECISION RULE ===")
    if k / n >= THRESHOLD and clo > 0.75:
        print(f"  PASS -- unanimity {k/n:.3f} with lower bound {clo:.3f} > 0.75.")
        print("  The task outcome is a stable property of the grasp, so G3's")
        print("  correlations describe something real.")
    else:
        print(f"  FAIL -- unanimity {k/n:.3f}, lower bound {clo:.3f}.")
        print("  The task outcome is noise-dominated. G3's metric result is")
        print("  WITHDRAWN and restated as 'task success in this setup is not")
        print("  reproducible'. Repair the contact/task model before any")
        print("  further metric sweep.")


if __name__ == "__main__":
    main()
