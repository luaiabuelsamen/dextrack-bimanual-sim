"""G5 analysis -- exactly the tests fixed in docs/G5_PREREGISTRATION.md.

Nothing here chooses a test after seeing the data. H1 and H2 carry decision
rules; H3 is reported with intervals and no decision attached, because three
hands and a continuous covariate on one dataset is not a confirmatory test.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.g4_analysis import wilson

H1_THRESHOLD = 0.50
EARLY, LATE = 0.20, 0.80       # window-position split, fixed in advance


def load(path="results/g5_hold.json"):
    d = json.loads(Path(path).read_text())
    return d.get("config", {}), [r for r in d["rows"] if "error" not in r]


def by_cluster(rows):
    """Group observations by sequence -- the resampling unit."""
    g = defaultdict(list)
    for r in rows:
        g[(r["hand"], r["subject"], r["seq"])].append(r)
    return list(g.values())


def cluster_bootstrap(rows, stat, n_boot=4000, seed=0):
    """CI for a statistic, resampling SEQUENCES rather than observations.

    Ten frames from one sequence share a grasp, an object and a retarget; they
    are not ten independent trials. Resampling observations would understate
    the variance, which is the error this repository already made once with 316
    observations that were really 158 clusters of two.
    """
    clusters = by_cluster(rows)
    rng = np.random.default_rng(seed)
    n = len(clusters)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        flat = [r for i in pick for r in clusters[i]]
        v = stat(flat)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"),) * 3
    return (float(np.mean(vals)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def rate(rows):
    return float(np.mean([r["held"] for r in rows])) if rows else None


def main(path):
    cfg, rows = load(path)
    print(f"G5 -- retargeted grasps under gravity")
    print(f"config: {cfg}\n")

    clusters = by_cluster(rows)
    print(f"{len(rows)} observations over {len(clusters)} sequence-clusters, "
          f"{len({r['object'] for r in rows})} objects, "
          f"{len({r['hand'] for r in rows})} hand(s)")

    # ---- H1 -------------------------------------------------------------
    k = sum(r["held"] for r in rows)
    n = len(rows)
    lo, hi = wilson(k, n)
    m, blo, bhi = cluster_bootstrap(rows, rate)
    print(f"\nH1  hold rate {k}/{n} = {k/n:.3f}")
    print(f"    Wilson (ignores clustering) [{lo:.3f}, {hi:.3f}]")
    print(f"    clustered bootstrap         [{blo:.3f}, {bhi:.3f}]")
    verdict = "PASS" if blo > H1_THRESHOLD else (
        "FAIL" if bhi < H1_THRESHOLD else "INCONCLUSIVE")
    print(f"    threshold {H1_THRESHOLD:.2f}  ->  {verdict}")

    # ---- H2 -------------------------------------------------------------
    early = [r for r in rows if r["window_frac"] <= EARLY]
    mid = [r for r in rows if EARLY < r["window_frac"] < LATE]

    def diff(flat):
        e = [r for r in flat if r["window_frac"] <= EARLY]
        m_ = [r for r in flat if EARLY < r["window_frac"] < LATE]
        if not e or not m_:
            return None
        return rate(m_) - rate(e)

    d, dlo, dhi = cluster_bootstrap(rows, diff)
    print(f"\nH2  early (<= {EARLY:.0%} of window) {rate(early):.3f}  n={len(early)}")
    print(f"    middle                        {rate(mid):.3f}  n={len(mid)}")
    print(f"    middle - early = {d:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]")
    print(f"    -> {'SUPPORTED' if dlo > 0 else 'not supported'}")

    # ---- H3, exploratory ------------------------------------------------
    print("\nH3  (exploratory; no decision rule attached)")
    for hand in sorted({r["hand"] for r in rows}):
        sub = [r for r in rows if r["hand"] == hand]
        kk = sum(r["held"] for r in sub)
        m_, l_, h_ = cluster_bootstrap(sub, rate)
        print(f"    {hand:9s} {kk:4d}/{len(sub):4d} = {kk/len(sub):.3f}  "
              f"[{l_:.3f}, {h_:.3f}]")

    print("\n    by intent:")
    for it in sorted({r["intent"] for r in rows}):
        sub = [r for r in rows if r["intent"] == it]
        kk = sum(r["held"] for r in sub)
        print(f"      {it:12s} {kk:4d}/{len(sub):4d} = {kk/len(sub):.3f}  "
              f"({len({r['seq'] for r in sub})} sequences)")

    print("\n    worst objects (>= 3 sequences):")
    per = defaultdict(list)
    for r in rows:
        per[r["object"]].append(r)
    scored = [(rate(v), o, len(v), len({x['seq'] for x in v}))
              for o, v in per.items() if len({x['seq'] for x in v}) >= 3]
    for rt, o, nn, ns in sorted(scored)[:8]:
        print(f"      {o:14s} {rt:.3f}  ({nn} obs, {ns} seqs)")

    held = [r for r in rows if r["held"]]
    fell = [r for r in rows if not r["held"]]
    if held and fell:
        print(f"\n    contacts at reset: held {np.mean([r['n_contact'] for r in held]):.1f}"
              f"  fell {np.mean([r['n_contact'] for r in fell]):.1f}")
        print(f"    contact err (mm):  held {np.mean([r['contact_err_mm'] for r in held]):.1f}"
              f"  fell {np.mean([r['contact_err_mm'] for r in fell]):.1f}")
        print(f"    penetration (mm):  held {np.mean([r['pen_mm'] for r in held]):.1f}"
              f"  fell {np.mean([r['pen_mm'] for r in fell]):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/g5_hold.json")
    main(ap.parse_args().path)
