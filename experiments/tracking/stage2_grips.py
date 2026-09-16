"""Stage 2 gate: produce a grasp per reference and MEASURE whether it holds.

The retarget is a prior on where to search, not the state anything starts from
-- G5 measured the raw fit holding in 25.5% of frames, with failures making 0.1
contacts at reset against 17.4 for successes. So this runs retarget -> CEM
wrist search scored by a physical hold test -> hold, and reports the rate.

`equilibrium_residual` is recorded alongside because drop distance alone cannot
separate the two failure modes: a buried non-grasp need not drop, and a hand
that never touches drops exactly as fast as no hand at all. Residual reads 1.0x
for free fall and hundreds for burial, so the pair localises what went wrong.

The accepted grasps are cached, because stage 3 must start from a grip that
holds rather than from the fit.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

from oppdef.human import grab, scene, track as T, grasp as G
from oppdef.human.retarget import retarget_sequence
from experiments.tracking.grab_inventory import contact_mask, longest_run, _tree


def one(row, hand="shadow", stride=8, samples=24, rounds=2, seconds=0.8,
        grip=8.0, seed=0):
    name = row["seq"]
    s = grab.load(f"{row['subject']}/{name}.npz", verts=True, stride=stride)
    mk = contact_mask(s, _tree(s, {}), "rhand")
    st, ln = longest_run(mk)
    if ln < 5:
        return None
    fit = scene.build(hand, s.obj, obj_static=True)
    tr = retarget_sequence(s, "rhand", hand, window=(st, ln), sc=fit)

    env = T.GrabTrackEnv(s, hand=hand)
    k = len(tr.q) // 2                       # NOT frame 0: frame 0 of the hold
    q0 = tr.q[k]                             # window is a 20x outlier here
    base = env.hold(q0, seconds=seconds, grip=grip)
    fitq = G.synthesize(env, q0, samples=samples, rounds=rounds,
                        seconds=seconds, grip=grip, seed=seed)
    res = T.equilibrium_residual(env.sc, int(env.sc.model.jnt_bodyid[env.obj_jid]))
    return {
        "seq": name, "subject": row["subject"], "object": row["object"],
        "intent": row["intent"], "frame": int(tr.frames[k]),
        "raw_drop_m": float(base.drop_m), "raw_held": bool(base.drop_m < 0.05),
        "drop_m": float(fitq.drop_m), "held": bool(fitq.held),
        "n_contact": int(fitq.n_contact), "grip_n": float(fitq.grip_n),
        "equilibrium": float(res), "offset": fitq.offset.tolist(),
        "tried": int(fitq.tried),
        "q": fitq.q.tolist(), "ctrl": fitq.ctrl.tolist(),
        "window": [int(st), int(ln)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="shadow")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="results/stage2_grips.json")
    ap.add_argument("--seqs", default="",
                    help="comma-separated sequence names; overrides the "
                         "one-per-object default. Used to compute seeds for "
                         "exactly the references a downstream stage selects -- "
                         "the two stages otherwise pick different sequences of "
                         "the same object and the seeds do not apply.")
    ap.add_argument("--min-hold", type=int, default=5)
    a = ap.parse_args()

    rows = json.load(open("results/grab_inventory.json"))["rows"]
    if a.seqs:
        # Keyed by SUBJECT/SEQ. GRAB has 80 sequence names that exist under more
        # than one subject -- camera_takepicture_2 holds for 161 frames under s1
        # and 76 under s2 -- so a bare-name dict silently keeps one of them, and
        # a downstream stage that looks a seed up by name gets a grasp belonging
        # to a different recording. That happened: 7 of 10 references in the
        # first seeded stage-3 run were trained with another subject's wrist
        # offset, and it was invisible until two sessions computed the same
        # start count on different clips and disagreed.
        want = [x.strip() for x in a.seqs.split(",") if x.strip()]
        by = {f"{r['subject']}/{r['seq']}": r for r in rows}
        by_bare = {}
        for r in rows:
            by_bare.setdefault(r["seq"], []).append(r)
        order, missing = [], []
        for w in want:
            if w in by:
                order.append(by[w])
            elif w in by_bare and len(by_bare[w]) == 1:
                order.append(by_bare[w][0])
            elif w in by_bare:
                raise SystemExit(
                    f"{w!r} exists under {len(by_bare[w])} subjects "
                    f"({', '.join(r['subject'] for r in by_bare[w])}); "
                    f"name it as SUBJECT/SEQ")
            else:
                missing.append(w)
        if missing:
            print(f"  not in inventory: {missing}", file=sys.stderr, flush=True)
        return _run(a, order)
    cand = [r for r in rows if (r.get("rhand_hold_len") or 0) >= a.min_hold]
    # one sequence per object first, so the sample is not eight mugs
    seen, order = set(), []
    for r in sorted(cand, key=lambda x: x["seq"]):
        if r["object"] not in seen:
            seen.add(r["object"]); order.append(r)
    order += [r for r in sorted(cand, key=lambda x: x["seq"]) if r not in order]
    order = order[:a.n]
    return _run(a, order)


def _run(a, order):
    out, t0 = [], time.time()
    for i, r in enumerate(order):
        try:
            rec = one(r, hand=a.hand, stride=a.stride, samples=a.samples,
                      rounds=a.rounds)
        except Exception as e:                       # a bad mesh must not stop the sweep
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        if rec is None:
            continue
        out.append(rec)
        print(f"[{i+1}/{len(order)}] {rec['seq']:22s} raw {rec['raw_drop_m']*1000:7.1f} mm"
              f" -> {rec['drop_m']*1000:7.1f} mm  held={rec['held']}"
              f"  eq={rec['equilibrium']:6.2f}x  n={rec['n_contact']:3d}"
              f"  {time.time()-t0:6.0f}s", file=sys.stderr, flush=True)
        Path(a.out).write_text(json.dumps({"hand": a.hand, "rows": out}, indent=1))

    held = [r["held"] for r in out]
    raw = [r["raw_held"] for r in out]
    print(f"\nstage 2: {sum(held)}/{len(held)} hold after search, "
          f"{sum(raw)}/{len(raw)} from the raw fit", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
