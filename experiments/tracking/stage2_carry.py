"""Stage 2, scored by the END of a short carry instead of a static hold.

The static hold test is satisfied best by burying the hand inside the object,
and the open-loop carry sweep (`stage2_survives.py`) showed that the poses which
actually carry the object start buried and RELAX into a contact set in the first
frames of motion -- stamp_lift from 14 mm / 878 N to 0.3 mm / 4 N on five
fingers. The property the search needs is therefore dynamic: not "touch without
penetrating" at reset (three searches found nothing), but "end a short carry
held, out of the object, and opposed".

So this scores a wrist offset by carrying the hand along the reference for
`--frames` control frames from the middle of the hold window and reading the
LAST few frames of that carry: tracking error, whether the object is still in
hand, penetration depth, how many distinct hand links touch it, and how
one-sided those contacts are. The tail is averaged over several frames because
a single end frame is noisy -- the same stamp carry read 5 contacts after
frame 26 and 2 after frame 25.

The search is the same greedy wrist-offset hill-climb `synthesize_grasp` uses
(`objective="track"`), started from the stage-2 seed so the comparison is
"same neighbourhood, different objective". Every row records the seed's own
carry score first, so the table shows before -> after on identical starts.

Nothing here is a static hold, nothing gates penetration at reset, and no
controller is in the loop.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import mujoco

from oppdef.human import grab, track as T

W = 0.2 * 9.81            # object weight this pipeline standardises on

#: what counts as a CLEAN end state. Held: tail error under the drop threshold.
#: Out of the object: tail max-penetration under 3 mm. Opposed: at least two
#: distinct hand links touching, contact normals not all one way.
LOST_M = 0.10
CLEAN_PEN_M = 0.003
CLEAN_BODIES = 2
CLEAN_SIDED = 0.8
TAIL = 5


def contact_state(sc):
    """(max penetration m, n contacts, n distinct hand bodies, one-sidedness,
    total normal force N) of the CURRENT hand-object contact set."""
    m, d = sc.model, sc.data
    objs, hands = set(sc.obj_gids), set(sc.hand_gids)
    f = np.zeros(6)
    pen, n, bodies, acc, tot = 0.0, 0, set(), np.zeros(3), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ({g1, g2} & objs and {g1, g2} & hands):
            continue
        hg = g1 if g1 in hands else g2
        bodies.add(int(m.geom_bodyid[hg]))
        v = np.array(c.frame[:3])
        acc += -v if g1 in objs else v
        pen = max(pen, -float(c.dist))
        mujoco.mj_contactForce(m, d, i, f)
        tot += abs(float(f[0]))
        n += 1
    sided = float(np.linalg.norm(acc) / n) if n else 1.0
    return pen, n, len(bodies), sided, tot


def carry(rt, k0, n, record=False):
    """Carry the hand along the reference from frame k0 for n control frames,
    feedforward only. Returns per-frame arrays: pos_err, pen, ncon, nbodies,
    sided, grip -- every quantity read LIVE after the frame's last step."""
    m, d = rt.sim.model, rt.sim.data
    rt.reset_at(k0)
    cols = np.empty((n, 6))
    for i in range(n):
        rt.apply(k0 + i, None)
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(m, d)
        pe, _ = rt.error(k0 + i)
        cols[i] = (pe,) + contact_state(rt.sim)
    return cols


def score_of(cols, tail=TAIL):
    """Lower is better. Units are metres for the error and penetration terms,
    so 10 mm of penetration beyond the allowance costs what 70 mm of tracking
    error does, and losing the object costs more than either."""
    err, pen, ncon, nb, sided, grip = cols.T
    t = cols[-tail:]
    err_t, pen_t, ncon_t, nb_t, sided_t, grip_t = t.T
    held = bool(err_t.mean() < LOST_M)
    lost_frac = float(np.mean(err > LOST_M))
    pen_tail = float(pen_t.max())            # the worst frame of the tail
    nb_tail = float(nb_t.mean())
    sided_tail = float(sided_t.mean())
    wrap = sided_tail + (1.0 - min(nb_tail, 5.0) / 5.0)
    s = (float(np.minimum(err, 0.25).mean())
         + 0.5 * lost_frac
         + 10.0 * max(0.0, pen_tail - CLEAN_PEN_M)
         + 0.05 * wrap)
    clean = bool(held and pen_tail < CLEAN_PEN_M
                 and nb_tail >= CLEAN_BODIES and sided_tail < CLEAN_SIDED)
    return s, {
        "held": held, "clean": clean,
        "relaxed": bool(held and pen_tail < CLEAN_PEN_M),
        "mean_mm": float(err.mean() * 1000),
        "tail_mm": float(err_t.mean() * 1000),
        "tail_pen_mm": pen_tail * 1000,
        "tail_contacts": float(ncon_t.mean()), "tail_bodies": nb_tail,
        "tail_sided": sided_tail, "tail_grip_n": float(grip_t.mean()),
        # read after the FIRST control frame, not at reset -- the reset state
        # is in `stage2_survives.json`; this is one frame of motion later
        "frame1_pen_mm": float(pen[0] * 1000), "frame1_grip_n": float(grip[0]),
    }


def search(rt, k0, n, samples, rounds, seed, sigma_pos, sigma_rot, log=None):
    """Greedy wrist-offset hill-climb on the carry score. Returns
    (offset relative to the start, best score, best summary, evaluations)."""
    rng = np.random.default_rng(seed)
    sig0 = np.array([sigma_pos] * 3 + [sigma_rot] * 3)
    best, best_sum = score_of(carry(rt, k0, n))
    base, tried = np.zeros(6), 1
    for r in range(rounds):
        sig = sig0 * (0.6 ** r)                # narrow as it converges
        for dlt in rng.normal(size=(samples, 6)) * sig:
            rt.apply_wrist_offset(dlt)
            v, sm = score_of(carry(rt, k0, n))
            tried += 1
            if v < best:
                best, best_sum, base = v, sm, base + dlt
                if log:
                    log(f"      round {r} accept {v:.4f}  "
                        f"{sm['tail_pen_mm']:.2f} mm  {sm['tail_bodies']:.1f} bodies  "
                        f"sided {sm['tail_sided']:.2f}  held={sm['held']}")
            else:
                rt.apply_wrist_offset(-dlt)
    return base, best, best_sum, tried


def one(row, a, log):
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = T.ReferenceTracker(seq, hand="shadow")
    seed_off = np.asarray(row["offset"], float)
    rt.apply_wrist_offset(seed_off)
    k0 = rt.T // 2                             # the frame stage 2 scored on
    n = int(min(a.frames, rt.T - k0))
    rec = {"seq": row["seq"], "subject": row["subject"], "object": row["object"],
           "seed_held": bool(row["held"]), "seed_contacts": int(row["n_contact"]),
           "seed_grip_n": float(row["grip_n"]), "start": int(k0),
           "frames": int(n), "T": int(rt.T)}
    s0, sum0 = score_of(carry(rt, k0, n))
    rec["seed_score"], rec["seed"] = float(s0), sum0
    off, s1, sum1, tried = search(rt, k0, n, a.samples, a.rounds, a.seed,
                                  a.sigma_pos, a.sigma_rot, log)
    rec["score"], rec["end"], rec["tried"] = float(s1), sum1, int(tried)
    rec["delta"] = off.tolist()
    rec["offset"] = (seed_off + off).tolist()   # absolute, like the stage-2 row
    # the per-frame trace of the accepted pose, so the relaxation curve is on
    # file without re-running it
    cols = carry(rt, k0, n)
    rec["trace"] = {k: np.round(v, 5).tolist() for k, v in zip(
        ("err_m", "pen_m", "ncon", "nbodies", "sided", "grip_n"), cols.T)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grips", default="results/stage2_grips_clean.json")
    ap.add_argument("--out", default="results/stage2_carry.json")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sigma-pos", type=float, default=0.015)
    ap.add_argument("--sigma-rot", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqs", default="", help="comma-separated subset")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    rows = json.loads(Path(a.grips).read_text())["rows"]
    if a.seqs:
        want = set(x.strip() for x in a.seqs.split(","))
        rows = [r for r in rows if r["seq"] in want or f"{r['subject']}/{r['seq']}" in want]
    if a.n:
        rows = rows[:a.n]
    print(f"{len(rows)} references from {a.grips}; carry {a.frames} frames "
          f"from T//2, {a.samples}x{a.rounds} search", flush=True)
    log = (lambda s: print(s, flush=True)) if a.verbose else None

    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        try:
            rec = one(r, a, log)
        except Exception as e:                  # a bad mesh must not stop it
            print(f"  {r['seq']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        out.append(rec)
        s, e = rec["seed"], rec["end"]
        flag = "CLEAN" if e["clean"] else ("relaxed" if e["relaxed"] else
                                           ("held" if e["held"] else "drop"))
        print(f"[{i+1}/{len(rows)}] {rec['seq']:24s} n={rec['frames']:2d}  "
              f"seed {rec['seed_score']:7.3f} ({s['tail_pen_mm']:5.2f} mm, "
              f"{s['tail_bodies']:3.1f} b, held={int(s['held'])})  ->  "
              f"{rec['score']:7.3f} ({e['tail_pen_mm']:5.2f} mm, "
              f"{e['tail_bodies']:3.1f} b, sided {e['tail_sided']:.2f}, "
              f"{e['tail_grip_n']:6.0f} N, err {e['tail_mm']:6.1f} mm)  {flag:7s} "
              f"({time.time()-t0:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps(
            {"grips": a.grips, "frames": a.frames, "samples": a.samples,
             "rounds": a.rounds, "seed": a.seed, "sigma_pos": a.sigma_pos,
             "sigma_rot": a.sigma_rot, "rows": out}, indent=1))

    def count(key, which):
        return sum(bool(r[which][key]) for r in out)
    print(f"\n{len(out)} references")
    for key in ("held", "relaxed", "clean"):
        print(f"  {key:8s} seed {count(key, 'seed'):2d}  ->  "
              f"carry-scored {count(key, 'end'):2d}")
    cl = [r["seq"] for r in out if r["end"]["clean"]]
    print(f"  clean end states: {cl}")


if __name__ == "__main__":
    main()
