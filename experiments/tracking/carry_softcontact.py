"""Does softer physics change which seeds relax into a grasp?

DexTrack's setup differs from ours in the direction that would make burial
hurt less: PhysX contacts with a capped depenetration velocity, soft joint PD
(kp 20, kv 1), an actuated free base instead of a weld. This runs the SAME
carry-scored search under a small grid of MuJoCo settings that move our scene
that way, on the four references that stay buried under every accepted offset
(alarm clock, apple, bowl, bunny) and on two that relax cleanly as controls
(cube, flashlight):

    default    the scene as shipped
    soft       hand-object contact solref timeconst 0.02 -> 0.06 s (three times softer)
    softweak   soft + every finger actuator's force range scaled by 0.3
    compliant  weld solref 0.01 -> 0.10 s (the wrist yields)
    all        all three

The score, the search, the seeds and the clean criterion are unchanged, so
the only variable is the physics. Reported per setting: the accepted pose's
tail penetration, grip, links, and whether it is clean; the carry is also
re-rolled under 8 wrist perturbations so a knife-edge does not count.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import mujoco

from oppdef.human import grab, track as T
from experiments.tracking.stage2_carry import carry, score_of, search

SETTINGS = {
    "default": {},
    "soft": {"contact_tc": 0.06},
    "softweak": {"contact_tc": 0.06, "force_scale": 0.3},
    "compliant": {"weld_tc": 0.10},
    "all": {"contact_tc": 0.06, "force_scale": 0.3, "weld_tc": 0.10},
}


def apply(rt, cfg):
    m = rt.sim.model
    if "contact_tc" in cfg:
        for g in list(rt.sim.hand_gids) + list(rt.sim.obj_gids):
            m.geom_solref[g, 0] = cfg["contact_tc"]
    if "force_scale" in cfg:
        for a in range(m.nu):
            if m.actuator_forcelimited[a]:
                m.actuator_forcerange[a] *= cfg["force_scale"]
            else:
                m.actuator_forcelimited[a] = 1
                m.actuator_forcerange[a] = [-5.0 * cfg["force_scale"],
                                            5.0 * cfg["force_scale"]]
    if "weld_tc" in cfg:
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD:
                m.eq_solref[e, 0] = cfg["weld_tc"]


def one(row, setting, a, rng):
    seq = grab.load(f"{row['subject']}/{row['seq']}.npz", verts=True, stride=8)
    rt = T.ReferenceTracker(seq, hand="shadow")
    apply(rt, SETTINGS[setting])
    rt.apply_wrist_offset(np.asarray(row["offset"], float))
    k0 = rt.T // 2
    n = rt.T - k0
    s0, sum0 = score_of(carry(rt, k0, n))
    off, s1, sum1, tried = search(rt, k0, n, a.samples, a.rounds, a.seed,
                                  0.015, 0.35)
    sig = np.array([0.0005] * 3 + [0.01] * 3)
    robust = 0
    for _ in range(a.k):
        dlt = rng.normal(size=6) * sig
        rt.apply_wrist_offset(dlt)
        _, sm = score_of(carry(rt, k0, n))
        rt.apply_wrist_offset(-dlt)
        robust += int(sm["clean"])
    return {"seq": row["seq"], "subject": row["subject"], "setting": setting,
            "start": k0, "frames": n, "seed_score": float(s0), "seed": sum0,
            "score": float(s1), "end": sum1, "robust_clean": robust, "k": a.k,
            "offset": (np.asarray(row["offset"]) + off).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grips", default="results/stage2_grips_clean.json")
    ap.add_argument("--seqs", default="alarmclock_lift,apple_eat_1,bowl_drink_1,"
                    "stanfordbunny_inspect_1,cubemedium_inspect_1,flashlight_lift")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--out", default="results/stage2_carry_softcontact.json")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()

    rows = {f"{r['subject']}/{r['seq']}": r
            for r in json.loads(Path(a.grips).read_text())["rows"]}
    want = []
    for name in a.seqs.split(","):
        hits = [k for k in rows if k == name or k.split("/", 1)[1] == name]
        if len(hits) != 1:
            raise SystemExit(f"{name!r} matches {hits}")
        want.append(rows[hits[0]])
    rng = np.random.default_rng(a.seed)
    out, t0 = [], time.time()
    for setting in a.settings.split(","):
        for r in want:
            try:
                rec = one(r, setting, a, rng)
            except Exception as e:                        # noqa: BLE001
                print(f"  {r['seq']} {setting}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            out.append(rec)
            e = rec["end"]
            print(f"{setting:9s} {rec['seq']:24s} seed {rec['seed_score']:.3f} -> "
                  f"{rec['score']:.3f}  tail {e['tail_pen_mm']:5.2f} mm  "
                  f"{e['tail_bodies']:3.1f} links  {e['tail_grip_n']:7.0f} N "
                  f"({e['tail_grip_x']:5.1f}x)  held={int(e['held'])} "
                  f"clean={int(e['clean'])}  robust {rec['robust_clean']}/{a.k}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            Path(a.out).write_text(json.dumps(
                {"settings": SETTINGS, "samples": a.samples, "rounds": a.rounds,
                 "seed": a.seed, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
