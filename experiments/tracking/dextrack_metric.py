"""Score our rollouts on DexTrack's success criterion, next to ours.

DexTrack (Liu et al., ICLR 2025, Table 1) calls a tracking attempt a success
when, over the rollout, object translation error <= 10 cm, rotation error
<= 20 deg (strict) or 40 deg (loose), and 0.5 E_wrist + 0.5 E_finger <= 0.8
(strict) or 1.2 (loose). On 197 GRAB s1 sequences their generalist scores
46.70 / 65.48 and their per-trajectory PPO baseline 38.58 / 54.82.

Our criterion is the end state: held, under 3 mm penetration, at least two
links, opposed, under 40x weight. Theirs never looks at the hand-object
contact. Both are reported here on the same rollouts so the two claims can
be read side by side. Errors are MEANS over the rollout, as theirs are.

Approximations, stated: E_wrist is the palm's position error in metres plus
its rotation error in radians against the retargeted wrist trajectory;
E_finger is the mean absolute joint error in radians against the retargeted
finger angles. Their exact normalisation is not published, so the hand-error
half of the criterion is ours; the object half is exact.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import mujoco
import torch

from handsim.human import grab, track as T, rl
from experiments.tracking.stage2_carry import contact_state, LOST_M, CLEAN_GRIP_X, W

STRICT = (0.10, np.radians(20), 0.8)
LOOSE = (0.10, np.radians(40), 1.2)


def hand_errors(rt, k):
    """(wrist error, finger error) against the retargeted trajectory at k."""
    m, d = rt.sim.model, rt.sim.data
    p = d.xpos[rt.sim.wrist_bid]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, d.xmat[rt.sim.wrist_bid].copy())
    e_p = float(np.linalg.norm(p - rt.P[k]))
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, q, rt.Q[k] * np.array([1.0, -1, -1, -1]))
    e_r = float(2 * np.arccos(np.clip(abs(dq[0]), 0, 1)))
    qh = d.qpos[rt.sim.qadr]
    ref = rt.mh.config_from(rt.vals[k])
    e_f = float(np.mean(np.abs(qh - ref)))
    return e_p + e_r, e_f


def roll(rt, k0, net=None):
    cfg = rl.RLConfig()
    scale = np.concatenate([np.full(3, cfg.a_pos), np.full(3, cfg.a_rot),
                            np.full(rt.n_action - 6, cfg.a_fin)])
    rt.reset_at(k0)
    pe, re, ew, ef = [], [], [], []
    for k in range(k0, rt.T):
        if net is not None:
            x = torch.as_tensor(rt.observe(k), dtype=torch.float32)[None]
            with torch.no_grad():
                a = net.dist(x).mean.numpy()[0]
            rt.apply(k, np.clip(a, -1, 1) * scale)
        else:
            rt.apply(k, None)
        for _ in range(rt.ctrl_every):
            mujoco.mj_step(rt.sim.model, rt.sim.data)
        p, r = rt.error(k); pe.append(p); re.append(r)
        w, f = hand_errors(rt, k); ew.append(w); ef.append(f)
    pe, re, ew, ef = map(np.array, (pe, re, ew, ef))
    pen, n, nb, sided, grip = contact_state(rt.sim)
    hand = 0.5 * ew.mean() + 0.5 * ef.mean()
    out = {"T_err_cm": float(pe.mean() * 100), "R_err_deg": float(np.degrees(re.mean())),
           "E_wrist": float(ew.mean()), "E_finger": float(ef.mean()),
           "hand_err": float(hand),
           "end_pen_mm": pen * 1000, "end_links": nb, "end_grip_x": grip / W,
           "end_held": bool(pe[-1] <= LOST_M)}
    for name, (t, r, h) in (("dextrack_strict", STRICT), ("dextrack_loose", LOOSE)):
        out[name] = bool(pe.mean() <= t and re.mean() <= r and hand <= h)
    out["ours_clean"] = bool(out["end_held"] and pen < 0.003 and nb >= 2
                             and sided < 0.8 and grip / W < CLEAN_GRIP_X)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="results/stage2_carry_fcap_robust_seeds.json")
    ap.add_argument("--ckpt", default="results/ppo_carry")
    ap.add_argument("--out", default="results/dextrack_metric.json")
    a = ap.parse_args()
    rows = json.loads(Path(a.seeds).read_text())["rows"]
    out = []
    for r in rows:
        seq = grab.load(f"{r['subject']}/{r['seq']}.npz", verts=True, stride=8)
        rt = T.ReferenceTracker(seq, hand="shadow")
        rt.apply_wrist_offset(np.asarray(r["offset"], float))
        k0 = int(r["start"])
        rec = {"seq": r["seq"], "subject": r["subject"], "start": k0, "T": int(rt.T),
               "robust_clean": r.get("robust_clean")}
        rec["feedforward"] = roll(rt, k0)
        ck = Path(a.ckpt) / f"ppo_{r['seq']}_carry.pt"
        if ck.exists():
            net = rl.make_policy(rt.n_obs, rt.n_action)
            net.load_state_dict(torch.load(ck, map_location="cpu")); net.eval()
            rec["ppo"] = roll(rt, k0, net)
        out.append(rec)
        f = rec["feedforward"]
        line = (f"{r['seq']:24s} FF  T {f['T_err_cm']:5.2f} cm  R {f['R_err_deg']:5.1f} deg  "
                f"hand {f['hand_err']:.2f}  strict={int(f['dextrack_strict'])} "
                f"loose={int(f['dextrack_loose'])} ours={int(f['ours_clean'])}")
        if "ppo" in rec:
            p = rec["ppo"]
            line += (f" | PPO T {p['T_err_cm']:5.2f} R {p['R_err_deg']:5.1f} "
                     f"strict={int(p['dextrack_strict'])} loose={int(p['dextrack_loose'])} "
                     f"ours={int(p['ours_clean'])}")
        print(line, flush=True)
        Path(a.out).write_text(json.dumps({"seeds": a.seeds, "rows": out}, indent=1))
    n = len(out)
    for who in ("feedforward", "ppo"):
        sel = [x[who] for x in out if who in x]
        if sel:
            print(f"\n{who}: n={len(sel)}  DexTrack strict {sum(x['dextrack_strict'] for x in sel)}  "
                  f"loose {sum(x['dextrack_loose'] for x in sel)}  ours clean "
                  f"{sum(x['ours_clean'] for x in sel)}   (of 40 references: "
                  f"strict {100*sum(x['dextrack_strict'] for x in sel)/40:.0f}%  "
                  f"loose {100*sum(x['dextrack_loose'] for x in sel)/40:.0f}%)")


if __name__ == "__main__":
    main()
