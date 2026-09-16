"""What stage 2 actually produced, including how much of it is burial."""
import json, sys
import numpy as np

d = json.load(open(__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "results/stage2_grips.json"))
r = d["rows"]
held = np.array([x["held"] for x in r])
raw = np.array([x["raw_held"] for x in r])
eq = np.array([x["equilibrium"] for x in r])
nc = np.array([x["n_contact"] for x in r], float)
drop = np.array([x["drop_m"] for x in r]) * 1000

print(f"{len(r)} references, hand = {d['hand']}\n")
print(f"hold rate   raw retarget      {raw.sum():2d}/{len(r)}  {raw.mean():.3f}")
print(f"            after the search  {held.sum():2d}/{len(r)}  {held.mean():.3f}")
lo, hi = _ci = None, None
# clustered by OBJECT, since sequences of the same object are near-duplicates
objs = {}
for x, h in zip(r, held):
    objs.setdefault(x["object"], []).append(bool(h))
rng = np.random.default_rng(0)
keys = list(objs)
boot = [np.mean([v for k in rng.choice(keys, len(keys)) for v in objs[k]])
        for _ in range(4000)]
print(f"            95% CI (clustered by object, {len(keys)} clusters) "
      f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]\n")

h = held.astype(bool)
print(f"of the {h.sum()} that hold, contacts at the accepted grasp:")
for name, sel in (("<= 12 contacts (a grasp)", nc[h] <= 12),
                  ("13-30            ", (nc[h] > 12) & (nc[h] <= 30)),
                  ("> 30 (burial)    ", nc[h] > 30)):
    print(f"  {name:26s} {int(sel.sum()):2d}/{int(h.sum())}")
print(f"\n  median contacts {np.median(nc[h]):.0f}   median drop {np.median(drop[h]):.1f} mm")

print("\nthe six that never hold:")
for x in r:
    if not x["held"]:
        print(f"  {x['seq']:26s} {x['object']:14s} drop {x['drop_m']*1000:8.1f} mm"
              f"  n={x['n_contact']:3d}  eq={x['equilibrium']:6.2f}x")

print("\nworst burial among the holds (by contact count):")
for x in sorted([y for y in r if y["held"]], key=lambda y: -y["n_contact"])[:6]:
    print(f"  {x['seq']:26s} {x['object']:14s} n={x['n_contact']:3d}"
          f"  grip {x['grip_n']:9.0f} N  drop {x['drop_m']*1000:6.2f} mm")
