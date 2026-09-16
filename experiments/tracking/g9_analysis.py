"""Read out stages 3, 4 and 5 from one g9 run.

Three separate questions that a single mean would blur:

  stage 3  does a policy trained on ONE reference track that reference?
  stage 4  does one network fit the union of those policies' behaviour?
  stage 5  does it transfer to an object no policy in the mixture ever saw?

Stage 5 is the only one of the three that can fail for an interesting reason,
so it is reported per held-out reference rather than pooled, next to the
feedforward baseline and that reference's own PPO -- the two numbers that say
whether distillation gained or lost anything.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/g9_ppo_distill.json")
    a = ap.parse_args()
    d = json.loads(Path(a.path).read_text())
    per = d.get("per_reference", [])
    held = d.get("held_out", [])

    print(f"hand {d.get('hand','shadow')}, {d.get('steps'):,} steps/reference, "
          f"seed {d.get('seed')}\n")

    # Join the stage-2 grasp each policy was seeded from, and say whether it
    # was a GRASP or a BURIAL. A peer session measured that the mug's stage-2
    # wrist offset IS its burial -- 20.6 mm and 13.6 kN -- and that a policy
    # tracking from it at 23.2 mm drops the object entirely once un-buried.
    # So a tracking number means nothing without the class of the state it
    # started from, and the stage-2 sweep says 6 of 34 successes are burials.
    seeds = {}
    for cand in ("results/stage2_grips_g9.json", "results/stage2_grips.json"):
        q = Path(cand)
        if q.exists():
            for r in json.loads(q.read_text())["rows"]:
                seeds.setdefault(r["seq"], r)

    def kind(n):
        r = seeds.get(n)
        if r is None or not r.get("held"):
            return "?", float("nan"), float("nan")
        nc, gn = r["n_contact"], r["grip_n"]
        return ("BURIAL" if nc > 30 else "grasp" if nc <= 12 else "mixed"), nc, gn

    print(f"STAGE 3 -- per-reference PPO, {len(per)} references trained")
    if per:
        mm = np.array([x["ppo_mm"] for x in per])
        print(f"  {'reference':26s} {'object':14s} {'PPO':>9}  {'seed':>6} "
              f"{'con':>4} {'grip N':>9}")
        for x in sorted(per, key=lambda y: y["ppo_mm"]):
            k, nc, gn = kind(x["seq"])
            print(f"  {x['seq'][:26]:26s} {x['object']:14s} {x['ppo_mm']:9.1f}"
                  f"  {k:>6} {nc:4.0f} {gn:9.0f}")
        print(f"  median {np.median(mm):.1f} mm   under 50 mm: "
              f"{int((mm < 50).sum())}/{len(mm)}")
        gr = np.array([x["ppo_mm"] for x in per if kind(x["seq"])[0] == "grasp"])
        bu = np.array([x["ppo_mm"] for x in per if kind(x["seq"])[0] == "BURIAL"])
        if len(gr):
            print(f"  from a GRASP seed  ({len(gr)}): median {np.median(gr):8.1f} mm")
        if len(bu):
            print(f"  from a BURIAL seed ({len(bu)}): median {np.median(bu):8.1f} mm"
                  f"   <- read these as tracking the contact solver")
    if not held:
        print("\nSTAGES 4-5 -- not reached: distillation needs >= 3 references "
              "so an object can be held out.")
        return

    print(f"\nSTAGE 4-5 -- distilled on all but {d.get('held_out_objects')}, "
          f"evaluated on the held-out objects")
    print(f"  {'reference':26s} {'object':14s} {'feedfwd':>10} {'own PPO':>10} "
          f"{'DISTILLED':>10}")
    for x in held:
        print(f"  {x['seq'][:26]:26s} {x['object']:14s} "
              f"{x['feedforward_mm']:10.1f} {x['ppo_mm']:10.1f} "
              f"{x['distilled_mm']:10.1f}")
    dm = np.array([x["distilled_mm"] for x in held])
    ff = np.array([x["feedforward_mm"] for x in held])
    pp = np.array([x["ppo_mm"] for x in held])
    print(f"\n  median   feedforward {np.median(ff):10.1f} mm")
    print(f"           own PPO     {np.median(pp):10.1f} mm")
    print(f"           DISTILLED   {np.median(dm):10.1f} mm")
    print(f"\n  distilled beats feedforward on {int((dm < ff).sum())}/{len(dm)}")
    print(f"  distilled within 2x of that reference's own PPO on "
          f"{int((dm < 2 * pp).sum())}/{len(dm)}")


if __name__ == "__main__":
    main()
