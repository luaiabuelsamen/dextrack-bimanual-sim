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

    print(f"STAGE 3 -- per-reference PPO, {len(per)} references trained")
    if per:
        mm = np.array([x["ppo_mm"] for x in per])
        for x in sorted(per, key=lambda y: y["ppo_mm"]):
            print(f"  {x['seq'][:26]:26s} {x['object']:14s} {x['ppo_mm']:8.1f} mm"
                  f"  {x['transitions']:5d} transitions")
        print(f"  median {np.median(mm):.1f} mm   under 50 mm: "
              f"{int((mm < 50).sum())}/{len(mm)}")
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
