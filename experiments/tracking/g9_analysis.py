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
    # The incremental writer emits the store dict verbatim, which names the
    # object `obj`; the final writer renames it. Accept either rather than
    # crash on a partial file -- reading partial results as they land is the
    # whole point of writing them incrementally.
    for x in per:
        x.setdefault("object", x.get("obj", "?"))
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
                seeds.setdefault(f"{r['subject']}/{r['seq']}", r)

    # Classify on contact count AND grip force. Contact count alone calls
    # mug_drink_2 a grasp on 9 contacts while it grips at 528 N -- 264x the
    # object's own weight -- and calls flashlight_on_2 a grasp on 3 contacts at
    # 0.6 N, which is LESS than the object weighs and is barely touching it.
    # Both extremes matter: a policy handed the first is tracking a near-burial,
    # and one handed the second has almost nothing to hold.
    W = 0.2 * 9.81          # the object mass this pipeline standardises on

    def kind(n):
        r = seeds.get(n)     # n must be subject/seq; a bare name is ambiguous
        if r is None or not r.get("held"):
            return "?", float("nan"), float("nan")
        nc, gn = r["n_contact"], r["grip_n"]
        if nc > 30 or gn > 200 * W:
            k = "BURIAL"
        elif gn < W:
            k = "thin"
        elif nc <= 12 and gn <= 50 * W:
            k = "grasp"
        else:
            k = "mixed"
        return k, nc, gn

    print(f"STAGE 3 -- per-reference PPO, {len(per)} references trained")
    if per:
        mm = np.array([x["ppo_mm"] for x in per])
        print(f"  {'reference':24s} {'object':12s} {'PPO':>8}  {'seed':>6} "
              f"{'con':>4} | {'end pen':>8} {'end con':>7} {'end grip':>9}")
        for x in sorted(per, key=lambda y: y["ppo_mm"]):
            k, nc, gn = kind(f"{x.get('subject','?')}/{x['seq']}")
            print(f"  {x['seq'][:24]:24s} {x['object'][:12]:12s} "
                  f"{x['ppo_mm']:8.1f}  {k:>6} {nc:4.0f} | "
                  f"{x.get('end_pen_mm', float('nan')):8.2f} "
                  f"{x.get('end_contacts', -1):7d} "
                  f"{x.get('end_grip_n', float('nan')):9.0f}")
        print(f"  median {np.median(mm):.1f} mm   under 50 mm: "
              f"{int((mm < 50).sum())}/{len(mm)}")
        for lab in ("thin",):
            sel = [x for x in per if kind(f"{x.get('subject','?')}/{x['seq']}")[0] == lab]
            if sel:
                print(f"  from a THIN seed   ({len(sel)}): gripping less than the "
                      f"object weighs -- little for a policy to hold")
        gr = np.array([x["ppo_mm"] for x in per if kind(f"{x.get('subject','?')}/{x['seq']}")[0] == "grasp"])
        bu = np.array([x["ppo_mm"] for x in per if kind(f"{x.get('subject','?')}/{x['seq']}")[0] == "BURIAL"])
        if len(gr):
            print(f"  from a GRASP seed  ({len(gr)}): median {np.median(gr):8.1f} mm")
        if len(bu):
            print(f"  from a BURIAL seed ({len(bu)}): median {np.median(bu):8.1f} mm"
                  f"   <- read these as tracking the contact solver")

        # The end state decides whether ANY of these is a tracking result. A
        # peer session measured two policies ending a "successful" 111-frame
        # track at 10.83 mm inside on 12 bodies at 1330x weight -- one trained
        # on a burial, one on a valid grasp, identical to the millimetre.
        ep = np.array([x.get("end_pen_mm", np.nan) for x in per], float)
        ok = np.isfinite(ep)
        if ok.any():
            clean = (ep[ok] <= 3.0)
            print(f"\n  END OF ROLLOUT: {int(clean.sum())}/{int(ok.sum())} finish "
                  f"under 3 mm of penetration; median {np.nanmedian(ep):.2f} mm")
            good = [x for x in per
                    if np.isfinite(x.get("end_pen_mm", np.nan))
                    and x["end_pen_mm"] <= 3.0 and x["ppo_mm"] < 50]
            print(f"  tracking under 50 mm AND ending un-buried: "
                  f"{len(good)}/{len(per)}"
                  + ("  <- the only rows that are tracking results"
                     if good else "  <- no row here is a tracking result"))
            for x in good:
                print(f"      {x['seq'][:24]:24s} {x['ppo_mm']:7.1f} mm at "
                      f"{x['end_pen_mm']:.2f} mm, {x.get('end_grip_n',0):.0f} N")
    if not held:
        print("\nSTAGES 4-5 -- not reached: distillation needs >= 3 references "
              "so an object can be held out.")
        return

    mix = d.get("mixture") or {}
    if mix:
        tot = sum(mix.values())
        gr = mix.get("grasp", 0)
        print(f"\ndistillation mixture: "
              + ", ".join(f"{v} {k}-seeded" for k, v in sorted(mix.items())))
        if gr * 2 <= tot:
            print("  WARNING: at most half the mixture started from a grasp. A "
                  "distilled\n  policy trained mostly on burials has been shown "
                  "the contact solver,\n  not a grasp, and its transfer number "
                  "means correspondingly less.")
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
