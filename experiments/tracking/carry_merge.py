"""Merge several search seeds' carry results by robustness into one seed set.

For each reference, keep the seed whose accepted pose stayed clean under the
most of the 8 wrist perturbations, and only if that is at least `--min`.
Prints the per-seed and union counts that the docs quote.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="e.g. results/stage2_carry_rot ; expects _s{k}.json, "
                         "_s{k}_validate.json, _s{k}_robust.json")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--min", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    key = lambda r: f"{r['subject']}/{r['seq']}"
    sw = {s: {key(r): r for r in json.loads(Path(f"{a.prefix}_s{s}.json").read_text())["rows"]}
          for s in seeds}
    rb = {s: {key(r): r for r in json.loads(Path(f"{a.prefix}_s{s}_robust.json").read_text())["rows"]}
          for s in seeds}
    va = {s: {key(r): r for r in json.loads(Path(f"{a.prefix}_s{s}_validate.json").read_text())["rows"]}
          for s in seeds}
    refs = sorted(set().union(*[set(sw[s]) for s in seeds]))
    rows = []
    for k in refs:
        best = None
        for s in seeds:
            r = rb[s].get(k)
            if r and (best is None or r["n_clean"] > best[1]):
                best = (s, r["n_clean"])
        if best and best[1] >= a.min:
            row = dict(sw[best[0]][k]); row.pop("trace", None)
            row["search_seed"] = best[0]; row["robust_clean"] = best[1]
            row["robust_held"] = rb[best[0]][k]["n_held"]
            v = va[best[0]].get(k)
            row["validated"] = bool(v and v["carried_frac"] >= 0.999 and v["ref_end_held"])
            rows.append(row)
    rows.sort(key=lambda r: (-r["robust_clean"], r["seq"]))
    per_clean = [sum(sw[s][k]["end"]["clean"] for k in sw[s]) for s in seeds]
    per_half = [sum(1 for k in rb[s] if rb[s][k]["n_clean"] >= a.min) for s in seeds]
    per_all = [sum(1 for k in rb[s] if rb[s][k]["n_clean"] == 8) for s in seeds]
    per_loose = [sum(sw[s][k]["end"].get("dextrack_obj_loose", False) for k in sw[s]) for s in seeds]
    print(f"seeds {seeds}: clean {per_clean}  >={a.min}/8 {per_half}  8/8 {per_all}  "
          f"dextrack-obj-loose {per_loose}")
    print(f"union >={a.min}/8: {len(rows)}   union 8/8: {sum(r['robust_clean']==8 for r in rows)}   "
          f"union loose-and-robust: "
          f"{sum(r['end'].get('dextrack_obj_loose', False) for r in rows)}")
    Path(a.out).write_text(json.dumps({"prefix": a.prefix, "seeds": seeds, "min": a.min,
                                       "per_seed": {"clean": per_clean, "robust": per_half,
                                                    "all8": per_all, "loose": per_loose},
                                       "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
