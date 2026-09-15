"""Cache a convex decomposition for every GRAB object. Run once."""
import json, time
from pathlib import Path
from oppdef import paths
from oppdef.human import decompose as D

objs = sorted(p.stem for p in
              (paths.GRAB/"tools/object_meshes/contact_meshes").glob("*.ply"))
rows = []
for i, o in enumerate(objs):
    t0 = time.time()
    try:
        r = D.report(o); r["secs"] = round(time.time()-t0, 1)
    except Exception as e:
        r = {"object": o, "error": repr(e)}
    rows.append(r)
    print(f"[{i+1}/{len(objs)}] {r}", flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/grab_convex.json").write_text(json.dumps(rows, indent=1))
print("done")
