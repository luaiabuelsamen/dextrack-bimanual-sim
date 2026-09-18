"""How far inside the object is a DexTrack reference, before anything is simulated?

Every result this project has on their policies measures what a trained
policy does to the object. This measures what they asked it to do. The
reference is a kinematic hand trajectory and an object trajectory; placing
the hand by forward kinematics on their own URDF and running the same probe
the training runs use gives the interpenetration built into the target.

It matters for two reasons. A policy rewarded for tracking a reference that
lies 10 mm inside the object is being asked to interpenetrate, and no reward
term applied downstream can fully undo that. And picking a clip for two-hand
work needs a reference that is not already a burial, which is the gate this
supplies.

    python dextrack/reference_penetration.py out/refs/*.npy --obj-root out/dextrack_assets/obj
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def measure(ref_path: Path, obj_dir: Path, spacing: float = 0.002, stride: int = 2) -> dict:
    """Per-frame penetration of a reference's hand or hands into its object."""
    import torch
    from dextrack.render_reference import Kinematics, HANDS
    from dextrack.penetration_torch import PenetrationProbe

    ref = np.load(ref_path, allow_pickle=True).item()
    obj_pos = np.asarray(ref["object_transl"], float)
    obj_quat = np.asarray(ref["object_rot_quat"], float)
    hands = {"right": np.asarray(ref["robot_delta_states_weights_np"], float)}
    if "robot_delta_states_weights_np_left" in ref:
        hands["left"] = np.asarray(ref["robot_delta_states_weights_np_left"], float)

    out = {"clip": ref_path.stem, "frames": 0, "hands": {}}
    for side, q in hands.items():
        fk = Kinematics(HANDS[side])
        names = [n for n in fk.links if n not in (None, "world")]
        probe = PenetrationProbe(str(HANDS[side]), str(obj_dir), names, "cpu", spacing=spacing)
        ks = range(0, len(q), stride)
        rb = torch.zeros(len(ks), len(names), 13)
        op = torch.zeros(len(ks), 7)
        for i, k in enumerate(ks):
            poses = fk(q[k])
            for j, n in enumerate(names):
                p, quat = poses[n]
                rb[i, j, :3] = torch.tensor(p)
                rb[i, j, 3:7] = torch.tensor([quat[1], quat[2], quat[3], quat[0]])
            op[i] = torch.tensor(np.concatenate([obj_pos[k], obj_quat[k]]), dtype=torch.float32)
        depth, n_touch, _ = probe(rb, op)
        d = depth.numpy() * 1000
        out["frames"] = len(ks)
        out["hands"][side] = {
            "pen_mm_mean": float(d.mean()), "pen_mm_median": float(np.median(d)),
            "pen_mm_max": float(d.max()),
            "frac_over_2mm": float((d > 2).mean()), "frac_over_5mm": float((d > 5).mean()),
            "frac_over_10mm": float((d > 10).mean()),
            "frac_touching": float((n_touch.numpy() > 0).mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+", type=Path)
    ap.add_argument("--obj-root", type=Path, default=Path("out/dextrack_assets/obj"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = []
    print(f"{'clip':34s} {'hand':6s} {'mean':>7s} {'median':>7s} {'max':>7s} {'>2mm':>6s} {'>5mm':>6s} {'>10mm':>6s} {'touch':>6s}")
    for p in a.refs:
        inst = p.stem.replace("passive_active_info_", "").replace("_nf_300", "")
        obj = a.obj_root / inst
        if not (obj / "decomposed.obj").exists():
            print(f"{inst:34s} no object decomposition at {obj}")
            continue
        r = measure(p, obj, stride=a.stride)
        rows.append(r | {"inst": inst})
        for side, h in r["hands"].items():
            print(f"{inst:34s} {side:6s} {h['pen_mm_mean']:6.2f}  {h['pen_mm_median']:6.2f}  {h['pen_mm_max']:6.2f} "
                  f"{h['frac_over_2mm']*100:5.0f}% {h['frac_over_5mm']*100:5.0f}% {h['frac_over_10mm']*100:5.0f}% {h['frac_touching']*100:5.0f}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
