"""Probe the LEFT hand too, so a two-hand number is not a one-hand number.

The all-environment evaluation builds one penetration probe over the right
hand's bodies. With a second hand in the scene that reports half the story.
The left hand's bodies are the last block of the rigid-body tensor, because
the bimanual patch appends it as the last actor, and its link names match the
right hand's, so the same probe class serves with the left urdf.

Adds `left_pen_mm_mean` and `left_frac_over_2mm` to the summary and a line to
the printed readout. No effect unless AUDIT_BIMANUAL is set.
"""
import sys
from pathlib import Path

p = str(Path(sys.argv[1] if len(sys.argv) > 1 else
             "/workspace/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py"))
s = open(p).read()
if "_ev_probe_left" in s:
    sys.exit("already patched")

# probe the left hand too, so a two-hand number is not a one-hand number
old = """            self._ev_rows.append(torch.stack([_d * 1000, (_d > 0.002).float(), _g, _e * 100], -1).detach().cpu().numpy())"""
new = """            if getattr(self, "_bi_left_asset", None) is not None:
                # BIMANUAL: the left hand's bodies are the last block of the
                # rigid-body tensor; its link names match the right hand's
                if not hasattr(self, "_ev_probe_left"):
                    self._ev_probe_left = PenetrationProbe(
                        "../assets/allegro_hand_description/urdf/allegro_hand_description_left_fly_v2.urdf",
                        f"../assets/meshdatav3_scaled/sem/{_inst}/coacd", list(self._audit_names), self.device,
                        spacing=float(os.environ.get("AUDIT_PROBE_SPACING", "0")),
                        sdf_res=float(os.environ.get("AUDIT_PROBE_SDF", "0")))
                _dl, _, _ = self._ev_probe_left(self.rigid_body_states[:, -self._bi_left_bodies:], self.object_pose)
            else:
                _dl = torch.zeros_like(_d)
            self._ev_rows.append(torch.stack([_d * 1000, (_d > 0.002).float(), _g, _e * 100,
                                              _dl * 1000, (_dl > 0.002).float()], -1).detach().cpu().numpy())"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''                                "object": str(self.object_code_list[0]), "audit_env": {k: v for k, v in os.environ.items() if k.startswith("AUDIT_")}},'''
new = '''                                "left_pen_mm_mean": float(_per_env[:, 4].mean()), "left_pen_mm_per_env": _per_env[:, 4].round(3).tolist(),
                                "left_frac_over_2mm": float(_per_env[:, 5].mean()),
                                "object": str(self.object_code_list[0]), "audit_env": {k: v for k, v in os.environ.items() if k.startswith("AUDIT_")}},'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''                print("AUDIT_ALLENV spacing=%s envs=%d  pen_mm mean %.2f (per-env %s)  frac>2mm %.2f  grip_x mean %.0f  err_cm mean %.2f  end err per env %s" % ('''
new = '''                print("AUDIT_ALLENV left_pen_mm mean %.2f  left_frac>2mm %.2f" % (_per_env[:, 4].mean(), _per_env[:, 5].mean()), flush=True)
                print("AUDIT_ALLENV spacing=%s envs=%d  pen_mm mean %.2f (per-env %s)  frac>2mm %.2f  grip_x mean %.0f  err_cm mean %.2f  end err per env %s" % ('''
assert s.count(old) == 1
s = s.replace(old, new, 1)

open(p, "w").write(s)
import ast; ast.parse(s); print("left-hand probe added; task parses")
