"""A contact term in the reward, so the policy cannot succeed by not touching.

Why this exists. Fine-tuning the right-hand policy inside the two-hand scene
improved every number by withdrawing: its hand touched the object on 23 of
299 frames against the generalist's 152, and the position-driven left hand
carried the object alone. An objective that only asks where the object ends
up is satisfied by a hand that does nothing, whenever something else in the
scene will do the work.

This is a known failure with a known fix. PhysHOI multiplies a contact-graph
term into its imitation reward, and the survey's note on it states plainly
that the term "exists to stop the policy learning not to touch the object".
ManipTrans rewards each fingertip separately with an exponential in its
distance to its target, `exp(-100 d)` on the thumb and `exp(-90 d)` on the
index in its released code. Both shape contact directly rather than hoping
it falls out of an object-pose reward.

The term here follows that form, on a distance that is defined everywhere.
The obvious choice, the probe's own signed depth, is not: outside its
bounding-sphere cull the probe returns a sentinel, so a hand that has
withdrawn 10 cm scores exactly zero with zero gradient and can never be
pulled back. Checked offline on the two logged rollouts before any of this
ran, which is the only reason it was caught. The term uses the distance from
each fingertip to the object's bounding sphere instead,

    d_i       = max(0, ||p_tip_i - p_object|| - r_object)
    r_contact = mean over fingertips of exp(-BETA * d_i)

which is 1 anywhere on or inside that sphere and decays smoothly over a few
centimetres outside it, with gradient at any distance. It is added with
weight `AUDIT_CONTACT_W`. It rewards approach and contact without rewarding
pressing, since a fingertip already inside scores the same as one on the
surface; pressing is what the penetration term is for, and the two are meant
to be used together.

    AUDIT_CONTACT_W=0.5 AUDIT_CONTACT_BETA=20 AUDIT_PEN_W=...

Unset, the reward is unchanged. Requires the penetration probe, which the
reward hook already builds.
"""
import sys
from pathlib import Path

T = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/workspace/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py")
s = T.read_text()
if "AUDIT_CONTACT_W" in s:
    sys.exit("already patched")

# the probe must run whenever a contact term is asked for, not only for depth
old = '        if _os.environ.get("AUDIT_PEN_W") or _os.environ.get("AUDIT_FORCE_W"):'
new = '        if _os.environ.get("AUDIT_PEN_W") or _os.environ.get("AUDIT_FORCE_W") or _os.environ.get("AUDIT_CONTACT_W"):'
assert s.count(old) == 1
s = s.replace(old, new, 1)

# fingertip indices within the probe's link list, found once by name
old = '''                self._pen_stats = []'''
new = '''                self._pen_stats = []
                # the four fingertip links, as indices into the probe's own link
                # list, for the contact term below
                _tips = ["link_3", "link_7", "link_11", "link_15"]
                self._pen_tip_body = [_i for _i, _n in enumerate(_names) if _n in _tips]
                # the object's bounding-sphere radius, from the same hulls the
                # probe uses, so "distance outside the object" is defined at any
                # separation rather than only within the probe's cull
                import trimesh as _tm
                _m = _tm.load(f"../assets/meshdatav3_scaled/sem/{_inst}/coacd/decomposed.obj",
                              force="mesh", process=False)
                _v = _np.asarray(_m.vertices)
                _c = 0.5 * (_v.min(0) + _v.max(0))
                self._obj_radius = float(_np.linalg.norm(_v - _c, axis=1).max())
                self._contact_w = float(_os.environ.get("AUDIT_CONTACT_W", "0"))
                self._contact_beta = float(_os.environ.get("AUDIT_CONTACT_BETA", "20"))'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# the term itself, alongside the penetration term
old = '''            _term = self._pen_w * _depth + self._force_w * torch.clamp((_grip_x - 40.0) / 40.0, min=0.0)'''
new = '''            _term = self._pen_w * _depth + self._force_w * torch.clamp((_grip_x - 40.0) / 40.0, min=0.0)
            if self._contact_w > 0 and self._pen_tip_body:
                # PhysHOI's contact-graph term in ManipTrans's exponential form,
                # on distance to the object's bounding sphere so it has gradient
                # however far the hand has withdrawn
                _tp = _rb[:, self._pen_tip_body, :3]
                _dist = torch.clamp(torch.norm(_tp - self.object_pose[:, None, :3], dim=-1)
                                    - self._obj_radius, min=0.0)
                _contact = torch.exp(-self._contact_beta * _dist).mean(-1)
                _term = _term - self._contact_w * _contact          # a reward, not a cost
                self._contact_last = float(_contact.mean())'''
assert s.count(old) == 1
s = s.replace(old, new, 1)

# report it beside the depth statistic
old = '''                print(f"PEN-REWARD last200 (tracked envs): depth mean {a[:,0].mean():.2f} mm, frac>2mm {a[:,1].mean():.2f}, grip_x mean {a[:,2].mean():.0f}, term mean {a[:,3].mean():.3f}, tracked {a[:,4].mean():.2f}", flush=True)'''
new = """                _c_last = getattr(self, '_contact_last', float('nan'))
                print(f"PEN-REWARD last200 (tracked envs): depth mean {a[:,0].mean():.2f} mm, frac>2mm {a[:,1].mean():.2f}, grip_x mean {a[:,2].mean():.0f}, term mean {a[:,3].mean():.3f}, tracked {a[:,4].mean():.2f}, contact {_c_last:.3f}", flush=True)"""
assert s.count(old) == 1
s = s.replace(old, new, 1)

import ast
ast.parse(s)          # never write a file that does not parse
T.write_text(s)
print(f"patched {T}: AUDIT_CONTACT_W adds a fingertip contact reward")
