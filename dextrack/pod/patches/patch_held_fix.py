"""Fix the `held` count, which reads a frame taken after the episode resets.

The all-environment evaluation summarises `_ev_rows`, one row per control
step, and takes its end-of-clip error from the LAST row. On these clips the
last row is logged after the episode has already reset: progress returns to
zero and the object is back at its start pose, which is where the goal also
sits at frame zero. The error therefore reads 0.00 cm in every environment
and `held` is 16 of 16 unconditionally.

It is not a small error. The camera run scored 16 of 16 while leaving the
object on the ground a metre and a quarter from the goal, and the two-hand
result on the gamecontroller was reported as "still held" when the object was
in fact being dropped.

This takes the end-of-clip error from the last row BEFORE the reset instead,
found by walking back while progress is still increasing, and averages the
last ten such frames so a single frame's contact transient cannot decide it.
`held` then means what it says.

    python patch_held_fix.py [task file]
"""
import sys
from pathlib import Path

T = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/workspace/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py")
s = T.read_text()
if "_ev_last_real" in s:
    sys.exit("already patched")

anchor = """                _a = np.stack(self._ev_rows[1:], 0)      # (T, N, 4)
                _per_env = _per_env = _a.mean(0)                    # per env"""
if anchor not in s:                       # the usual spelling
    anchor = """                _a = np.stack(self._ev_rows[1:], 0)      # (T, N, 4)
                _per_env = _a.mean(0)                    # per env"""
assert s.count(anchor) == 1, s.count(anchor)

add = anchor + """
                # the last row is logged after the episode resets: the object is
                # back at its start pose, which is where the goal sits at frame
                # zero, so its error is 0 for every env and `held` becomes
                # unconditional. Take the end of the clip from before the reset.
                _ev_last_real = _a.shape[0] - 1
                while _ev_last_real > 0 and float(_a[_ev_last_real, :, 3].mean()) < 1e-3 \\
                        and float(_a[_ev_last_real - 1, :, 3].mean()) > 1e-3:
                    _ev_last_real -= 1
                _ev_end = _a[max(0, _ev_last_real - 9):_ev_last_real + 1, :, 3].mean(0)"""
s = s.replace(anchor, add, 1)

# and use it wherever the end of the clip is reported
for old, new in (
    ('"end_err_cm_per_env": _a[-1, :, 3].round(2).tolist(),', '"end_err_cm_per_env": _ev_end.round(2).tolist(),'),
    ('"held": int((_a[-1, :, 3] < 10.0).sum()),', '"held": int((_ev_end < 10.0).sum()),'),
    ('np.round(_a[-1, :, 3], 1).tolist()', 'np.round(_ev_end, 1).tolist()'),
):
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new, 1)

T.write_text(s)
import ast
ast.parse(s)
print(f"patched {T}: held now measured before the reset")
