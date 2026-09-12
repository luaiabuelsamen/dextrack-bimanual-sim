# retracted / superseded results

Kept because NOTES.md cites several of them by name, and because the retractions
are part of the record. **Nothing in this directory may be quoted.**

| file | why it is here |
|---|---|
| `final_table.json` | its config-A row was ORDER-DEPENDENT. `retarget_keypoints` seeded its optimiser from the robot's current pose, so running the baselines first changed the answer: clean resets gave ncon=2 / kp=13.21 cm three times over, while after a random-walk baseline the same call gave ncon=0 / kp=13.60 and ncon=5 / kp=15.76. eps=0 and "never moves the box" survive every ordering; the contact count and exact keypoint distance do not. Fixed in `probe_m15.retarget_keypoints` by resetting first, but the table was never re-run because the direction it served (Vega f5d6 bimanual) was abandoned. |
| `expert_baseline.json`, `squeeze_sweep.json`, `lift_tune.json`, `best_trace.json` | the Vega f5d6 arms-as-gripper line. The lift they measure is real but it is two arms wedging a box, not manipulation. Superseded by the two-LEAP-hand environment. |
| `probe_*.json` | M1.5 probe iterations, superseded by `grasp_bench.py`. |
| `leap_*.json`, `hand_bench.json` | single-hand grasp attempts before the Menagerie model and the calibrated closure. `hand_bench.json`'s LEAP 0/72 is a bug in my closure, not a property of LEAP, and is recorded as such. |

Live results are in the parent directory.
