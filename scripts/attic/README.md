# attic

Superseded scripts, kept because several of them produced the findings the live
code depends on, and because the failures are documented in NOTES.md by name.
Nothing here should be developed further.

| script | what it was | why it is dead |
|---|---|---|
| `probe_m15.py` | first M1.5 probe: keypoint retarget vs epsilon-max, Vega f5d6 | superseded by `grasp_bench.py`; its config A was order-dependent (retracted in NOTES) |
| `probe_v2.py` | probe with lowered table + force-seated closure | superseded; the seating bisection never converged (50-114 N at a 2 N target) |
| `squeeze_sweep.py` | bimanual grip-margin sweep on Vega | produced the +11.4 cm arms-as-gripper lift; direction abandoned |
| `lift_tune.py` | progressive-squeeze lift tuning | same direction |
| `final_table.py` | one-scene A/B/baseline table | its A row was computed after baselines had moved the robot |
| `expert_baseline.py` | ran the scripted bimanual expert through this harness | **found the spawn defect.** Keep as evidence |
| `spawn_check.py` | do-nothing / random / expert at both spawn heights | **the retraction table.** Keep as evidence |
| `hand_bench.py` | first floating-hand bench, raw URDF compile | invented gains, guessed closure; LEAP scored 0/72 (my bug). `compile_mjcf` still imported by nothing live |
| `leap_bench.py` | fingertips-to-a-point closure | fingers crushed each other (dip_2 vs dip_3 at 44 N) |
| `leap_lift.py` | pedestal lift, raw URDF | pedestal impaled the palm (45 contacts at reset) |
| `leap_pinch.py` | gap-based pinch planning, raw URDF | superseded by the Menagerie model |
| `leap_pick_midair.py` | mid-air pick, raw URDF | superseded |
| `leap_menagerie_pick.py` | first Menagerie attempt | lift ctrlrange clamped to 6 cm |
| `leap_pinch_menagerie.py` | pinch on Menagerie | inverted lift sign |
| `leap_pedestal_pick.py` | pedestal on Menagerie | pedestal through palm again |
| `leap_torque_grasp.py` | constant-torque closure | free gap settles at 7-18 cm, never touches a 4 cm block |
| `leap_final.py` | the first LEAP grasp+lift that passed | superseded by `grasp_bench.py`, which closes its two artifacts |
| `render_*.py` (several) | one-off renders for the above | kept only where the figure is cited in NOTES |
