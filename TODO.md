# Outstanding work, and what "done" means for each

Fixed before starting, so progress can be checked rather than claimed.

| # | task | done when | status |
|---|---|---|---|
| A | Success criterion that is not inside backend noise | criterion fixed IN ADVANCE, expert passes and control fails on BOTH CPU and warp | **DONE** — criterion v2 (base LIFT). CPU True/False, warp True/False. |
| B | M2 second dimension: what each hand can HOLD | grasp bench across 4 hands, one shared object set | **DONE** — shadow 7, leap 5, allegro 3, f5d6 0; monotone in the opposition floor |
| C | BC from the scripted expert | trained policy vs do-nothing/random/expert, 3 seeds, spread | **DONE** — 67%+/-47% vs 0%/0%/100%; phase input was essential |
| D | RL residual on the BC policy | beats BC and all trivial baselines, 3 seeds, or is reported as not beating them | |
| E | Retract or re-run the superseded Vega tables | final_table.py's A row was order-dependent; either re-run or mark retracted | |

Rules carried in: trivial baselines every time (do-nothing, random, scripted
expert); three seeds with spread; criteria fixed before the run; a 0/N is a
defect until paired with a working reference; render a frame before sweeping.
No Modal -- everything validated locally on the Orin.
