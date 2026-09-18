# Retracted and superseded results

Kept so every withdrawn number can be traced to the run that produced it. None
of these should be quoted. `NOTES.md` records what was wrong with each.

| file(s) | why |
|---|---|
| `hand_axis*.json` | invalid fixture — the palm fell 1.05 m during closure with the arm joints unactuated |
| `deficit_*.json`, `conf_*.json` | conditioned on a "deficit cohort" defined by an opposition floor measured at distal joint origins; the cohort does not exist on the corrected axis |
| `pvw*.json` | superseded by the budget-matched comparison; their wrench arm had one free scalar against the pose arm's sixteen |
| `chunk_bc*.json`, `bc.json` | the task they were run on is solved by open-loop replay (8/8 at 13.49 cm), so they say nothing about learned control |
| `mppi_*.json` | superseded; kept for the reward-hacking history in NOTES |
| `opposition_axis_corrected.json` | superseded by `results/opposition_axis.json`, which is produced by the public command with provenance |
