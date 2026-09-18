# Retracted and superseded experiments

These produced results that have been withdrawn, or have been replaced by a
corrected implementation. They are kept for provenance, not for use; nothing
under `experiments/` or `src/` imports them.

| script | status |
|---|---|
| `hand_axis.py` | invalid fixture (palm fell 1.05 m during closure) |
| `opposition_axis.py` | superseded by `python -m handsim.hands.axis`, which uses collision-derived fingertips, enforces the mimic couplings and records provenance |
| `retarget_compare.py`, `fig_retarget.py` | geometric retargeting — its poses had fingers 19 mm inside the object |
| `deficit_repair.py`, `fig_deficit.py` | the deficit cohort it conditions on does not exist on the corrected axis |
| `pose_vs_wrench.py` | superseded by `matched.py` and `g1.py`, which match the search budget |
| `chunk_bc.py`, `render_bc.py` | the task is solved by open-loop replay, so these measure no feedback control |
| `floor_poses.py`, `viz_floor_poses.py` | drew poses from the retracted axis; the renderer moved here from `src/handsim/viz/` on 16 Sep, having been unimportable since 8cc4b63 removed the helpers it used |
| `retracted_fig_thesis.py` | hardcoded a withdrawn p-value; refuses to run |
