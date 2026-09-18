# What a pod needs, and where it comes from

Five archives, all from third parties, none redistributed here. Put them in
`/workspace/inputs/` and run `setup_pod.sh`; it verifies them against
`data.sha256` before extracting.

| archive | size | source | extracts to | what it is |
|---|---:|---|---|---|
| `IsaacGym_Preview_4_Package.tar` | 201 MB | [NVIDIA](https://developer.nvidia.com/isaac-gym) (login) | `/workspace/isaacgym_pkg/isaacgym/` | the simulator; `python/` is pip-installed editable |
| `data.zip` | 1.64 GB | DexTrack README, "retargeted data" (OneDrive) | `DexTrack/isaacgymenvs/data/`, then its inner zips extracted in place | per-clip references: `GRAB_Tracking_PK_reduced_300/data/passive_active_info_<clip>_nf_300.npy` (object pose + a 300 × 22 right-Allegro trajectory); LEAP and TACO variants unused |
| `objs.zip` | 1.65 GB | DexTrack README, object files part 2 | `DexTrack/assets/rsc/objs/` | object URDFs and meshes |
| `meshdatav3_scaled.zip` | 1.03 GB | DexTrack README, object files part 3 | `DexTrack/assets/meshdatav3_scaled/sem/<inst>/coacd/decomposed.obj` | the convex decompositions their simulator collides with, 4,162 instances; the probe reads these |
| `ckpts.zip` | 4.35 GB | DexTrack README, checkpoints | `DexTrack/isaacgymenvs/ckpts/` | 14 released checkpoints: `s2_cubesmall_inspect_ckpt.pth`, `s2_duck_inspect_ckpt.pth`, `s2_flute_pass_ckpt.pth`, `grab_trajs_tracking_ckpt.pth` (the generalist), and others |

Not provided by the release and stubbed: `assets/datasetv4.1/` (their code
reads a `qpos` and a `scale` per instance from it; the values are unused
and the scale is 1.0 for every GRAB instance; `setup_pod.sh` writes the
stubs). GRAB itself is not needed on the pod; DexTrack's references are
already retargeted.

Our own artifacts (fine-tuned checkpoints, logs, audits, renders) live in
the private Hub dataset `luaia/dextrack-bimanual-runs`, one folder per run.
