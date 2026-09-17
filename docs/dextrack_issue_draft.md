# Draft: GitHub issue for Meowuu7/DexTrack

Title: Released GRAB checkpoints do not run as released: the reset restores an
object state captured after the zero-pose hand has already ejected it

Hi, and thanks for releasing the code, data and checkpoints. Running the
released GRAB checkpoints (`s2_cubesmall_inspect`, `s2_duck_inspect`,
`s2_flute_pass`, and `grab_trajs_tracking_ckpt`) headless with
`run_tracking_headless_grab_single_test.sh` on a fresh install (Isaac Gym
Preview 4, Python 3.8, RTX 3090), every rollout loses the object in the
first physics step and `tot_obj_succ_nn` prints 0. I traced it to
initialization, not to the policies, and a two-part fix makes them track.

**What happens**

1. In `_create_envs` the hand actor is created with all joints at zero, so
   the palm collision box sits at the origin, on top of the object, which is
   created at its first-frame pose (for the cube, `(0, 0, 0.025)`).
2. `VecTask.__init__` steps physics before `self.saved_root_tensor =
   self.root_state_tensor.clone()` (task file, around line 1746). During
   that step PhysX resolves the palm-object overlap and launches the object.
   The snapshot therefore stores the object mid-flight: for the cube,
   position `(-0.001, -0.008, 0.144)`, linear velocity about 2.5 m/s,
   angular velocity about 63 rad/s.
3. `reset_idx` restores the object from that snapshot on every reset, so
   every episode starts with the object already flying into the palm; PhysX
   reports about 270 N on it in step one and it is gone by step 30.

Separately, when the object is instead placed exactly at the first-frame
pose at rest, an object resting exactly on the ground plane (cube at
`z = 0.025` with a 0.025 half-extent hull) is ejected at about 9 m/s in the
first step by depenetration. Creating it 3 mm above the plane avoids that.

**Fix that makes the released checkpoints track** (switchable, no change
to the policy, reward or references; diff attached):

- in `reset_idx`, on the initial reset, park the objects 10 m up, take one
  `simulate` so the links reach the written joint state, put the objects
  back at their creation pose at rest;
- overwrite `saved_root_tensor[object_indices]` with `object_init_state`
  and zero velocities;
- create the object 3 mm above its first-frame height.

With those, `s2_cubesmall_inspect` tracks at 0.44 cm mean position error
over 298 frames, `s2_duck_inspect` at 0.26 cm, and the generalist holds 17
of 43 GRAB clips to the end. `tot_obj_succ_nn` still prints 0 for all of
them; is that expected for the released test configuration?

Two questions: were the numbers in Table 1 produced with this reset path,
and is there an intended initialization I am missing (for example a
`set_actor_dof_states` at creation that did not make it into the release)?

Repro: fresh clone, README install, `bash scripts/run_tracking_headless_grab_single_test.sh 0 ori_grab_s2_cubesmall_inspect_1 ./ckpts/s2_cubesmall_inspect_ckpt.pth True`, then read the object position from `ts_to_hand_obj_obs_reset_1.npy` at ts 1: it is at z ≈ 0.2 with the hand 15 cm away.

Diff: `results/dextrack_audit/audit_hook.diff` in
https://github.com/luaiabuelsamen/dextrack-bimanual-sim (the `AUDIT_FIX`
blocks; the `AUDIT_LOG` blocks are instrumentation and can be ignored).
