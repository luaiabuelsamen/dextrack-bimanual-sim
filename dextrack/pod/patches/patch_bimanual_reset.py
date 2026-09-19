"""Place the left hand during THEIR reset, not from the physics step.

The bimanual patch first put the left hand at its reference's opening pose by
calling `set_dof_state_tensor_indexed` itself, from the same place it writes
the hand's position targets, guarded to fire only on the first step or two of
an episode. That works at 4 and at 256 environments and corrupts Isaac Gym at
1024: the next kernel to touch any tensor dies with an illegal memory access,
reported against whatever line happens to run next, which was an integer
increment three functions away.

Bisected by disabling the call, at which point 1024 environments ran to
completion, so the mid-step write is the cause and not the contact reward it
was first blamed on.

The fix is to do it where the simulator already expects a state write: inside
`reset_idx`, on the environments being reset, writing into the same
`dof_state` buffer their own reset fills and adding the left hand's actor
indices to the single `set_dof_state_tensor_indexed` call they already make.
One call, at reset, through their path. The left hand's position targets are
seeded there too, so the first step after a reset does not swing it.

    python patch_bimanual_reset.py [task file]
"""
import sys
from pathlib import Path

T = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/workspace/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py")
s = T.read_text()
if "_bi_reset_place" in s:
    sys.exit("already patched")

# 1. place the left hand as part of their reset, and include it in their call
anchor = """        hand_indices = self.hand_indices[env_ids].to(torch.int32) # hand indices #
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))"""
add = """        hand_indices = self.hand_indices[env_ids].to(torch.int32) # hand indices #
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        if getattr(self, "_bi_left_asset", None) is not None:
            # BIMANUAL: the left hand belongs to this reset. Writing its state
            # from the physics step instead, with a second indexed call, faults
            # the simulator at 1024 environments.
            self._bi_reset_place = True
            self._bi_load_ref()
            _q0 = self._bi_left_ref[0]
            _dof = self.dof_state.view(self.num_envs, -1, 2)
            _dof[env_ids, self._bi_left_slice, 0] = _q0
            _dof[env_ids, self._bi_left_slice, 1] = 0.0
            self.prev_targets[env_ids, self._bi_left_slice] = _q0
            self.cur_targets[env_ids, self._bi_left_slice] = _q0
            all_hand_indices = torch.unique(torch.cat(
                [all_hand_indices, self._bi_left_indices[env_ids].to(torch.int32)]))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))"""
assert s.count(anchor) == 1, s.count(anchor)
s = s.replace(anchor, add, 1)

# 2. the driver keeps the position targets and gives up the state write
old = """        fresh = (self.progress_buf <= 1) & (~self._bi_reset_done)
        if _os.environ.get("AUDIT_BI_TELEPORT", "1") != "1":
            fresh = torch.zeros_like(fresh)
        if fresh.any():
            dof = self.dof_state.view(self.num_envs, -1, 2)
            dof[fresh, self._bi_left_slice, 0] = q[fresh]
            dof[fresh, self._bi_left_slice, 1] = 0.0
            idx = self._bi_left_indices[fresh].to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(idx), len(idx))
            self._bi_reset_done |= fresh
        self._bi_reset_done &= (self.progress_buf > 1)
"""
if old not in s:                       # the pre-guard spelling
    old = """        fresh = (self.progress_buf <= 1) & (~self._bi_reset_done)
        if fresh.any():
            dof = self.dof_state.view(self.num_envs, -1, 2)
            dof[fresh, self._bi_left_slice, 0] = q[fresh]
            dof[fresh, self._bi_left_slice, 1] = 0.0
            idx = self._bi_left_indices[fresh].to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(idx), len(idx))
            self._bi_reset_done |= fresh
        self._bi_reset_done &= (self.progress_buf > 1)
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "", 1)

import ast
ast.parse(s)          # never write a file that does not parse
T.write_text(s)
print(f"patched {T}: the left hand is placed by reset_idx")
