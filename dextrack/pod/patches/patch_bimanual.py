"""Add a second, left hand to DexTrack's task, driven by a two-hand reference.

Their task builds one hand, one object and a goal marker per environment. This
appends a left hand as the LAST actor, which is the whole trick: actor order
fixes the layout of the degree-of-freedom and rigid-body tensors, so putting
the left hand last leaves every existing slice in their 13,000-line task still
addressing the right hand. Nothing they wrote has to change.

The left hand is position-driven to the reference's left trajectory each step
while the policy keeps the right hand. That is a real two-handed scene in
physics, with both hands and the object colliding, and it is reachable from
their released checkpoints, which a doubled action space would not be.

Gated on AUDIT_BIMANUAL, the path to a reference built by
`dextrack/bimanual_reference.py`. Unset, the task is exactly theirs.

    AUDIT_BIMANUAL=/workspace/refs/bimanual_camera_takepicture_2.npy
    AUDIT_LEFT_URDF=allegro_hand_description/urdf/allegro_hand_description_left_fly_v2.urdf

Run once against the task file; it asserts on every anchor, so a second run
or a changed file fails loudly rather than corrupting the task.
"""
import sys
from pathlib import Path

T = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/workspace/DexTrack/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py")
s = T.read_text()
if "AUDIT_BIMANUAL" in s:
    sys.exit("already patched")

# 1. load the left asset alongside the right, with identical options
anchor = "        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)"
add = anchor + '''

        # BIMANUAL: a second hand, loaded with the right hand's own asset
        # options so the two differ only in geometry. Built by
        # dextrack/left_hand.py; their own left urdf has no base.
        self._bi_path = _os.environ.get("AUDIT_BIMANUAL")
        self._bi_left_asset = None
        if self._bi_path:
            _left_file = _os.environ.get(
                "AUDIT_LEFT_URDF",
                "allegro_hand_description/urdf/allegro_hand_description_left_fly_v2.urdf")
            self._bi_left_asset = self.gym.load_asset(self.sim, asset_root, _left_file, asset_options)
            self._bi_left_bodies = self.gym.get_asset_rigid_body_count(self._bi_left_asset)
            self._bi_left_shapes = self.gym.get_asset_rigid_shape_count(self._bi_left_asset)
            self._bi_left_dofs = self.gym.get_asset_dof_count(self._bi_left_asset)
            print(f"BIMANUAL: left hand {_left_file}: {self._bi_left_bodies} bodies, "
                  f"{self._bi_left_dofs} dofs, reference {self._bi_path}", flush=True)'''
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# 2. the aggregate has to have room for it
anchor = """        max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * maxx_num_obj_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * maxx_num_obj_shapes + 1"""
add = anchor + """
        if self._bi_left_asset is not None:                      # BIMANUAL
            max_agg_bodies += self._bi_left_bodies
            max_agg_shapes += self._bi_left_shapes"""
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# 3. create it last, inside the aggregate
anchor = """            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)"""
add = """            # BIMANUAL: last actor, so the dof and rigid-body tensors keep
            # the right hand at the indices the rest of this file assumes
            if self._bi_left_asset is not None:
                left_actor = self.gym.create_actor(env_ptr, self._bi_left_asset,
                                                   shadow_hand_start_pose, "hand_left", i, -1, 0)
                self.gym.set_actor_dof_properties(env_ptr, left_actor, shadow_hand_dof_props)
                if not hasattr(self, "_bi_left_handles"):
                    self._bi_left_handles, self._bi_left_indices = [], []
                self._bi_left_handles.append(left_actor)
                self._bi_left_indices.append(self.gym.get_actor_index(env_ptr, left_actor, gymapi.DOMAIN_SIM))
                self._bi_left_dof_start = self.gym.get_actor_dof_index(env_ptr, left_actor, 0, gymapi.DOMAIN_ENV)

""" + anchor
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# 4. once the envs exist: the index tensor and the reference
anchor = "        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)"
add = anchor + '''
        if getattr(self, "_bi_left_asset", None) is not None:     # BIMANUAL
            self._bi_left_indices = to_torch(self._bi_left_indices, dtype=torch.long, device=self.device)
            _ref = _np.load(self._bi_path, allow_pickle=True)          # .npz here: no
            _ref = _ref if hasattr(_ref, "files") else _ref.item()      # pickle across numpy versions
            _q = _ref["robot_delta_states_weights_np_left"]
            self._bi_left_ref = torch.tensor(_np.asarray(_q, dtype=_np.float32), device=self.device)
            self._bi_left_slice = slice(self._bi_left_dof_start, self._bi_left_dof_start + self._bi_left_dofs)
            self._bi_reset_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            print(f"BIMANUAL: left reference {tuple(self._bi_left_ref.shape)}, "
                  f"env dofs {self._bi_left_slice.start}:{self._bi_left_slice.stop}", flush=True)'''
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# 5. drive it, and teleport it into place on the first step of an episode
anchor = """        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))"""
add = """        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        if getattr(self, "_bi_left_asset", None) is not None:      # BIMANUAL
            self._bi_drive()
            all_hand_indices = torch.unique(
                torch.cat([self.hand_indices, self._bi_left_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))"""
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# 6. the driver itself, next to the audit helpers
anchor = "    def post_physics_step(self):\n        self._audit_log()"
add = '''    def _bi_drive(self):
        """Point the left hand at its reference for each environment's own frame.

        Position targets, the same drive the right hand is under, so the left
        hand pushes on the object rather than passing through it. On the first
        step after a reset the joints are also written directly: a hand that
        started at the zero pose would otherwise be flung at the object by the
        first target, which is the failure their own initialization has.
        """
        ts = torch.clamp(self.progress_buf, 0, self._bi_left_ref.shape[0] - 1)
        q = self._bi_left_ref[ts]                                   # (envs, 22)
        self.prev_targets[:, self._bi_left_slice] = q

        fresh = (self.progress_buf <= 1) & (~self._bi_reset_done)
        if fresh.any():
            dof = self.dof_state.view(self.num_envs, -1, 2)
            dof[fresh, self._bi_left_slice, 0] = q[fresh]
            dof[fresh, self._bi_left_slice, 1] = 0.0
            idx = self._bi_left_indices[fresh].to(torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(idx), len(idx))
            self._bi_reset_done |= fresh
        self._bi_reset_done &= (self.progress_buf > 1)

''' + anchor
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# 7. their default-pose write assumes the only dofs in the sim are the hand's
anchor = "        self.dof_state[:, 0] = self.shadow_hand_default_dof_pos.view(-1).contiguous()"
add = """        if getattr(self, "_bi_left_asset", None) is not None:      # BIMANUAL
            # the sim now holds both hands' dofs; seed only the right hand's
            self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs, 0] = \\
                self.shadow_hand_default_dof_pos
        else:
            self.dof_state[:, 0] = self.shadow_hand_default_dof_pos.view(-1).contiguous()"""
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)

# the helpers the added code needs, imported under names the file does not use
anchor = "import numpy as np\n"
assert anchor in s
s = s.replace(anchor, anchor + "import numpy as _np\nimport os as _os\n", 1)

T.write_text(s)
print(f"patched {T}")
