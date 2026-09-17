p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        if __import__("os").environ.get("AUDIT_LOG"):
            if not hasattr(self, "_audit_rb_pre_t"):
"""
new="""        if __import__("os").environ.get("AUDIT_FIX") == "2" and not getattr(self, "_audit_settled", False):
            # AUDIT FIX 2: the joint state written above reaches the link poses
            # only during the next physics step, and that step collides the
            # stale zero pose of the hand with the object. Park every object
            # 10 m up, take one physics step so the links move to the written
            # joint state, then put the objects back at rest. Initial reset only.
            self._audit_settled = True
            self.gym.refresh_actor_root_state_tensor(self.sim)
            _oi = self.object_indices.to(torch.int32)
            _saved = self.root_state_tensor[_oi.long()].clone()
            self.root_state_tensor[_oi.long(), 2] += 10.0
            self.root_state_tensor[_oi.long(), 7:13] = 0.0
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(_oi), len(_oi))
            self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True)
            self.root_state_tensor[_oi.long()] = _saved
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(_oi), len(_oi))
            self.shadow_hand_dof_pos[env_ids, :] = self.shadow_hand_default_dof_pos[env_ids, :]
            self.shadow_hand_dof_state[env_ids, :, 1] = 0.0
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
            print("AUDIT_FIX 2: settled links with objects parked", flush=True)
        if __import__("os").environ.get("AUDIT_LOG"):
            if not hasattr(self, "_audit_rb_pre_t"):
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("patched")
