p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""            self.root_state_tensor[_oi.long()] = _saved
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(_oi), len(_oi))
"""
new="""            if __import__("os").environ.get("AUDIT_PARK") != "1":
                self.root_state_tensor[_oi.long()] = _saved
                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(_oi), len(_oi))
"""
assert s.count(old)==1; s=s.replace(old,new,1)
# log per-shape collision geometry of the hand once, and the object's shape count
old2="""            self._audit_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)
"""
new2="""            self._audit_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)
            try:
                _props = self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)
                _bsm = self.gym.get_actor_rigid_body_shape_indices(self.envs[0], 0)
                print("AUDIT hand shapes:", len(_props), "per body:", [(self._audit_names[b], _bsm[b].start, _bsm[b].count) for b in range(len(_bsm)) if _bsm[b].count > 0], flush=True)
                _oprops = self.gym.get_actor_rigid_shape_properties(self.envs[0], 1)
                print("AUDIT object shapes:", len(_oprops), "actor names:", [self.gym.get_actor_name(self.envs[0], a) for a in range(self.gym.get_actor_count(self.envs[0]))], flush=True)
            except Exception as e:
                print("AUDIT shape query failed:", e, flush=True)
"""
assert s.count(old2)==1; s=s.replace(old2,new2,1); open(p,"w").write(s); print("patched")
