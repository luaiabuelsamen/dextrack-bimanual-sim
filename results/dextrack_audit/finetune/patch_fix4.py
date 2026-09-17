p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
# settle applies for FIX 2 and 3
old='if __import__("os").environ.get("AUDIT_FIX") == "2" and not getattr(self, "_audit_settled", False):'
new='if __import__("os").environ.get("AUDIT_FIX") in ("2", "3") and not getattr(self, "_audit_settled", False):'
assert s.count(old)==1; s=s.replace(old,new,1)
# pin the object at its creation pose, at rest, right before the first physics step
anchor="    def _audit_pre(self):\n"
assert s.count(anchor)==1
pin = """    def _audit_pin_object(self):
        # AUDIT FIX 3b: on the first control step, push the creation pose at
        # rest to PhysX for every object, regardless of what the reset wrote.
        import os
        if os.environ.get("AUDIT_FIX") != "3" or getattr(self, "_audit_pinned", False):
            return
        self._audit_pinned = True
        _oi = self.object_indices.to(torch.int32)
        self.root_state_tensor[_oi.long(), :7] = self.object_init_state[:, :7]
        self.root_state_tensor[_oi.long(), 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(_oi), len(_oi))
        print("AUDIT_FIX 3b: object pinned at creation pose before the first step", flush=True)

"""
s = s.replace(anchor, pin + anchor, 1)
# call it at the very end of pre_physics_step: after the LAST set_dof_position_target_tensor_indexed call in the file
k = s.rindex("        self.gym.set_dof_position_target_tensor_indexed(self.sim,")
depth=0; j=k
while True:
    c=s[j]
    if c=="(": depth+=1
    elif c==")":
        depth-=1
        if depth==0: break
    j+=1
end = s.index("\n", j)+1
s = s[:end] + "        self._audit_pin_object()\n" + s[end:]
open(p,"w").write(s); print("patched")
