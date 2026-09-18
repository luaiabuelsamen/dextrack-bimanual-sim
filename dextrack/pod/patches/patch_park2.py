p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        if os.environ.get("AUDIT_FIX") != "3" or getattr(self, "_audit_pinned", False):
            return
        self._audit_pinned = True
        _oi = self.object_indices.to(torch.int32)
        self.root_state_tensor[_oi.long(), :7] = self.object_init_state[:, :7]
        self.root_state_tensor[_oi.long(), 7:13] = 0.0
"""
new="""        if os.environ.get("AUDIT_FIX") != "3":
            return
        self._audit_pin_n = getattr(self, "_audit_pin_n", 0) + 1
        park = os.environ.get("AUDIT_PARK") == "1" and self._audit_pin_n <= 3
        if self._audit_pin_n > 1 and not park:
            return
        _oi = self.object_indices.to(torch.int32)
        self.root_state_tensor[_oi.long(), :7] = self.object_init_state[:, :7]
        if park:
            self.root_state_tensor[_oi.long(), 2] += 10.0
        self.root_state_tensor[_oi.long(), 7:13] = 0.0
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("patched")
