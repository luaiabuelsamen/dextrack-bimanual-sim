p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
"""
new="""        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        if __import__("os").environ.get("AUDIT_FIX") in ("2", "3"):
            # AUDIT FIX 3: the snapshot above was taken after the object had
            # already been ejected by the zero-pose hand (velocities of metres
            # and tens of radians per second at init). Restore resets from the
            # creation pose at rest instead.
            self.saved_root_tensor[self.object_indices, :7] = self.object_init_state[:, :7]
            self.saved_root_tensor[self.object_indices, 7:13] = 0.0
            print("AUDIT_FIX 3: reset snapshot replaced with creation pose at rest", flush=True)
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("patched")
