p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
"""
new="""        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        if __import__("os").environ.get("AUDIT_LOG"):
            print("AUDIT init: object saved_root", self.saved_root_tensor[self.object_indices[0]].detach().cpu().numpy().round(3).tolist(),
                  "object_init_state", self.object_init_state[0].detach().cpu().numpy().round(3).tolist()[:7],
                  "goal_init", self.goal_init_state[0].detach().cpu().numpy().round(3).tolist()[:3] if hasattr(self, "goal_init_state") else None, flush=True)
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("patched")
