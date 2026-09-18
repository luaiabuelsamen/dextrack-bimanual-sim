p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        if park:
            self.root_state_tensor[_oi.long(), 2] += 10.0
"""
new="""        if park:
            self.root_state_tensor[_oi.long(), 2] += float(os.environ.get("AUDIT_PARK_DZ", "10.0"))
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("patched")
