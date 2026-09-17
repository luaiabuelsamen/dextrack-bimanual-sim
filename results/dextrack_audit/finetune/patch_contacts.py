p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        nb = self._audit_nb0
"""
new="""        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        nb = self._audit_nb0
        if os.environ.get("AUDIT_CONTACTS") and len(self._audit_rows) < 3:
            try:
                cs = self.gym.get_env_rigid_contacts(self.envs[0])
                names_all = self._audit_names + ["object", "goal_object"]
                out = []
                for c in cs[:40]:
                    b0, b1 = int(c["body0"]), int(c["body1"])
                    out.append((names_all[b0] if 0 <= b0 < len(names_all) else b0, names_all[b1] if 0 <= b1 < len(names_all) else b1, round(float(c["initialOverlap"]),4), round(float(c["lambda"]),2)))
                print("AUDIT_CONTACTS step", len(self._audit_rows), "n=", len(cs), out, flush=True)
            except Exception as e:
                print("AUDIT_CONTACTS failed:", repr(e), flush=True)
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("patched")
