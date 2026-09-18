p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
L = s.index("    def compute_reward(self, actions, id=-1):")
# insert right after the jitted reward assignment inside compute_reward: the
# statement that starts with "self.rew_buf[:], self.reset_buf[:]" and ends at
# its matching close paren
i = s.index("        self.rew_buf[:], self.reset_buf[:]", L)
k = s.index("(", i); depth = 0
while True:
    c = s[k]
    if c == "(": depth += 1
    elif c == ")":
        depth -= 1
        if depth == 0: break
    k += 1
j = s.index("\n", k) + 1
hook = """        # PENETRATION-AWARE TERM (env-var gated; off by default). depth = deepest
        # sampled hand-surface point inside the object's convex parts, metres;
        # grip = PhysX net contact force summed over links geometrically touching
        # the object, in multiples of the object's weight. Weights:
        #   AUDIT_PEN_W   reward per metre of depth (100 -> 0.1 per mm)
        #   AUDIT_FORCE_W reward per unit of (grip_x - 40)/40 above the cap
        import os as _os
        if _os.environ.get("AUDIT_PEN_W") or _os.environ.get("AUDIT_FORCE_W"):
            if not hasattr(self, "_pen_probe"):
                import sys as _sys; _sys.path.insert(0, "/workspace")
                from penetration_torch import PenetrationProbe
                _names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)
                _inst = self.object_code_list[0].split("_nf_")[0]
                self._pen_probe = PenetrationProbe(
                    "../assets/allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf",
                    f"../assets/meshdatav3_scaled/sem/{_inst}/coacd", _names, self.device)
                self._pen_nb0 = self.gym.get_env_rigid_body_count(self.envs[0])
                self._pen_nhand = len(_names)
                self._pen_cf = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
                self._pen_weight = 500.0 * self._pen_probe.volume * 9.81
                self._pen_w = float(_os.environ.get("AUDIT_PEN_W", "0")); self._force_w = float(_os.environ.get("AUDIT_FORCE_W", "0"))
                self._pen_stats = []
                print(f"PEN-REWARD on: weight {self._pen_weight:.2f} N, hulls {self._pen_probe.planes.shape[0]}, w_pen {self._pen_w}, w_force {self._force_w}", flush=True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            _rb = self.rigid_body_states[:, :self._pen_nhand]
            _depth, _ntouch, _per_link = self._pen_probe(_rb, self.object_pose)
            _cf = self._pen_cf.view(self.num_envs, -1, 3)[:, :self._pen_nhand][:, self._pen_probe.link_idx]
            _touch = (_per_link > -self._pen_probe.touch).float()
            _grip_x = (torch.norm(_cf, dim=-1) * _touch).sum(-1) / self._pen_weight
            _term = self._pen_w * _depth + self._force_w * torch.clamp((_grip_x - 40.0) / 40.0, min=0.0)
            self.rew_buf[:] = self.rew_buf - _term
            self._pen_stats.append((float(_depth.mean()) * 1000, float((_depth > 0.002).float().mean()), float(_grip_x.mean()), float(_term.mean())))
            if len(self._pen_stats) % 200 == 0:
                a = np.array(self._pen_stats[-200:])
                print(f"PEN-REWARD last200: depth mean {a[:,0].mean():.2f} mm, frac>2mm {a[:,1].mean():.2f}, grip_x mean {a[:,2].mean():.0f}, term mean {a[:,3].mean():.3f}", flush=True)
"""
s = s[:j] + hook + s[j:]
open(p,"w").write(s); print("reward hook installed")
