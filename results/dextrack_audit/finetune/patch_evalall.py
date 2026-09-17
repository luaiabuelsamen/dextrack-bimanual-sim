p="tasks/allegro_hand_tracking_generalist.py"; s=open(p).read()
old="""        self._audit_rows.append(row)
"""
new="""        # all-environment penetration at eval time, via the GPU probe, so a
        # checkpoint's number is a mean over envs rather than env 0 alone
        if os.environ.get("AUDIT_ALLENV"):
            if not hasattr(self, "_ev_probe"):
                import sys as _sys; _sys.path.insert(0, "/workspace")
                from penetration_torch import PenetrationProbe
                _inst = self.object_code_list[0].split("_nf_")[0]
                self._ev_probe = PenetrationProbe("../assets/allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf", f"../assets/meshdatav3_scaled/sem/{_inst}/coacd", list(self._audit_names), self.device)
                self._ev_rows = []
            _d, _nt, _pl = self._ev_probe(self.rigid_body_states[:, :len(self._audit_names)], self.object_pose)
            _cf = self._audit_cf_t.view(self.num_envs, -1, 3)[:, :len(self._audit_names)][:, self._ev_probe.link_idx]
            _g = (torch.norm(_cf, dim=-1) * (_pl > -self._ev_probe.touch).float()).sum(-1) / (500.0 * self._ev_probe.volume * 9.81)
            _e = torch.norm(self.object_pos - self.goal_pos, dim=-1) if hasattr(self, "goal_pos") else torch.zeros(self.num_envs, device=self.device)
            self._ev_rows.append(torch.stack([_d * 1000, (_d > 0.002).float(), _g, _e * 100], -1).detach().cpu().numpy())
            if self.ref_ts >= 269:
                _a = np.stack(self._ev_rows[1:], 0)      # (T, N, 4)
                _np = _a.mean(0)                         # per env
                print("AUDIT_ALLENV envs=%d  pen_mm mean %.2f (per-env %s)  frac>2mm %.2f  grip_x mean %.0f  err_cm mean %.2f  end err per env %s" % (
                    _a.shape[1], _np[:, 0].mean(), np.round(_np[:, 0], 2).tolist(), _np[:, 1].mean(), _np[:, 2].mean(), _np[:, 3].mean(), np.round(_a[-1, :, 3], 1).tolist()), flush=True)
        self._audit_rows.append(row)
"""
assert s.count(old)==1; s=s.replace(old,new,1); open(p,"w").write(s); print("eval-all patch installed")
