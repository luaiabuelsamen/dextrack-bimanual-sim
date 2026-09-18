"""W&B observer for rl_games, restored (the release imports it commented out).
Mirrors rl_games' TensorBoard scalars (rewards, episode lengths, losses, lr)
into a W&B run and records every AUDIT_* environment variable in the config,
so a run's penetration-term settings are on its page."""
import os
import wandb
from rl_games.common.algo_observer import AlgoObserver


class WandbAlgoObserver(AlgoObserver):
    def __init__(self, cfg):
        self.cfg = cfg

    def before_init(self, base_name, config, *args, **kwargs):
        cfg = self.cfg
        name = os.environ.get("AUDIT_RUN_NAME") or str(cfg.wandb_name)
        wandb.init(project=str(cfg.wandb_project), entity=str(cfg.wandb_entity) or None,
                   group=str(cfg.wandb_group) or None, tags=[str(t) for t in cfg.wandb_tags],
                   name=name, job_type="train", sync_tensorboard=True, resume="allow",
                   settings=wandb.Settings(start_method="fork"))
        audit = {k: v for k, v in os.environ.items() if k.startswith("AUDIT_")}
        wandb.config.update({"audit": audit, "num_envs": int(cfg.task.env.numEnvs),
                             "max_epochs": int(cfg.train.params.config.max_epochs),
                             "checkpoint": str(cfg.checkpoint), "object": str(cfg.task.env.object_name)},
                            allow_val_change=True)

    def after_init(self, algo):
        pass
