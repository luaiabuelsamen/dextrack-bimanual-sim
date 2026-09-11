# opposition-deficit

**Question:** when a robot hand cannot oppose the way a human hand does, what should human
hand-object interaction data be retargeted *to*?

**Claim under test:** not the hand pose. Pose-fidelity retargeting — the objective behind
DexPilot, AnyTeleop, `dex-retargeting`, and the last mile of every HOI generation pipeline —
is anti-correlated with task success on opposition-deficient hands, and the second arm is the
repair.

- `GOAL.md` — the goal, the four claims, what would kill it, the rules it inherits.
- `PLAN.md` — milestones M0–M6, each with a kill criterion.
- `EXPERIMENTS.md` — simulator protocol, tasks, success criteria fixed before collection.
- `NOTES.md` — measurement log. Every paper number traces here.

Inherits the embodiment characterization, MuJoCo/MJX scene, tracking env and PPO recipe from
`../dextrack_vega`.
