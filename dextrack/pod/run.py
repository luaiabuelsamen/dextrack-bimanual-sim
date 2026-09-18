#!/usr/bin/env python3
"""Launch a DexTrack run on the pod from a JSON spec, and leave a record.

    python run.py specs/cube_w1000_dense.json            # train + evaluate
    python run.py specs/cube_w1000_dense.json --dry-run  # print the commands
    python run.py specs/cube_scratch.json --tag cube_scratch_s2 --set seed=2

A spec (see ../runs/*.json in the repository) names the clip, the base
command line (their own test command for that clip family), the checkpoint
to start from (or none, for a from-scratch run), the penetration term and
probe settings, epochs, environments and seed. The launcher writes
<out>/<tag>.spec.json with the spec, the resolved commands, both
repositories' commits and the timings, then trains, then evaluates base
and fine-tuned at 16 environments with the exact plane test, writing
<tag>.<who>.summary.json. The Jetson-side ledger (dextrack/track.py)
reads that spec back, so every ledger row can be relaunched.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

WS = Path(os.environ.get("AUDIT_WORKSPACE", "/workspace"))
DEXTRACK = WS / "DexTrack" / "isaacgymenvs"
BASE_CMDS = {"cube": WS / "train_cmd.txt", "gen": WS / "gen_cmd.txt"}
CLIP_IN_BASE = {"cube": "ori_grab_s2_cubesmall_inspect_1", "gen": "ori_grab_s4_apple_lift"}
ENTITY = "luai-abuelsamen-university-of-california-berkeley"
PROJECT = "dextrack-bimanual"

DEFAULTS = {"kind": "finetune", "base": "cube", "checkpoint": None, "w_pen": 0.0, "w_force": 0.0, "gate_m": 0.0,
            "spacing": 0.002, "sdf_res": 0.001, "epochs": 150, "envs": 1024, "seed": None, "eval_envs": 16,
            "wandb": True, "set": {}, "note": ""}


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def ckpt_epoch(ck: str) -> int:
    code = f"import torch;d=torch.load('{ck}',map_location='cpu');print(int(d.get('epoch',0)))"
    r = subprocess.run([str(WS / "dextrack-venv/bin/python"), "-c", code], capture_output=True, text=True, cwd=DEXTRACK)
    return int(r.stdout.strip() or 0)


def build(spec: dict, tag: str, out: Path) -> tuple[str, dict[str, str], list[tuple[str, str, dict[str, str]]]]:
    base = BASE_CMDS[spec["base"]].read_text().strip()
    clip = spec["clip"]
    sub = lambda s: s.replace(CLIP_IN_BASE[spec["base"]], clip)          # noqa: E731
    ck = spec.get("checkpoint")
    train = sub(base)
    train = re.sub(r"checkpoint=\S*", f"checkpoint={ck}" if ck else "", train)
    train = re.sub(r"task\.env\.numEnvs=\d+ ", f"task.env.numEnvs={spec['envs']} ", train)
    train = re.sub(r"minibatch_size=\d+ ", f"minibatch_size={spec['envs']} ", train)
    train = train.replace("test=True", "test=False")
    start = ckpt_epoch(ck) if ck else 0
    train = re.sub(r"train\.params\.config\.max_epochs=\d+", f"train.params.config.max_epochs={start + spec['epochs']}", train)
    train = re.sub(r"train\.params\.config\.log_path=\S*", f"train.params.config.log_path=./logs/{tag}", train)
    train = re.sub(r"train\.params\.config\.train_dir=\S*", f"train.params.config.train_dir=./logs/{tag}", train)
    if spec.get("seed") is not None:
        train += f" seed={spec['seed']}"
    if spec.get("wandb", True):
        group = clip if spec["kind"] == "finetune" else spec["kind"]
        train += f" wandb_activate=True wandb_project={PROJECT} wandb_entity={ENTITY} wandb_name={tag} wandb_group={group}"
    for k, v in spec.get("set", {}).items():
        train += f" {k}={v}"
    env = {"AUDIT_FIX": "3", "AUDIT_PARK": "1", "AUDIT_PARK_DZ": "0.003", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "AUDIT_RUN_NAME": tag}
    if spec["w_pen"] or spec["w_force"]:
        env.update({"AUDIT_PEN_W": str(spec["w_pen"]), "AUDIT_FORCE_W": str(spec["w_force"]),
                    "AUDIT_PROBE_SPACING": str(spec["spacing"]), "AUDIT_PROBE_SDF": str(spec["sdf_res"])})
        if spec["gate_m"]:
            env["AUDIT_PEN_GATE"] = str(spec["gate_m"])
    evals = []
    for who, c in (("base", ck), ("ft", f"$(ls -t logs/{tag}/*/nn/last_*.pth | head -1)")):
        if c is None:
            continue
        e = sub(base)
        e = re.sub(r"checkpoint=\S*", f"checkpoint={c}", e)
        e = re.sub(r"task\.env\.numEnvs=\d+ ", f"task.env.numEnvs={spec['eval_envs']} ", e)
        e = re.sub(r"minibatch_size=\d+ ", f"minibatch_size={spec['eval_envs']} ", e)
        eenv = {"AUDIT_FIX": "3", "AUDIT_PARK": "1", "AUDIT_PARK_DZ": "0.003", "AUDIT_ALLENV": "1", "AUDIT_PROBE_SPACING": "0.002",
                "AUDIT_SUMMARY": str(out / f"{tag}.{who}.summary.json"), "AUDIT_LOG": str(out / f"{tag}.{who}.npy")}
        evals.append((who, e, eenv))
    return train, env, evals


def run(cmd: str, env: dict[str, str], log: Path, timeout: int) -> int:
    full = dict(os.environ, **env)
    full["PATH"] = f"{WS}/dextrack-venv/bin:" + full.get("PATH", "")
    libdir = subprocess.run([str(WS / "dextrack-venv/bin/python"), "-c", "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"],
                            capture_output=True, text=True).stdout.strip()
    full["LD_LIBRARY_PATH"] = libdir + ":" + full.get("LD_LIBRARY_PATH", "")
    with log.open("w") as f:
        return subprocess.run(["bash", "-c", cmd], cwd=DEXTRACK, env=full, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec"); ap.add_argument("--tag"); ap.add_argument("--out", default=str(WS / "runs"))
    ap.add_argument("--set", action="append", default=[], help="extra hydra override key=value")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--train-timeout", type=int, default=6 * 3600)
    a = ap.parse_args()
    spec = {**DEFAULTS, **json.loads(Path(a.spec).read_text())}
    for kv in a.set:
        k, v = kv.split("=", 1); spec["set"][k] = v
    tag = a.tag or spec.get("tag") or Path(a.spec).stem
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train, env, evals = build(spec, tag, out)
    record = {"tag": tag, "spec": spec, "spec_file": str(a.spec), "train_cmd": train, "train_env": env,
              "evals": [{"who": w, "cmd": c, "env": e} for w, c, e in evals],
              "commits": {"DexTrack": git_commit(WS / "DexTrack"), "repo": git_commit(WS / "dextrack-bimanual-sim")},
              "host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.dry_run:
        print(json.dumps(record, indent=1)); return
    (out / f"{tag}.spec.json").write_text(json.dumps(record, indent=1))
    t0 = time.time()
    rc = run(train, env, out / f"{tag}.train.log", a.train_timeout)
    record["train_rc"] = rc; record["train_seconds"] = round(time.time() - t0)
    for who, cmd, eenv in evals:
        rc = run(cmd, eenv, out / f"{tag}.{who}.log", 1800)
        record.setdefault("eval_rc", {})[who] = rc
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / f"{tag}.spec.json").write_text(json.dumps(record, indent=1))
    with (out / "done.txt").open("a") as f:
        f.write(f"{tag} DONE\n")
    for who, _, _ in evals:
        s = out / f"{tag}.{who}.summary.json"
        if s.exists():
            d = json.loads(s.read_text())
            print(f"{tag} {who}: {d['pen_mm_mean']:.2f} mm, {d['frac_over_2mm']*100:.0f} % over 2 mm, grip {d['grip_x_mean']:.0f}x, held {d['held']} of {d['envs']}")


if __name__ == "__main__":
    main()
