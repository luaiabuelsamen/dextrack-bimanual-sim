"""Run ledger, W&B evaluation runs and Hugging Face upload for DexTrack runs.

One row per run in `results/dextrack_audit/runs.jsonl`: what was trained
(clip, checkpoint, term weight, probe settings, epochs) and what the exact
test measured for base and fine-tuned (mean penetration, frames over 2 mm,
grip, error, held). The ledger is the source of truth; W&B and the Hub
mirror it.

  python dextrack/track.py add   --tag cube_w1000_sdf --clip ori_grab_s2_cubesmall_inspect_1 \\
      --dir results/dextrack_audit/finetune/dense [--gif figures/x.gif] [--audit out/x.json] [--ckpt path]
  python dextrack/track.py backfill --dir results/dextrack_audit/finetune/dense      # every *.ft.log / *.base.log pair
  python dextrack/track.py publish --tag cube_w1000_sdf                             # upload the run folder to the Hub
  python dextrack/track.py relog                                                    # W&B runs for ledger rows that have none

W&B: project `dextrack-bimanual`, one run of job_type `eval` per tag, with
the base and fine-tuned metrics side by side, the per-env values as a
table, the per-frame penetration as a curve, and the render as media.
Hub: private dataset repo `luaia/dextrack-bimanual-runs`, one folder per
run. Private because every log holds object trajectories derived from GRAB,
which is licensed and not redistributable.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

LEDGER = Path("results/dextrack_audit/runs.jsonl")
PROJECT = "dextrack-bimanual"
ENTITY = "luai-abuelsamen-university-of-california-berkeley"
HF_REPO = "luaia/dextrack-bimanual-runs"

_LINE = re.compile(r"AUDIT_ALLENV(?: spacing=(?P<spacing>[\d.]+))? envs=(?P<envs>\d+)\s+pen_mm mean (?P<pen>[\d.]+) \(per-env (?P<per>\[.*?\])\)\s+frac>2mm (?P<frac>[\d.]+)\s+grip_x mean (?P<grip>\d+)\s+err_cm mean (?P<err>[\d.]+)\s+end err per env (?P<end>\[.*?\])")


def parse_eval_log(path: Path) -> dict | None:
    """The last AUDIT_ALLENV line of an evaluation log, as a dict."""
    lines = [l for l in path.read_text(errors="ignore").splitlines() if "AUDIT_ALLENV" in l]
    if not lines:
        return None
    m = _LINE.search(lines[-1])
    if not m:
        return None
    end = json.loads(m["end"])
    return {"spacing": float(m["spacing"] or 0), "envs": int(m["envs"]), "pen_mm_mean": float(m["pen"]),
            "pen_mm_per_env": json.loads(m["per"]), "frac_over_2mm": float(m["frac"]), "grip_x_mean": float(m["grip"]),
            "err_cm_mean": float(m["err"]), "end_err_cm_per_env": end, "held": sum(1 for x in end if x < 10.0)}


def load_summary(d: Path, tag: str, who: str) -> dict | None:
    js = d / f"{tag}.{who}.summary.json"
    if js.exists():
        return json.loads(js.read_text())
    log = d / f"{tag}.{who}.log"
    return parse_eval_log(log) if log.exists() else None


def train_settings(d: Path, tag: str) -> dict:
    """Weight, probe and epoch settings from the training log's PEN-REWARD header, if any."""
    out = {}
    log = d / f"{tag}.train.log"
    if log.exists():
        txt = log.read_text(errors="ignore")
        m = re.search(r"PEN-REWARD on: weight ([\d.]+) N, hulls (\d+)(?:, points (\d+) spacing ([\d.]+))?, w_pen ([\d.]+), w_force ([\d.]+)", txt)
        if m:
            out.update({"object_weight_n": float(m[1]), "hulls": int(m[2]), "probe_points": int(m[3] or 0),
                        "probe_spacing": float(m[4] or 0), "w_pen": float(m[5]), "w_force": float(m[6])})
        ep = re.findall(r"epoch: (\d+)/(\d+)", txt.replace("\r", "\n"))
        if ep:
            out["epochs"] = [int(ep[0][0]), int(ep[-1][1])]
        last = [l for l in txt.splitlines() if "PEN-REWARD last200" in l]
        if last:
            mm = re.search(r"depth mean ([\d.]+) mm", last[-1])
            if mm:
                out["train_depth_mm_final"] = float(mm[1])
    return out


def add(args) -> dict:
    d = Path(args.dir)
    row = {"tag": args.tag, "clip": args.clip, "dir": str(d), "time": time.strftime("%Y-%m-%dT%H:%M"),
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "train": train_settings(d, args.tag), "base": load_summary(d, args.tag, "base"), "ft": load_summary(d, args.tag, "ft"),
           "gif": args.gif, "audit": args.audit, "ckpt": args.ckpt, "note": args.note}
    if args.audit and Path(args.audit).exists():
        row["audit_env0"] = json.loads(Path(args.audit).read_text())["summary"]
    spec = d / f"{args.tag}.spec.json"                  # written by pod/run.py: the spec, commands, commits, timings
    if spec.exists():
        rec = json.loads(spec.read_text())
        row["spec"] = rec.get("spec"); row["commits"] = rec.get("commits")
        row["train_seconds"] = rec.get("train_seconds"); row["train_cmd"] = rec.get("train_cmd")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"ledger: {args.tag}  base {fmt(row['base'])}  ft {fmt(row['ft'])}")
    if not args.no_wandb:
        row["wandb"] = log_wandb(row)
        rewrite_last(row)
    return row


def fmt(s):
    return "-" if not s else f"{s['pen_mm_mean']:.2f} mm / {s['frac_over_2mm']*100:.0f} % / {s['grip_x_mean']:.0f}x / {s['held']} of {s['envs']} held"


def rewrite_last(row):
    rows = LEDGER.read_text().splitlines()
    rows[-1] = json.dumps(row)
    LEDGER.write_text("\n".join(rows) + "\n")


def log_wandb(row) -> str | None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed; ledger only", file=sys.stderr)
        return None
    run = wandb.init(project=PROJECT, entity=ENTITY, name=row["tag"], job_type="eval", group=row["clip"],
                     config={"clip": row["clip"], **row["train"], "commit": row["commit"], "note": row["note"]},
                     reinit=True, settings=wandb.Settings(silent=True))
    summary = {}
    for who in ("base", "ft"):
        s = row.get(who)
        if not s:
            continue
        for k in ("pen_mm_mean", "frac_over_2mm", "grip_x_mean", "err_cm_mean", "held"):
            summary[f"{who}/{k}"] = s[k]
        try:
            t = wandb.Table(columns=["env", "pen_mm", "end_err_cm"] + (["grip_x"] if "grip_x_per_env" in s else []))
            for i, (p, e) in enumerate(zip(s["pen_mm_per_env"], s["end_err_cm_per_env"])):
                t.add_data(i, p, e, *([s["grip_x_per_env"][i]] if "grip_x_per_env" in s else []))
            wandb.log({f"{who}/per_env": t})
        except Exception as exc:                      # a broken pandas makes wandb refuse tables
            print(f"table skipped ({exc.__class__.__name__}); per-env values kept in the summary", file=sys.stderr)
            summary[f"{who}/pen_mm_per_env"] = s["pen_mm_per_env"]
            summary[f"{who}/end_err_cm_per_env"] = s["end_err_cm_per_env"]
        if "pen_mm_per_frame" in s:
            for fr, p in enumerate(s["pen_mm_per_frame"]):
                wandb.log({f"{who}/pen_mm_frame": p, "frame": fr})
    if row.get("audit_env0"):
        a = row["audit_env0"]
        summary.update({"env0/pen_mm_mean": a["pen_mm_mean"], "env0/frames_over_2mm": a["frames_over_2mm"],
                        "env0/grip_touch_x_median": a["grip_touch_x_median"], "env0/pos_err_cm_mean": a["pos_err_cm_mean"]})
    if row.get("gif") and Path(row["gif"]).exists():
        wandb.log({"render": wandb.Video(row["gif"], format="gif")})
        prev = Path(row["gif"]).with_suffix(".preview.jpg")
        if prev.exists():
            wandb.log({"preview": wandb.Image(str(prev))})
    wandb.summary.update(summary)
    url = run.url
    run.finish()
    print("wandb:", url)
    return url


def relog(args):
    """Log W&B runs for ledger rows that have none (or all, with --force)."""
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines()]
    for i, row in enumerate(rows):
        if row.get("wandb") and not args.force:
            continue
        if args.tag and row["tag"] != args.tag:
            continue
        row["wandb"] = log_wandb(row)
        rows[i] = row
    LEDGER.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def backfill(args):
    d = Path(args.dir)
    tags = sorted({p.name.rsplit(".", 2)[0] for p in d.glob("*.ft.log")} | {p.name.rsplit(".", 2)[0] for p in d.glob("*.base.log")})
    done = {json.loads(l)["tag"] for l in LEDGER.read_text().splitlines()} if LEDGER.exists() else set()
    for tag in tags:
        if tag in done and not args.force:
            continue
        clip = args.clip or ("ori_grab_s10_apple_eat_1" if tag.startswith("apple") else "ori_grab_s2_cubesmall_inspect_1")
        ns = argparse.Namespace(tag=tag, clip=clip, dir=str(d), gif=None, audit=None, ckpt=None, note="backfill", no_wandb=args.no_wandb)
        add(ns)


def publish(args):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines()]
    row = next((r for r in reversed(rows) if r["tag"] == args.tag), None)
    if row is None:
        sys.exit(f"no ledger row for {args.tag}")
    stage = Path("out/hf_stage") / args.tag
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    d = Path(row["dir"])
    for p in list(d.glob(f"{args.tag}.*")):
        if p.suffix in (".log", ".json"):
            shutil.copy(p, stage / p.name)
    for key in ("gif", "audit", "ckpt"):
        if row.get(key) and Path(row[key]).exists():
            shutil.copy(row[key], stage / Path(row[key]).name)
    (stage / "row.json").write_text(json.dumps(row, indent=1))
    api.upload_folder(folder_path=str(stage), path_in_repo=f"runs/{args.tag}", repo_id=HF_REPO, repo_type="dataset",
                      commit_message=f"{args.tag}: {fmt(row.get('ft')) if row.get('ft') else 'run'}")
    api.upload_file(path_or_fileobj=str(LEDGER), path_in_repo="runs.jsonl", repo_id=HF_REPO, repo_type="dataset",
                    commit_message="ledger")
    url = f"https://huggingface.co/datasets/{HF_REPO}/tree/main/runs/{args.tag}"
    print("hub:", url)
    row["hf"] = url
    rows[[i for i, r in enumerate(rows) if r["tag"] == args.tag][-1]] = row
    LEDGER.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("--tag", required=True); a.add_argument("--clip", required=True); a.add_argument("--dir", required=True)
    a.add_argument("--gif"); a.add_argument("--audit"); a.add_argument("--ckpt"); a.add_argument("--note", default=""); a.add_argument("--no-wandb", action="store_true")
    b = sub.add_parser("backfill"); b.add_argument("--dir", required=True); b.add_argument("--clip"); b.add_argument("--force", action="store_true"); b.add_argument("--no-wandb", action="store_true")
    p = sub.add_parser("publish"); p.add_argument("--tag", required=True)
    r = sub.add_parser("relog"); r.add_argument("--tag"); r.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"add": add, "backfill": backfill, "publish": publish, "relog": relog}[args.cmd](args)


if __name__ == "__main__":
    main()
