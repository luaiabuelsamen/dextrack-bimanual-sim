#!/bin/bash
# Penetration-aware fine-tuning sweep. Jobs: cube w100, cube w1000 (from the
# released cube ckpt); apple/torus/stamp/flashlight at w300 from the generalist
# ckpt. Each job: fine-tune 150 epochs at 1024 envs, then evaluate baseline and
# fine-tuned over 16 envs with the all-env probe. Two jobs at a time.
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
export AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003
mkdir -p /workspace/sweep
job() {   # tag inst base_cmd_file base_ckpt w
  tag=$1; inst=$2; basecmd=$3; ck=$4; w=$5
  cd /workspace/DexTrack/isaacgymenvs
  # the checkpoint's epoch counter continues, so read it and add 150
  ep=$(/workspace/dextrack-venv/bin/python -c "import torch;d=torch.load('$ck',map_location='cpu');print(int(d.get('epoch',0)))" 2>/dev/null); ep=${ep:-0}; maxe=$((ep+150))
  CMD=$(sed "s#ori_grab_s4_apple_lift#$inst#g; s#ori_grab_s2_cubesmall_inspect_1#$inst#g; s#checkpoint=[^ ]*#checkpoint=$ck#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=1024 /; s/minibatch_size=[0-9]* /minibatch_size=1024 /; s/test=True/test=False/; s/train.params.config.max_epochs=[0-9]*/train.params.config.max_epochs=$maxe/; s#train.params.config.log_path=[^ ]*#train.params.config.log_path=./logs/sweep_$tag#; s#train.params.config.train_dir=[^ ]*#train.params.config.train_dir=./logs/sweep_$tag#" $basecmd)
  AUDIT_PEN_W=$w AUDIT_FORCE_W=0 timeout 5400 bash -c "$CMD" > /workspace/sweep/$tag.train.log 2>&1
  FT=$(ls -t logs/sweep_$tag/*/nn/*.pth 2>/dev/null | head -1)
  for who in base ft; do
    if [ $who = base ]; then C=$ck; else C=$FT; fi
    [ -z "$C" ] && continue
    ECMD=$(sed "s#ori_grab_s4_apple_lift#$inst#g; s#ori_grab_s2_cubesmall_inspect_1#$inst#g; s#checkpoint=[^ ]*#checkpoint=$C#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=16 /; s/minibatch_size=[0-9]* /minibatch_size=16 /" $basecmd)
    AUDIT_ALLENV=1 AUDIT_LOG=/workspace/sweep/$tag.$who.npy timeout 900 bash -c "$ECMD" > /workspace/sweep/$tag.$who.log 2>&1
    echo "$tag $who $(grep -h AUDIT_ALLENV /workspace/sweep/$tag.$who.log | tail -1 | cut -c1-200)" >> /workspace/sweep/summary.txt
  done
  echo "$tag DONE" >> /workspace/sweep/done.txt
}
export -f job
CUBE=./ckpts/s2_cubesmall_inspect_ckpt.pth
GEN=./ckpts/grab_trajs_tracking_ckpt.pth
printf "%s\n" \
 "cube_w100 ori_grab_s2_cubesmall_inspect_1 /workspace/train_cmd.txt $CUBE 100" \
 "cube_w1000 ori_grab_s2_cubesmall_inspect_1 /workspace/train_cmd.txt $CUBE 1000" \
 "apple_w300 ori_grab_s10_apple_eat_1 /workspace/gen_cmd.txt $GEN 300" \
 "torus_w300 ori_grab_s10_torusmedium_inspect_1 /workspace/gen_cmd.txt $GEN 300" \
 "stamp_w300 ori_grab_s10_stamp_lift /workspace/gen_cmd.txt $GEN 300" \
 "flashlight_w300 ori_grab_s2_flashlight_on_1 /workspace/gen_cmd.txt $GEN 300" \
 | xargs -P 2 -L 1 bash -c 'job $0 $1 $2 $3 $4'
echo ALL >> /workspace/sweep/done.txt
