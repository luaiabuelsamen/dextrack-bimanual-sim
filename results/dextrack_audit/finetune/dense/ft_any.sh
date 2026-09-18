#!/bin/bash
# Fine-tune any checkpoint on any clip with the dense-volume penetration term,
# then the exact-test eval at 16 envs for base and fine-tuned.
# Usage: ft_any.sh <tag> <inst> <basecmd> <ckpt> <w> <spacing> <sdf> <epochs>
tag=$1; inst=$2; basecmd=$3; ck=$4; w=$5; sp=$6; sdf=$7; nep=${8:-150}
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
export AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/dense
ep=$(/workspace/dextrack-venv/bin/python -c "import torch;d=torch.load('$ck',map_location='cpu');print(int(d.get('epoch',0)))" 2>/dev/null); ep=${ep:-0}; maxe=$((ep+nep))
SUB="s#ori_grab_s4_apple_lift#$inst#g; s#ori_grab_s2_cubesmall_inspect_1#$inst#g"
CMD=$(sed "$SUB; s#checkpoint=[^ ]*#checkpoint=$ck#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=1024 /; s/minibatch_size=[0-9]* /minibatch_size=1024 /; s/test=True/test=False/; s/train.params.config.max_epochs=[0-9]*/train.params.config.max_epochs=$maxe/; s#train.params.config.log_path=[^ ]*#train.params.config.log_path=./logs/dense_$tag#; s#train.params.config.train_dir=[^ ]*#train.params.config.train_dir=./logs/dense_$tag#" $basecmd)
AUDIT_PEN_W=$w AUDIT_FORCE_W=0 AUDIT_PROBE_SPACING=$sp AUDIT_PROBE_SDF=$sdf timeout 7200 bash -c "$CMD" > /workspace/dense/$tag.train.log 2>&1
FT=$(ls -t logs/dense_$tag/*/nn/last_*.pth 2>/dev/null | head -1)
for who in base ft; do
  if [ $who = base ]; then C=$ck; else C=$FT; fi
  [ -z "$C" ] && continue
  ECMD=$(sed "$SUB; s#checkpoint=[^ ]*#checkpoint=$C#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=16 /; s/minibatch_size=[0-9]* /minibatch_size=16 /" $basecmd)
  AUDIT_ALLENV=1 AUDIT_PROBE_SPACING=0.002 AUDIT_LOG=/workspace/dense/$tag.$who.npy timeout 1800 bash -c "$ECMD" > /workspace/dense/$tag.$who.log 2>&1
  echo "$tag $who $(grep -h AUDIT_ALLENV /workspace/dense/$tag.$who.log | tail -1)" >> /workspace/dense/summary.txt
done
echo "$tag DONE" >> /workspace/dense/done.txt
