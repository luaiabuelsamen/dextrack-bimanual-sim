#!/bin/bash
# Cube fine-tune with the DENSE probe as the reward term (2 mm grid), then a
# dense 16-env eval. Usage: ft_dense.sh <tag> <w> [spacing]
tag=$1; w=$2; sp=${3:-0.002}
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
export AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003
mkdir -p /workspace/dense
ck=./ckpts/s2_cubesmall_inspect_ckpt.pth
ep=$(/workspace/dextrack-venv/bin/python -c "import torch;d=torch.load('$ck',map_location='cpu');print(int(d.get('epoch',0)))" 2>/dev/null); maxe=$((ep+150))
CMD=$(sed "s#checkpoint=[^ ]*#checkpoint=$ck#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=1024 /; s/minibatch_size=[0-9]* /minibatch_size=1024 /; s/test=True/test=False/; s/train.params.config.max_epochs=[0-9]*/train.params.config.max_epochs=$maxe/; s#train.params.config.log_path=[^ ]*#train.params.config.log_path=./logs/dense_$tag#; s#train.params.config.train_dir=[^ ]*#train.params.config.train_dir=./logs/dense_$tag#" /workspace/train_cmd.txt)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True AUDIT_PEN_W=$w AUDIT_FORCE_W=0 AUDIT_PROBE_SPACING=$sp timeout 7200 bash -c "$CMD" > /workspace/dense/$tag.train.log 2>&1
FT=$(ls -t logs/dense_$tag/*/nn/last_*.pth | head -1)
ECMD=$(sed "s#checkpoint=[^ ]*#checkpoint=$FT#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=16 /; s/minibatch_size=[0-9]* /minibatch_size=16 /" /workspace/train_cmd.txt)
AUDIT_ALLENV=1 AUDIT_PROBE_SPACING=0.002 AUDIT_LOG=/workspace/dense/$tag.ft.npy timeout 1200 bash -c "$ECMD" > /workspace/dense/$tag.ft.log 2>&1
echo "$tag ft $(grep -h AUDIT_ALLENV /workspace/dense/$tag.ft.log | tail -1)" >> /workspace/dense/summary.txt
echo "$tag DONE" >> /workspace/dense/done.txt
