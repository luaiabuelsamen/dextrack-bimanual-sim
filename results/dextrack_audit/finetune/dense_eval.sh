#!/bin/bash
# Re-score the cube checkpoints over 16 envs with the DENSE probe (2 mm grid).
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
export AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003
W100=$(ls -t logs/sweep_cube_w100/*/nn/last_*.pth | head -1)
W300=$(ls -t logs/pen_ft2/*/nn/last_*.pth | head -1)
W1000=$(ls -t logs/sweep_cube_w1000/*/nn/last_*.pth | head -1)
for pair in "base ./ckpts/s2_cubesmall_inspect_ckpt.pth" "w100 $W100" "w300 $W300" "w1000 $W1000"; do
  set -- $pair; tag=$1; C=$2
  ECMD=$(sed "s#checkpoint=[^ ]*#checkpoint=$C#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=16 /; s/minibatch_size=[0-9]* /minibatch_size=16 /" /workspace/train_cmd.txt)
  AUDIT_ALLENV=1 AUDIT_PROBE_SPACING=0.002 AUDIT_LOG=/workspace/dense/cube_$tag.npy timeout 1200 bash -c "$ECMD" > /workspace/dense/cube_$tag.log 2>&1
  echo "$tag $(grep -h AUDIT_ALLENV /workspace/dense/cube_$tag.log | tail -1)" >> /workspace/dense/summary.txt
done
echo ALL >> /workspace/dense/done.txt
