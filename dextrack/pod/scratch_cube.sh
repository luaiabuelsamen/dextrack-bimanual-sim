#!/bin/bash
# D4 step 1, the control: their per-clip policy trained FROM SCRATCH on the
# small cube, their command line, 1024 envs, no penetration term. Then the
# exact-test eval at 16 envs. Usage: scratch_cube.sh <tag> <epochs>
tag=$1; ep=${2:-3000}
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
export AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003
mkdir -p /workspace/scratch
CMD=$(sed "s#checkpoint=[^ ]*##g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=1024 /; s/minibatch_size=[0-9]* /minibatch_size=1024 /; s/test=True/test=False/; s/train.params.config.max_epochs=[0-9]*/train.params.config.max_epochs=$ep/; s#train.params.config.log_path=[^ ]*#train.params.config.log_path=./logs/scratch_$tag#; s#train.params.config.train_dir=[^ ]*#train.params.config.train_dir=./logs/scratch_$tag#" /workspace/train_cmd.txt)
echo "$CMD" > /workspace/scratch/$tag.cmd
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 14400 bash -c "$CMD" > /workspace/scratch/$tag.train.log 2>&1
FT=$(ls -t logs/scratch_$tag/*/nn/last_*.pth | head -1)
ECMD=$(sed "s#checkpoint=[^ ]*#checkpoint=$FT#g; s/task.env.numEnvs=[0-9]* /task.env.numEnvs=16 /; s/minibatch_size=[0-9]* /minibatch_size=16 /" /workspace/train_cmd.txt)
AUDIT_ALLENV=1 AUDIT_PROBE_SPACING=0.002 AUDIT_LOG=/workspace/scratch/$tag.npy timeout 1200 bash -c "$ECMD" > /workspace/scratch/$tag.eval.log 2>&1
echo "$tag $(grep -h AUDIT_ALLENV /workspace/scratch/$tag.eval.log | tail -1)" >> /workspace/scratch/summary.txt
echo "$tag DONE" >> /workspace/scratch/done.txt
