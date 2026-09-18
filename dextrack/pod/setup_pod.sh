#!/bin/bash
# Bring a fresh RunPod pod (image runpod/pytorch:*-cu12*-ubuntu2404, one
# 3090-class GPU, /workspace volume) to a state where every run in
# dextrack/runs/ launches. Idempotent. Expects the five archives from
# DATA.md in /workspace/inputs/.
set -euo pipefail
WS=/workspace; IN=$WS/inputs
export PATH=$HOME/.local/bin:$PATH
cd $WS

# 1. code: the DexTrack fork on its audit branch (the hook), and this repository
[ -d DexTrack ] || git clone -q -b audit https://github.com/luaiabuelsamen/DexTrack.git
[ -d dextrack-bimanual-sim ] || git clone -q https://github.com/luaiabuelsamen/dextrack-bimanual-sim.git
(cd dextrack-bimanual-sim && git pull -q)
cp dextrack-bimanual-sim/dextrack/penetration_torch.py $WS/penetration_torch.py
cp dextrack-bimanual-sim/dextrack/pod/run.py $WS/run.py
cp dextrack-bimanual-sim/dextrack/pod/*_cmd*.txt $WS/ 2>/dev/null || true
[ -f $WS/train_cmd.txt ] || cp $WS/train_cmd_cube.txt $WS/train_cmd.txt
rm -rf $WS/specs && cp -r dextrack-bimanual-sim/dextrack/runs $WS/specs

# 2. data: verify, then extract what is missing
(cd $IN && sha256sum -c $WS/dextrack-bimanual-sim/dextrack/pod/data.sha256)
[ -d isaacgym_pkg/isaacgym ] || (mkdir -p isaacgym_pkg && tar -xf $IN/IsaacGym_Preview_4_Package.tar -C isaacgym_pkg)
[ -d DexTrack/isaacgymenvs/ckpts ] || unzip -q $IN/ckpts.zip -d DexTrack/isaacgymenvs -x "__MACOSX/*"
[ -d DexTrack/assets/rsc/objs ] || (mkdir -p DexTrack/assets/rsc && unzip -q $IN/objs.zip -d DexTrack/assets/rsc)
[ -d DexTrack/assets/meshdatav3_scaled/sem ] || unzip -q $IN/meshdatav3_scaled.zip -d DexTrack/assets
if [ ! -d DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_reduced_300 ]; then
  unzip -q $IN/data.zip -d DexTrack/isaacgymenvs
  (cd DexTrack/isaacgymenvs/data && for z in *.zip; do unzip -q "$z" && rm "$z"; done)
fi
# the release omits datasetv4.1; their loader reads an unused qpos and a scale of 1.0
python3 - <<'PY'
import os, numpy as np
root = "/workspace/DexTrack/assets/datasetv4.1"; sem = "/workspace/DexTrack/assets/meshdatav3_scaled/sem"
os.makedirs(root, exist_ok=True); n = 0
for inst in os.listdir(sem):
    p = os.path.join(root, f"{inst}.npz")
    if not os.path.exists(p):
        np.savez(p, qpos=np.zeros(22, np.float32), scale=np.float32(1.0)); n += 1
print("datasetv4.1 stubs written:", n)
PY

# 3. python: uv, 3.8, the venv, the pinned stack
which uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null)
uv python install 3.8 >/dev/null
[ -x dextrack-venv/bin/python ] || uv venv -q --python 3.8 dextrack-venv
P=$WS/dextrack-venv/bin/python
uv pip install -q --python $P torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -q --python $P numpy==1.24.4 gym==0.23.1 omegaconf==2.3.1 hydra-core==1.3.2 rl-games==1.6.1 scipy==1.10.1 \
  trimesh==4.12.2 rtree ninja pyyaml termcolor tensorboard==2.14.0 tensorboardX "wandb>=0.17,<0.19" huggingface_hub imageio \
  matplotlib==3.7.5 mujoco==3.2.3 transforms3d opencv-python pillow==10.4.0 psutil setproctitle pygame==2.1.0 pyopengl glfw
uv pip install -q --python $P -e $WS/isaacgym_pkg/isaacgym/python
uv pip install -q --python $P $WS/DexTrack/whls/torch_cluster-1.6.3+pt24cu121-cp38-cp38-linux_x86_64.whl
echo $WS/DexTrack > dextrack-venv/lib/python3.8/site-packages/dextrack.pth
export PATH=$WS/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$($P -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"):${LD_LIBRARY_PATH:-}
(cd DexTrack/isaacgymenvs && $P -c "import isaacgym; from isaacgym import gymtorch; import isaacgymenvs.tasks, rl_games, trimesh, wandb; import torch; print('imports ok, cuda', torch.cuda.is_available())")

# 4. the guardrail: nothing runs at scale that did not run at 4 environments
cd DexTrack/isaacgymenvs
SMOKE=$(sed "s/task.env.numEnvs=[0-9]* /task.env.numEnvs=4 /; s/minibatch_size=[0-9]* /minibatch_size=4 /" $WS/train_cmd.txt)
AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003 AUDIT_ALLENV=1 AUDIT_PROBE_SPACING=0.002 timeout 900 bash -c "$SMOKE" > $WS/smoke.log 2>&1 || true
grep -h "AUDIT_ALLENV" $WS/smoke.log | tail -1 | cut -c1-120 || { echo "SMOKE FAILED, see $WS/smoke.log"; exit 1; }
echo "POD READY: python /workspace/run.py /workspace/specs/<spec>.json"
