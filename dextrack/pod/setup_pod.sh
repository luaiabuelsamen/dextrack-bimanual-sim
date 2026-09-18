#!/bin/bash
# Rebuild the DexTrack venv on the replacement pod (versions from the old venv).
set -x
export PATH=$HOME/.local/bin:$PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv python install 3.8
uv venv --python 3.8 /workspace/dextrack-venv
P=/workspace/dextrack-venv/bin/python
uv pip install --python $P torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python $P numpy==1.24.4 gym==0.23.1 omegaconf==2.3.1 hydra-core==1.3.2 rl-games==1.6.1 scipy==1.10.1 trimesh==4.12.2 rtree ninja pyyaml termcolor tensorboard==2.14.0 tensorboardX wandb==0.12.21 imageio matplotlib==3.7.5 mujoco==3.2.3 transforms3d opencv-python pillow==10.4.0 psutil setproctitle pygame==2.1.0 pyopengl glfw
echo BASE_DONE
until [ -f /workspace/PHASE1 ]; do sleep 15; done
uv pip install --python $P -e /workspace/isaacgym_pkg/isaacgym/python
uv pip install --python $P /workspace/DexTrack/whls/torch_cluster-1.6.3+pt24cu121-cp38-cp38-linux_x86_64.whl
cd /workspace/DexTrack/isaacgymenvs && uv pip install --python $P -e . --no-deps
$P -c "import isaacgym, isaacgymenvs, rl_games, torch, trimesh; print('IMPORTS OK', torch.__version__, torch.cuda.is_available())"
echo SETUP_DONE
