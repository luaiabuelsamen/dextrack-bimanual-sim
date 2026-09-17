#!/bin/bash
# usage: run_audit.sh <inst_tag> <ckpt> <tag>
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
export AUDIT_LOG=/workspace/audit_$3.npy
bash scripts/run_audit_single.sh 0 $1 $2 True
