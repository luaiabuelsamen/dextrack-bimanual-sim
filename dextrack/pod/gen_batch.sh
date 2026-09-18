#!/bin/bash
# generalist checkpoint on every clip in clips.txt, fixed init, 4 envs, audit per clip; 3 workers
cd /workspace/DexTrack/isaacgymenvs
export PATH=/workspace/dextrack-venv/bin:$PATH
export LD_LIBRARY_PATH=$(/workspace/dextrack-venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH
mkdir -p /workspace/gen
run_one() {
  inst=$1
  export AUDIT_LOG=/workspace/gen/$inst.npy AUDIT_FIX=3 AUDIT_PARK=1 AUDIT_PARK_DZ=0.003
  CMD=$(sed "s#ori_grab_s4_apple_lift#$inst#g" /workspace/gen_cmd.txt)
  timeout 1200 bash -c "$CMD" > /workspace/gen/$inst.log 2>&1
  cd /workspace && /workspace/dextrack-venv/bin/python /workspace/audit_dextrack.py --log /workspace/gen/$inst.npy --hand-urdf /workspace/DexTrack/assets/allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf --obj-dir /workspace/DexTrack/assets/meshdatav3_scaled/sem/$inst/coacd --out /workspace/gen/$inst.json > /workspace/gen/$inst.audit 2>&1
  echo "$inst $(grep -h tot_obj_succ_nn /workspace/gen/$inst.log | tail -1)" >> /workspace/gen/done.txt
  cd /workspace/DexTrack/isaacgymenvs
}
export -f run_one
cat /workspace/clips.txt | xargs -P 3 -I{} bash -c "run_one {}"
echo ALL >> /workspace/gen/done.txt
