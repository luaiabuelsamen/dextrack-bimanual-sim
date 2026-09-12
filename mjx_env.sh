#!/usr/bin/env bash
# JAX-on-GPU for this Jetson (Orin, aarch64/Tegra, JetPack R36.4.3).
#
# This works, which overturns the note carried since May that "JAX has NO working
# CUDA build on aarch64/Tegra". The aarch64 wheels now exist
# (jax_cuda12_pjrt-*-manylinux2014_aarch64) and ARE real aarch64 ELF objects; the
# only problem is that the plugin looks for CUDA libraries in the standard
# locations and Jetson does not use them. Putting the venv's OWN nvidia/*/lib
# directories on LD_LIBRARY_PATH is sufficient:
#
#   backend: gpu   devices: [CudaDevice(id=0)]
#
# Usage:  ./mjx_env.sh python scripts/whatever.py
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-mjx"
NVLIBS="$(find "$VENV/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLIBS}/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "$1" = "python" ]; then shift; exec "$VENV/bin/python" "$@"; fi
exec "$@"
