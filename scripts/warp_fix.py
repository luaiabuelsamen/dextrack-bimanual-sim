"""Workaround for a mujoco_warp/warp bug that blocks any scene with more than
one kind of convex collision pair.

The bug. `convex_narrowphase` builds a SEPARATE specialised CCD kernel for each
convex geom-type pair in the model (box-box, box-mesh, mesh-mesh, ...), and for
each one immediately asks warp for a suggested launch block size:

    ccd_k    = ccd_kernel_builder(g1, g2, ...)
    ccd_grid = _ccd_grid_size(ccd_k, d.naconmax, device)   # <-- raises
    wp.launch(ccd_k, dim=ccd_grid, ...)

`_ccd_grid_size` calls `wp.get_suggested_block_size`, which does
`module.load(device)` and then reads `module_exec.meta[<kernel>_smem_bytes]`.
Once the module has been loaded for the FIRST pair, warp returns the cached
`ModuleExec`, whose `meta` was written before the later kernels existed. The
second distinct pair type therefore misses from `meta` and raises

    KeyError: '..._ccd_kernel_<hash>_cuda_kernel_forward_smem_bytes'

which is exactly what a scene with only one convex pair type never hits -- a
single-mesh toy scene works, two LEAP hands plus a box do not.

The fix. On the missing key, unload the kernel's module so the next load
rebuilds it with every kernel currently registered, then retry; if that still
fails, fall back to `naconmax`, which is exactly what this same function returns
on CPU. Both branches change only module compilation and launch width -- never
the simulation.

The unload matters. Returning a grid size alone leaves the kernel genuinely
absent from the loaded module, and the wp.launch immediately after fails with
"Warp CUDA error 500: named symbol not found (wp_cuda_get_kernel)".

Call `apply()` before any mujoco_warp stepping.
"""
from __future__ import annotations


def apply(verbose=True):
    import mujoco_warp._src.collision_convex as cc

    if getattr(cc, "_ccd_grid_size_patched", False):
        return False
    original = cc._ccd_grid_size

    def patched(kernel, naconmax: int, device):
        try:
            return original(kernel, naconmax, device)
        except KeyError:
            # The kernel was registered AFTER its module was loaded, so the
            # cached ModuleExec's binary and metadata both predate it. Falling
            # back on the grid size alone is not enough -- the subsequent
            # wp.launch then fails with "CUDA error 500: named symbol not
            # found", because the kernel really is absent from the loaded
            # module. Unloading forces the next load to rebuild with every
            # kernel currently registered.
            try:
                kernel.module.unload()
            except Exception:
                pass
            try:
                return original(kernel, naconmax, device)
            except Exception:
                return naconmax

    cc._ccd_grid_size = patched
    cc._ccd_grid_size_patched = True
    if verbose:
        print("[warp_fix] _ccd_grid_size falls back to naconmax when warp's "
              "kernel metadata is missing (launch heuristic only, not physics)")
    return True
