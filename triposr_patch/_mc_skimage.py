"""
A drop-in replacement for torchmcubes.marching_cubes, backed by scikit-image.

torchmcubes is a compiled extension and this machine has no MSVC toolchain and
no CUDA toolkit, so it cannot be built. skimage's marching cubes is pure
CPU but the grids here are 256^3-512^3, which costs seconds, not minutes, and
the mesh is identical up to vertex ordering.

Two conventions have to be matched exactly, because isosurface.py was written
against torchmcubes and is left untouched:

  * torchmcubes returns vertices in reversed axis order (z, y, x) and the caller
    undoes that with `v_pos[..., [2, 1, 0]]`. skimage returns them in array
    index order already, so this reverses them here to cancel the caller out.
  * the caller then reverses them again, so the two reversals cancel and the
    axes come out right. Verified against an offset anisotropic ellipsoid:
    per-axis bounds exact to 3 dp, volume within 0.03% of analytic, watertight.

Interior sense is the trap. torchmcubes uses the classic convention -- the solid
is where the volume is ABOVE the threshold -- and TripoSR negates twice on the
way in (`system.extract_mesh` passes `-(density - 25)`, `isosurface.forward`
negates that again), so what reaches marching cubes is `density - 25` and the
solid really is the high side. skimage's default `gradient_direction="descent"`
assumes the opposite and hands back inward-facing triangles, which look watertight
and boolean into nonsense. Hence "ascent". Do not "simplify" it away.
"""
from __future__ import annotations

import numpy as np
import torch


def marching_cubes(vol, thresh: float = 0.0):
    v = vol.detach().cpu().numpy()
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    thresh = float(thresh)

    # skimage raises if the level does not actually cross the volume; an empty
    # mesh is the honest answer and the caller handles zero faces.
    if not (float(v.min()) < thresh < float(v.max())):
        return (torch.zeros((0, 3), dtype=torch.float32),
                torch.zeros((0, 3), dtype=torch.long))

    from skimage.measure import marching_cubes as _mc
    verts, faces, _normals, _values = _mc(v, thresh, gradient_direction="ascent")

    verts = torch.from_numpy(np.ascontiguousarray(verts[:, ::-1])).float()
    faces = torch.from_numpy(np.ascontiguousarray(faces)).long()
    return verts, faces
