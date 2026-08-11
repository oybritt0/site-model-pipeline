r"""
tile_render.py -- shaded views of a solid, without an OpenGL context.

Question 4 of the tile is whether the thing looks like a city or like a pile of
blobs, and that can only be answered by looking. There is no offscreen GL here
and a million-triangle mesh is far too slow to rasterise in Python, so this
samples the surface densely, projects the points, and paints them far-to-near.
Shading is Lambertian from the face normal plus a little ambient. It is a point
render, but at a few million samples it is indistinguishable from a mesh render
at these sizes, and it is a few seconds.
"""
from __future__ import annotations

import numpy as np


def _basis(azim_deg: float, elev_deg: float):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(up0, fwd)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return right, up, fwd


def render(mesh, size=1000, azim=35.0, elev=28.0, samples=None,
           light=(0.4, -0.7, 0.6), bg=1.0, margin=0.04, region=None):
    """Orthographic shaded render. Returns a float image in [0, 1].

    Sample count defaults to ~10 per pixel. Below about 6 the render speckles --
    points land by Poisson, so at 1.5 per pixel roughly a fifth of the image gets
    no sample at all and a large flat ground reads as noise.

    `region` is an (x0, y0, x1, y1) plan crop. A 12 in tile shown 800 px wide
    gives each 1.2 in mass 80 px, which is far coarser than the eye gets at
    arm's length from the printed object -- so judging the massing needs a crop,
    not just the whole tile.
    """
    if samples is None:
        samples = int(10 * size * size)
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, int(samples))
    nrm = mesh.face_normals[fid]

    if region is not None:
        x0, y0, x1, y1 = region
        keep = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        if keep.sum() < 1000:
            raise ValueError("crop region caught almost nothing")
        pts, nrm = pts[keep], nrm[keep]

    right, up, fwd = _basis(azim, elev)

    # Drop back-facing samples. For a closed mesh they can never be seen, and
    # leaving them in is what speckles a plan view: half the samples land on the
    # slab's dark underside, the lit top surface is left with holes, and the
    # underside shows through them.
    face = nrm @ fwd > 0
    if face.sum() > 1000:
        pts, nrm = pts[face], nrm[face]

    u = pts @ right
    v = pts @ up
    depth = pts @ fwd                       # larger = nearer the camera

    span = max(u.max() - u.min(), v.max() - v.min()) or 1.0
    k = (size * (1 - 2 * margin)) / span
    ix = np.clip(((u - u.min()) * k + size * margin).astype(np.int32), 0, size - 1)
    iy = np.clip((size - 1 - ((v - v.min()) * k + size * margin)).astype(np.int32),
                 0, size - 1)

    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    shade = 0.22 + 0.78 * np.clip(nrm @ L, 0, 1)
    # a touch of the viewing angle keeps silhouettes from going flat
    shade *= 0.80 + 0.20 * np.clip(np.abs(nrm @ fwd), 0, 1)

    img = np.full((size, size), float(bg), np.float32)
    painted = np.zeros((size, size), bool)
    order = np.argsort(depth)               # far first, near overwrites
    img[iy[order], ix[order]] = shade[order]
    painted[iy, ix] = True

    # A large flat face gets samples in proportion to its area, so at 700 px the
    # ground can still land only ~3 per pixel and a few percent of it comes out
    # unpainted -- white pinholes across the plate. Fill a hole only when it is
    # ringed by painted neighbours, so silhouette edges stay sharp.
    from scipy import ndimage as ndi
    for _ in range(2):
        holes = ~painted
        if not holes.any():
            break
        n = ndi.uniform_filter(painted.astype(np.float32), 3)
        fill = holes & (n >= 6 / 9)
        if not fill.any():
            break
        img[fill] = ndi.median_filter(img, 3)[fill]
        painted |= fill
    return np.clip(img, 0, 1)


def save(mesh, path, **kw):
    from PIL import Image
    img = render(mesh, **kw)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


def contact(mesh, path, size=760, views=((35, 28), (125, 28), (0, 90), (0, 6)),
            samples=None):
    """Four views in a row: two obliques, the plan, and a near-elevation."""
    from PIL import Image
    tiles = [render(mesh, size=size, azim=a, elev=e, samples=samples)
             for a, e in views]
    strip = np.concatenate(tiles, axis=1)
    Image.fromarray((strip * 255).astype(np.uint8)).save(path)
    return path
