r"""
tile_measure.py -- score a mesh against the assemblage it came from.

Shared by tile_triposr.py (local) and tile_tripo.py (paid API) so the two
backends are measured by exactly the same ruler and can be read against each
other rather than argued about.

The camera is TripoSR's convention -- right-handed, x back, y right, z up, input
view at azimuth 0 -- so the input silhouette is the (y, -z) projection. Tripo's
API models come back in a different frame, so `mesh_silhouette` takes the axes to
project onto; `best_silhouette` finds them once per backend by trying all six and
keeping the best, which is a one-off calibration, not a per-mesh fudge.

    iou           mesh silhouette against the input alpha, both normalised to
                  their own bounding box, so this is shape agreement and not a
                  scale artefact
    thin_recall   the same, restricted to input pixels lying in parts thinner
                  than one printed nozzle width at this object's real size. A
                  mesh can score a fine iou by getting the mass right and losing
                  every tracery, so this is the number that answers the question.
    interstice    the fraction of background pixels ENCLOSED by the input
                  silhouette that the mesh fills in. This is the one TripoSR
                  fails: it closes the gaps between fragments, and the gaps are
                  what makes an assemblage read as an assemblage.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

RASTER = 512
NOZZLE_IN = 0.4 / 25.4          # 0.4 mm nozzle, the printable feature floor

# (axis for image u, sign, axis for image v, sign)
#
# All 24, not 6. An earlier version tried only the six ordered axis pairs at
# fixed signs, which cannot express a mirrored view -- and a mirrored match scores
# like a total failure, so a perfectly good mesh read as broken. Tripo's GLBs are
# glTF, hence y-up, where this project is z-up, so the wrong guess is the normal
# case rather than the exception.
AXES = {f"{'xyz'[a]}{'+-'[sa]}{'xyz'[b]}{'+-'[sb]}": (a, 1 - 2 * sa, b, 1 - 2 * sb)
        for a in range(3) for b in range(3) if a != b
        for sa in range(2) for sb in range(2)}
TRIPOSR_AXES = "y+z-"           # x back, y right, z up, input view at azimuth 0


def fit_square(im: Image.Image, n: int = RASTER) -> np.ndarray:
    """Scale a mask into an n x n frame, preserving aspect, centred."""
    w, h = im.size
    k = (n - 2) / max(w, h)
    im = im.resize((max(1, int(round(w * k))), max(1, int(round(h * k)))),
                   Image.NEAREST)
    out = Image.new("L", (n, n), 0)
    out.paste(im, ((n - im.size[0]) // 2, (n - im.size[1]) // 2))
    return np.asarray(out) > 127


def rasterise_alpha(alpha: np.ndarray, n: int = RASTER) -> np.ndarray:
    m = alpha > 0.5
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros((n, n), bool)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return fit_square(Image.fromarray((crop * 255).astype(np.uint8)), n)


def mesh_silhouette(mesh, axes=TRIPOSR_AXES, n: int = RASTER,
                    max_faces: int | None = None) -> np.ndarray:
    """Union of every triangle projected onto the image plane. For a closed
    mesh that union is exactly the silhouette, occlusion and all.

    `max_faces` thins the draw for the orientation search only. The hosted models
    return 1.5 M faces, and rasterising that 24 times over in PIL takes longer
    than the generation did; a deterministic tenth of the triangles fixes the
    orientation just as well, and the winner is then drawn in full.
    """
    v = np.asarray(mesh.vertices, np.float64)
    f = np.asarray(mesh.faces, np.int64)
    if len(f) == 0:
        return np.zeros((n, n), bool)
    if max_faces and len(f) > max_faces:
        f = f[::max(1, len(f) // max_faces)]
    ia, sa, ib, sb = AXES[axes]
    u, w = v[:, ia] * sa, v[:, ib] * sb
    du, dw = u.max() - u.min(), w.max() - w.min()
    span = max(du, dw) or 1.0
    k = (n - 2) / span
    px = np.stack([(u - u.min()) * k + 1 + (n - du * k) / 2 - 1,
                   (w - w.min()) * k + 1 + (n - dw * k) / 2 - 1], 1)

    im = Image.new("1", (n, n), 0)
    d = ImageDraw.Draw(im)
    for t in px[f]:
        d.polygon([tuple(p) for p in t], fill=1)
    return np.asarray(im, bool)


def best_silhouette(mesh, inp, n: int = RASTER, search_n=128, search_faces=25_000):
    """Try all 24 framings cheaply, then draw the winner properly.

    Different backends hand back different canonical frames -- glTF is y-up,
    TripoSR is z-up -- and guessing wrong reads as a total failure, so this is
    calibrated per mesh rather than assumed.
    """
    small = np.asarray(Image.fromarray(inp.astype(np.uint8) * 255)
                       .resize((search_n, search_n), Image.NEAREST)) > 127
    best = None
    for a in AXES:
        s = mesh_silhouette(mesh, a, search_n, max_faces=search_faces)
        iou = float((small & s).sum()) / max(float((small | s).sum()), 1.0)
        if best is None or iou > best[1]:
            best = (a, iou)
    got = mesh_silhouette(mesh, best[0], n)
    iou = float((inp & got).sum()) / max(float((inp | got).sum()), 1.0)
    return got, best[0], iou


def compare(inp: np.ndarray, got: np.ndarray, size_in: float) -> dict:
    from scipy import ndimage as ndi
    inter = float((inp & got).sum())
    union = float((inp | got).sum()) or 1.0

    # a pixel's distance to the nearest background pixel is its local half-width;
    # scale it into inches through the object's printed size
    edt = ndi.distance_transform_edt(inp) * (size_in / inp.shape[0])
    thin = inp & (edt < NOZZLE_IN)
    nthin = float(thin.sum())

    # holes: background that the input silhouette encloses. filling the outside
    # from the border and inverting leaves exactly the enclosed gaps.
    outside = np.zeros_like(inp)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside = ndi.binary_propagation(outside, mask=~inp)
    holes = (~inp) & (~outside)
    nholes = float(holes.sum())

    return {
        "iou": round(inter / union, 4),
        "recall": round(inter / max(float(inp.sum()), 1.0), 4),
        "precision": round(inter / max(float(got.sum()), 1.0), 4),
        "thin_frac": round(nthin / max(float(inp.sum()), 1.0), 4),
        "thin_recall": round(float((thin & got).sum()) / max(nthin, 1.0), 4),
        # 0 = every gap between fragments survives, 1 = all of them filled in
        "interstice_fill": round(float((holes & got).sum()) / max(nholes, 1.0), 4),
        "interstice_frac": round(nholes / max(float(inp.size), 1.0), 4),
        "nozzle_in": round(NOZZLE_IN, 5),
    }


def mesh_stats(m) -> dict:
    comps = m.split(only_watertight=False)
    vols = [abs(float(c.volume)) for c in comps] or [abs(float(m.volume))]
    e = m.extents
    return {
        "faces": int(len(m.faces)), "verts": int(len(m.vertices)),
        "watertight": bool(m.is_watertight), "bodies": int(len(comps)),
        "volume": round(float(m.volume), 6),
        "fill": round(float(abs(m.volume)) / max(float(np.prod(e)), 1e-12), 4),
        "largest_body_share": round(max(vols) / max(sum(vols), 1e-12), 4),
    }
