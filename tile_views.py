r"""
tile_views.py -- look at a finished tile properly.

A 12 in tile shown 800 px wide gives each 1.2 in mass about 80 px, which is far
coarser than the eye gets at arm's length from the printed object. Judging
whether the thing reads as a city therefore needs a cut block at something near
print scale as well as the whole plate, so this makes both. The crop is a real
boolean cut, not a filtered point cloud, so the section face reads as a section.

    python tile_views.py --stl .../tile_density_upright_mc192.stl
    python tile_views.py --stl A.stl --stl-b B.stl        # two variants, paired
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import gap_space_core as G
import tile_render as TR

OUT = G.OUT / "site_tile"


def load(p):
    # the PLY beside the STL keeps the vertex indices as built; the STL round
    # trip drops degenerate faces and leaves the solid a few faces short of
    # closed, which kills the boolean cut
    ply = Path(p).with_suffix(".ply")
    src = ply if ply.exists() else Path(p)
    m = trimesh.load(src, process=(src.suffix.lower() == ".stl"),
                     merge_tex=True, merge_norm=True)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
    return m


def cut(m, x0, y0, x1, y1):
    """A real boolean cut when the mesh survives the STL round trip as a volume.

    It often does not: STL stores loose triangles, and merging 2.4 M of them back
    together at a tolerance leaves a handful of holes that hole-filling will not
    close. manifold3d then refuses the operand outright. The point crop is the
    honest fallback -- it shows the same block, just without a capped section
    face -- so the sheet still gets made.
    """
    if not m.is_watertight:
        return None
    zt = float(m.bounds[1][2]) + 2.0
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, zt * 3])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, 0])
    try:
        return trimesh.boolean.intersection([m, b], engine="manifold")
    except Exception as ex:
        print(f"  cut failed ({type(ex).__name__}); falling back to a point crop")
        return None


def label(img, text, h=34):
    from PIL import ImageDraw
    im = Image.fromarray((img * 255).astype(np.uint8)).convert("L")
    out = Image.new("L", (im.width, im.height + h), 255)
    out.paste(im, (0, h))
    ImageDraw.Draw(out).text((10, 9), text, fill=0)
    return np.asarray(out) / 255.0


def sheet(path, stls, region, size):
    cols = []
    for p in stls:
        m = load(p)
        name = Path(p).stem
        c = cut(m, *region)
        rc = (dict(region=region) if c is None else {})
        src = m if c is None else c
        tag = "crop" if c is None else "cut block"
        col = [label(TR.render(m, size=size, azim=35, elev=26), f"{name}  plate"),
               label(TR.render(m, size=size, azim=0, elev=90), f"{name}  plan"),
               label(TR.render(src, size=size, azim=35, elev=20, **rc),
                     f"{name}  {tag} {region[2]-region[0]:.0f} x "
                     f"{region[3]-region[1]:.0f} in"),
               label(TR.render(src, size=size, azim=0, elev=8, **rc),
                     f"{name}  elevation of the {tag}")]
        cols.append(np.concatenate(col, axis=0))
    w = min(c.shape[1] for c in cols)
    Image.fromarray((np.concatenate([c[:, :w] for c in cols], axis=1) * 255)
                    .astype(np.uint8)).save(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", action="append", required=True)
    ap.add_argument("--region", default="3.5,3.5,8.5,8.5",
                    help="plan crop in inches, x0,y0,x1,y1 -- default is the "
                         "5 in square around Site A at the tile centre")
    ap.add_argument("--size", type=int, default=820)
    ap.add_argument("--out", default=str(OUT / "views.png"))
    a = ap.parse_args()
    region = tuple(float(v) for v in a.region.split(","))
    p = sheet(Path(a.out), a.stl, region, a.size)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
