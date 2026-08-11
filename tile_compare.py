r"""
tile_compare.py -- the hallucinated volume against the derived one, same ground.

Two routes exist to a solid from the same sparse_fit composition:

    TripoSR              infers a volume from the picture. Soft, closed, and
                         object-like; it does not know what a fragment is.
    assemblage_geometry  replays the placement and traces each fragment's alpha
                         into a closed curve with real arcs, then extrudes.
                         Prismatic, exact, drawing-like; every fragment survives
                         as its own solid and carries its period and four terms.

They are not competitors to be argued about in the abstract, so this runs both on
the same coordinates and measures them, then renders them side by side. The
geometry route's native reading is flat -- the composition is a plan -- but it is
stood upright here so the only difference on show is what depth each route
infers.

    python tile_compare.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import trimesh

import gap_space_core as G

OUT = G.OUT / "site_tile"


def load_geometry(path: Path, reading="curated"):
    """solids.obj holds both readings as named objects; keep one."""
    s = trimesh.load(path, process=False, split_object=True, group_material=False)
    if isinstance(s, trimesh.Trimesh):
        return s, 0
    keys = [k for k in s.geometry if k.startswith(reading + "__")]
    if not keys:
        keys = list(s.geometry)
    return trimesh.util.concatenate([s.geometry[k] for k in keys]), len(keys)


def stats(m, label):
    comps = m.split(only_watertight=False)
    vols = [abs(float(c.volume)) for c in comps] or [abs(float(m.volume))]
    e = m.extents
    horiz = max(e[0], e[1])
    return {
        "route": label,
        "faces": int(len(m.faces)), "verts": int(len(m.vertices)),
        "watertight": bool(m.is_watertight), "bodies": int(len(comps)),
        "extents": np.round(e / horiz, 4).tolist(),
        "depth_over_width": round(float(min(e[0], e[1]) / horiz), 4),
        "fill": round(float(abs(m.volume)) / max(float(np.prod(e)), 1e-12), 4),
        "largest_body_share": round(max(vols) / max(sum(vols), 1e-12), 4),
    }


def upright(m):
    """Composition plane from horizontal to vertical, so both routes show the
    assemblage as an elevation and the comparison is only about depth."""
    m = m.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(-90), [1, 0, 0]))
    return m


def norm(m, size=1.0):
    m = m.copy()
    e = m.extents
    m.apply_scale(size / max(e[0], e[1]))
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default=str(OUT / "geometry"))
    ap.add_argument("--meshes", default=str(OUT / "meshes_n32"))
    ap.add_argument("--mc", type=int, default=192)
    ap.add_argument("--reading", default="curated", choices=["curated", "true"])
    ap.add_argument("--out", default=str(OUT / "compare"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import tile_render as TR
    from PIL import Image

    gdirs = sorted(Path(a.geometry).glob("*__drift"))
    mdirs = list(Path(a.meshes).glob("*"))

    def coord(name):
        m = re.search(r"x(-?[\d.p]+)_y(-?[\d.p]+)", name)
        return (round(float(m.group(1).replace("p", ".")), 2),
                round(float(m.group(2).replace("p", ".")), 2)) if m else None

    by_xy = {coord(d.name): d for d in mdirs if coord(d.name)}

    rows, tiles = [], []
    for g in gdirs:
        xy = coord(g.name)
        md = by_xy.get(xy)
        if md is None or not (md / f"mesh_mc{a.mc}.obj").exists():
            print(f"  no TripoSR mesh for {g.name} at {xy}")
            continue
        geo, nobj = load_geometry(g / "solids.obj", a.reading)
        tri = trimesh.load(md / f"mesh_mc{a.mc}.obj", process=False)

        gg = norm(upright(geo))
        tt = norm(tri)
        sg, st = stats(gg, "assemblage_geometry"), stats(tt, "TripoSR")
        sg["fragments"] = nobj
        rows.append({"point": g.name, "x": xy[0], "y": xy[1],
                     "geometry": sg, "triposr": st})
        print(f"{g.name}")
        for s in (st, sg):
            print(f"   {s['route']:20s} {s['faces']:7d} faces  "
                  f"{s['bodies']:4d} bodies  wt {str(s['watertight']):5s}  "
                  f"depth/width {s['depth_over_width']:.3f}  "
                  f"fill {s['fill']:.3f}")

        pair = [TR.render(tt, size=520, azim=32, elev=22),
                TR.render(gg, size=520, azim=32, elev=22)]
        tiles.append(np.concatenate(pair, axis=0))

    if tiles:
        strip = np.concatenate(tiles, axis=1)
        Image.fromarray((strip * 255).astype(np.uint8)).save(out / "compare.png")
        print(f"\nwrote {out / 'compare.png'}  (top row TripoSR, "
              f"bottom row assemblage_geometry)")

    def mean(route, key):
        return round(float(np.mean([r[route][key] for r in rows])), 4)

    summary = {
        "mc_resolution": a.mc, "reading": a.reading, "n": len(rows),
        "means": {
            "triposr": {k: mean("triposr", k) for k in
                        ("faces", "bodies", "depth_over_width", "fill",
                         "largest_body_share")},
            "assemblage_geometry": {k: mean("geometry", k) for k in
                                    ("faces", "bodies", "depth_over_width",
                                     "fill", "largest_body_share")},
        },
        "watertight": {
            "triposr": sum(r["triposr"]["watertight"] for r in rows),
            "assemblage_geometry": sum(r["geometry"]["watertight"] for r in rows),
        },
        "points": rows,
    }
    (out / "compare.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary["means"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
