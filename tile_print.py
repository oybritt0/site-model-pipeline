r"""
tile_print.py -- split the tile to fit the bed, then get a real slicer estimate.

A 12 x 12 in tile is 304.8 mm square. `EXHIBITION_SITES.md` 5b assumes a 256 mm
bed, so the tile does NOT fit whole and has to be quartered; each quarter is
152.4 mm and fits with room. On a 350 mm machine it would fit in one piece, and
`--bed 350` says so. Either way the split is butt-jointed on the tile grid, which
is the same grid the full 40 x 40 in model is tiled on, so the seams are already
part of the object rather than an artefact of this test.

The estimate is not analytic. Bambu Studio ships a CLI and it is installed, so
each piece is really sliced and the gcode's own numbers are read back. Filament
weight is computed from length, because the system filament profile leaves
`filament_density` at 0 when the CLI resolves it -- 1.24 g/cm3 for PLA, stated
rather than assumed silently.

    python tile_print.py
    python tile_print.py --bed 350 --stl .../tile_density_upright_mc192.stl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh

import gap_space_core as G

OUT = G.OUT / "site_tile"
MM_PER_IN = 25.4
PLA_DENSITY = 1.24                       # g/cm3
FILAMENT_MM2 = math.pi * (1.75 / 2) ** 2   # cross-section of 1.75 mm filament

BAMBU = Path(r"C:\Program Files\Bambu Studio\bambu-studio.exe")
SYS = Path(r"C:\Users\rbritain\AppData\Roaming\BambuStudio\system\BBL")


def split_pieces(solid, tile_in, nx, ny, engine="manifold"):
    zt = float(solid.bounds[1][2]) + 1.0
    w, h = tile_in / nx, tile_in / ny
    out = []
    for j in range(ny):
        for i in range(nx):
            box = trimesh.creation.box(extents=[w, h, zt * 3])
            box.apply_translation([(i + 0.5) * w, (j + 0.5) * h, 0])
            p = trimesh.boolean.intersection([solid, box], engine=engine)
            if p is None or len(p.faces) == 0:
                continue
            # each piece prints on its own, so it must stand at the origin
            p.apply_translation(-p.bounds[0])
            p, dropped = drop_floaters(p)
            if dropped:
                print(f"  piece {i}{j}: dropped {dropped} severed fragment(s)")
            out.append(((i, j), p))
    return out


def drop_floaters(p, tol=1e-3):
    """Keep only what stands on the bed.

    Cutting the tile into quarters severs any mass that spanned a seam, and the
    off-cut is left floating inside the neighbouring piece. Those pieces cannot
    print, and at least one of them is enough to make the slicer return
    "nothing to be sliced" for the whole plate.
    """
    parts = p.split(only_watertight=False)
    if len(parts) <= 1:
        return p, 0
    z0 = float(p.bounds[0][2])
    grounded = [c for c in parts if float(c.bounds[0][2]) <= z0 + tol]
    if not grounded:
        return p, 0
    keep = trimesh.util.concatenate(grounded)
    return keep, len(parts) - len(grounded)


def slice_piece(stl: Path, workdir: Path, machine: str, process: str,
                filament: str, printable_height: float = 250.0) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    for f in workdir.glob("plate_*.gcode"):
        f.unlink()
    cmd = [str(BAMBU),
           "--load-settings", f"{SYS/'machine'/machine};{SYS/'process'/process}",
           "--load-filaments", str(SYS / "filament" / filament),
           # The BBL profile chain sets printable_area but never
           # printable_height, so the CLI falls back to roughly 100 mm -- a
           # tenth of the machine. Anything taller fails with "no object is
           # fully inside the print volume", which reads like a broken mesh and
           # is not: a plain 40 x 40 x 105 mm box fails the same way.
           "--printable-height", str(printable_height),
           "--slice", "0",
           "--export-3mf", "out.3mf",
           "--outputdir", str(workdir),
           str(stl)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    g = sorted(workdir.glob("plate_*.gcode"))
    if not g:
        return {"ok": False, "stderr": (r.stderr or r.stdout)[-800:],
                "cli_s": round(time.time() - t0, 1)}
    txt = g[0].read_text(encoding="utf-8", errors="ignore")
    head = txt[:4000] + txt[-40000:]

    def grab(pat, cast=float):
        m = re.search(pat, head)
        return cast(m.group(1)) if m else None

    length = grab(r"total filament length \[mm\]\s*:\s*([\d.]+)")
    t_model = grab(r"model printing time:\s*([^;\n]+)", str)
    t_total = grab(r"total estimated time:\s*([^;\n]+)", str)
    vol_cm3 = (length * FILAMENT_MM2 / 1000.0) if length else None
    return {
        "ok": True,
        "filament_mm": length,
        "filament_cm3": round(vol_cm3, 2) if vol_cm3 else None,
        "filament_g": round(vol_cm3 * PLA_DENSITY, 1) if vol_cm3 else None,
        "model_time": (t_model or "").strip(),
        "total_time": (t_total or "").strip(),
        "total_min": hhmm_to_min(t_total),
        "gcode_mb": round(g[0].stat().st_size / 1e6, 1),
        "cli_s": round(time.time() - t0, 1),
    }


def hhmm_to_min(s):
    if not s:
        return None
    h = re.search(r"(\d+)h", s)
    m = re.search(r"(\d+)m", s)
    sec = re.search(r"(\d+)s", s)
    return round((int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)
                 + (int(sec.group(1)) / 60 if sec else 0), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", default=None)
    ap.add_argument("--out", default=str(OUT / "print"))
    ap.add_argument("--bed", type=float, default=256.0, help="bed size in mm")
    ap.add_argument("--bed-margin", type=float, default=10.0)
    ap.add_argument("--machine", default="Bambu Lab P1S 0.4 nozzle.json")
    ap.add_argument("--process", default="0.20mm Standard @BBL X1C.json")
    ap.add_argument("--filament", default="Bambu PLA Basic @BBL X1C.json")
    ap.add_argument("--full-model-sqft", type=float, default=11.1)
    ap.add_argument("--printable-height", type=float, default=250.0,
                    dest="printable_height",
                    help="mm. Must be passed: the system profile omits it and "
                         "the CLI then defaults to about 100 mm")
    ap.add_argument("--no-slice", action="store_true")
    a = ap.parse_args()

    stl = Path(a.stl) if a.stl else sorted((OUT / "tile").glob("tile_*.stl"))[-1]
    rec = json.loads(stl.with_suffix(".json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Prefer the PLY beside the STL: it keeps the vertex indices tile_model
    # built, where the STL round trip loses a few degenerate faces and leaves
    # the solid open, which manifold3d refuses. Fall back to the STL, merged on
    # load, if only that exists.
    ply = stl.with_suffix(".ply")
    src = ply if ply.exists() else stl
    solid = trimesh.load(src, process=(src is stl),
                         merge_tex=True, merge_norm=True)
    if not solid.is_watertight:
        trimesh.repair.fill_holes(solid)
        trimesh.repair.fix_normals(solid)
    print(f"loaded {src.name}: {len(solid.faces):,} faces, "
          f"watertight {solid.is_watertight}, bodies {solid.body_count}")
    tile_in = rec["tile_in"]
    tile_mm = tile_in * MM_PER_IN
    usable = a.bed - a.bed_margin
    n = max(1, math.ceil(tile_mm / usable))
    print(f"{stl.name}: {tile_in} in = {tile_mm:.1f} mm on a {a.bed:.0f} mm bed "
          f"-> {n} x {n} pieces of {tile_mm/n:.1f} mm")

    pieces = split_pieces(solid, tile_in, n, n) if n > 1 else [((0, 0), solid)]
    rows = []
    for (i, j), p in pieces:
        # inches to millimetres: the STL is authored in inches, printers are not
        p = p.copy()
        p.apply_scale(MM_PER_IN)
        f = out / f"piece_{i}{j}.stl"
        p.export(f)
        r = {"piece": f"{i}{j}", "stl": f.name,
             "faces": int(len(p.faces)),
             "size_mm": np.round(p.extents, 2).tolist(),
             "volume_cm3": round(float(p.volume) / 1000.0, 2),
             "watertight": bool(p.is_watertight)}
        if not a.no_slice and BAMBU.exists():
            r["slice"] = slice_piece(f, out / f"_slice_{i}{j}",
                                     a.machine, a.process, a.filament,
                                     a.printable_height)
            s = r["slice"]
            print(f"  piece {i}{j}: {r['size_mm'][0]:.0f} x {r['size_mm'][1]:.0f} "
                  f"x {r['size_mm'][2]:.0f} mm, solid {r['volume_cm3']:.0f} cm3"
                  + (f"  ->  {s['filament_g']} g, {s['total_time']}"
                     if s.get("ok") else f"  -> SLICE FAILED: {s.get('stderr','')[:200]}"))
        else:
            print(f"  piece {i}{j}: {r['size_mm']} mm, "
                  f"solid {r['volume_cm3']:.0f} cm3")
        rows.append(r)

    ok = [r for r in rows if r.get("slice", {}).get("ok")]
    tot_g = sum(r["slice"]["filament_g"] for r in ok) if ok else None
    tot_min = sum(r["slice"]["total_min"] for r in ok) if ok else None
    tiles_full = a.full_model_sqft / (tile_in * tile_in / 144.0)

    summary = {
        "stl": stl.name, "scale": rec["scale"], "tile_in": tile_in,
        "bed_mm": a.bed, "pieces": f"{n}x{n}", "n_pieces": len(rows), "n_sliced": len(ok),
        "machine": a.machine, "process": a.process, "filament": a.filament,
        "pla_density_g_cm3": PLA_DENSITY,
        "solid_volume_cm3": round(float(solid.volume) * 16.3871, 1),
        "tile_filament_g": round(tot_g, 1) if tot_g else None,
        "tile_print_min": round(tot_min, 1) if tot_min else None,
        "tile_print_h": round(tot_min / 60, 1) if tot_min else None,
        "full_model_sqft": a.full_model_sqft,
        "full_model_tiles": round(tiles_full, 1),
        "full_model_filament_kg": round(tot_g * tiles_full / 1000, 2) if tot_g else None,
        "full_model_print_h": round(tot_min * tiles_full / 60, 1) if tot_min else None,
        "full_model_print_days_one_machine":
            round(tot_min * tiles_full / 60 / 24, 1) if tot_min else None,
        "pieces_detail": rows,
    }
    (out / "print_estimate.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    if tot_g:
        if len(ok) != len(rows):
            print(f"\n!! only {len(ok)} of {len(rows)} pieces sliced -- every "
                  f"total below is an UNDERSTATEMENT")
        print(f"\none tile: {tot_g:.0f} g, {tot_min/60:.1f} h "
              f"across {len(ok)} sliced of {len(rows)} pieces")
        print(f"{a.full_model_sqft} sq ft = {tiles_full:.1f} tiles: "
              f"{tot_g*tiles_full/1000:.1f} kg, {tot_min*tiles_full/60:.0f} h "
              f"({tot_min*tiles_full/60/24:.1f} days on one machine)")
    print(f"wrote {out / 'print_estimate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
