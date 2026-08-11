r"""
tile_score.py -- score any directory of per-point meshes against its assemblages.

Backend-agnostic on purpose. Point it at TripoSR's output or the paid API's and
it applies the same ruler, so "the hosted model is better" stops being a belief
and becomes a table. Runs in the project environment; no GPU, no model.

The measure that matters here is `interstice_fill`. An assemblage reads as an
assemblage because of the gaps BETWEEN its fragments. TripoSR closes them. A
mesh can score a fine iou while having fused the whole composition into one
rock, and that is exactly what the local model does.

    python tile_score.py --meshes data/gap_space/site_tile/meshes --pattern "mesh_mc128.obj"
    python tile_score.py --meshes ... --pattern "mesh_tripo.obj" --label tripo
    python tile_score.py --compare data/gap_space/site_tile/scores/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import gap_space_core as G
import tile_measure as TM

OUT = G.OUT / "site_tile"


def score_dir(meshes: Path, assemblages: Path, pattern: str, protocol: str,
              axes: str | None, limit: int = 0) -> list:
    rows = []
    dirs = sorted(d for d in meshes.iterdir() if d.is_dir() and (d / pattern).exists())
    if limit:
        dirs = dirs[:limit]
    for d in dirs:
        a = assemblages / d.name
        matte = a / f"{protocol}_matte.png"
        side = a / f"{protocol}.json"
        if not matte.exists() or not side.exists():
            continue
        meta = json.loads(side.read_text(encoding="utf-8"))
        alpha = np.asarray(Image.open(matte).convert("RGBA"),
                           np.float32)[:, :, 3] / 255.0
        inp = TM.rasterise_alpha(alpha)
        m = trimesh.load(d / pattern, process=False)
        if axes:
            sil, used = TM.mesh_silhouette(m, axes), axes
        else:
            sil, used, _ = TM.best_silhouette(m, inp)
        r = TM.compare(inp, sil, float(meta["size_in"]))
        r.update(TM.mesh_stats(m))
        r.update(dir=d.name, point_mode=meta["point_mode"], projection=used,
                 size_in=meta["size_in"])
        rows.append(r)
        print(f"  {d.name:34s} iou {r['iou']:.3f}  thin {r['thin_recall']:.3f}  "
              f"gaps kept {1-r['interstice_fill']:.2f}  {r['faces']:7d} faces  "
              f"bodies {r['bodies']:4d}  fill {r['fill']:.3f}")
    return rows


KEYS = ["iou", "thin_recall", "interstice_fill", "fill", "faces", "bodies",
        "largest_body_share"]


def digest(label, rows):
    d = {"label": label, "n": len(rows),
         "watertight": sum(r["watertight"] for r in rows)}
    for k in KEYS:
        d[k] = round(float(np.mean([r[k] for r in rows])), 4)
    d["gaps_kept"] = round(1 - d["interstice_fill"], 4)
    return d


def table(digests):
    hdr = (f"{'backend':22s} {'n':>3s} {'iou':>6s} {'thin':>6s} {'gaps kept':>10s} "
           f"{'fill':>6s} {'faces':>8s} {'bodies':>7s} {'watertight':>11s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for d in digests:
        print(f"{d['label']:22s} {d['n']:3d} {d['iou']:6.3f} {d['thin_recall']:6.3f} "
              f"{d['gaps_kept']:10.3f} {d['fill']:6.3f} {int(d['faces']):8,d} "
              f"{d['bodies']:7.1f} {d['watertight']:6d}/{d['n']:<4d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", default=None)
    ap.add_argument("--assemblages", default=str(OUT / "assemblages"))
    ap.add_argument("--pattern", default="mesh_mc128.obj")
    ap.add_argument("--protocol", default="drift")
    ap.add_argument("--axes", default=None,
                    help="fix the projection (yz for TripoSR); default searches "
                         "all six and keeps the best, which is right for a "
                         "backend whose canonical frame is unknown")
    ap.add_argument("--label", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "scores"))
    ap.add_argument("--compare", nargs="*", default=None,
                    help="score files to tabulate together instead of scoring")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.compare is not None:
        files = [Path(f) for pat in (a.compare or [str(out / "*.json")])
                 for f in sorted(glob.glob(pat))]
        table([json.loads(f.read_text(encoding="utf-8"))["digest"] for f in files])
        return 0

    if not a.meshes:
        raise SystemExit("--meshes is required unless --compare is used")
    label = a.label or f"{Path(a.meshes).name}/{Path(a.pattern).stem}"
    rows = score_dir(Path(a.meshes), Path(a.assemblages), a.pattern,
                     a.protocol, a.axes, a.limit)
    if not rows:
        raise SystemExit("nothing scored")
    dg = digest(label, rows)
    f = out / f"{label.replace('/', '__')}.json"
    f.write_text(json.dumps({"digest": dg, "meshes": rows}, indent=2),
                 encoding="utf-8")
    table([dg])
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
