r"""
tile_triposr.py -- hallucinate a volume from each tile assemblage.

MUST be run with the TripoSR venv, which is separate from this project's:

    "../tools/triposr/.venv/Scripts/python.exe" tile_triposr.py

It feeds sparse_fit's own matte to the model rather than letting rembg guess.
rembg is trained on photographs and a fitted assemblage is not one, and the
matte is exact -- so background removal is skipped and the alpha is composited
onto the 0.5 grey the model expects, through TripoSR's own resize_foreground so
the framing matches what it was trained on.

Question 1 of the tile is whether the meshes hold the fragment silhouettes at
printed size, so every mesh is measured rather than eyeballed:

  iou             the mesh's silhouette from the input camera against the input
                  alpha, both normalised to their own bounding box, so this is
                  shape agreement and not a scale artefact
  thin_recall     the same, restricted to input pixels lying in parts thinner
                  than one printed nozzle width at this object's real size --
                  which is the actual question. A mesh can score a fine iou by
                  getting the mass right and losing every tracery.
  thin_frac       how much of the input is that thin in the first place

The camera is TripoSR's own: right-handed, x back, y right, z up, input view at
azimuth 0. So the input silhouette is (y, -z).

    ../tools/triposr/.venv/Scripts/python.exe tile_triposr.py --mc-resolution 256
    ../tools/triposr/.venv/Scripts/python.exe tile_triposr.py --only 3 \
        --mc-resolution 128,256,384,512 --out data/gap_space/site_tile/mc_sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import os                                                     # noqa: E402

TRIPOSR = Path(os.environ.get(
    "TRIPOSR_DIR",
    Path(__file__).resolve().parent.parent / "tools" / "triposr"))
if not (TRIPOSR / "tsr").is_dir():
    raise SystemExit(f"TripoSR not found at {TRIPOSR}; set TRIPOSR_DIR")
sys.path.insert(0, str(TRIPOSR))

from tsr.system import TSR                                    # noqa: E402
from tsr.utils import resize_foreground                       # noqa: E402

import tile_measure as TM                                     # noqa: E402

RASTER = TM.RASTER


def prepare(matte_path: Path, ratio: float) -> Image.Image:
    """RGBA matte -> the RGB-on-grey square the model was trained to see."""
    im = Image.open(matte_path).convert("RGBA")
    im = resize_foreground(im, ratio)
    a = np.asarray(im, np.float32) / 255.0
    rgb = a[:, :, :3] * a[:, :, 3:4] + (1 - a[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8)), a[:, :, 3]



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemblages", default="data/gap_space/site_tile/assemblages")
    ap.add_argument("--points", default="data/gap_space/site_tile/points.json")
    ap.add_argument("--out", default="data/gap_space/site_tile/meshes")
    ap.add_argument("--protocol", default="drift")
    ap.add_argument("--mc-resolution", default="256",
                    help="comma list; every value is extracted from one forward pass")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--foreground-ratio", type=float, default=0.85)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    res_list = [int(r) for r in a.mc_resolution.split(",") if r.strip()]
    src = Path(a.assemblages)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    man = json.loads(Path(a.points).read_text(encoding="utf-8"))
    dirs = sorted(d for d in src.iterdir()
                  if d.is_dir() and (d / f"{a.protocol}_matte.png").exists())
    if a.only:
        dirs = dirs[:a.only]
    print(f"{len(dirs)} assemblages, mc resolutions {res_list}")

    device = a.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = TSR.from_pretrained("stabilityai/TripoSR",
                                config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(a.chunk_size)
    model.to(device)
    t_load = time.time() - t0
    print(f"model loaded in {t_load:.1f}s on {device}")

    import trimesh
    rows, t_run = [], time.time()
    for n, d in enumerate(dirs):
        side = json.loads((d / f"{a.protocol}.json").read_text(encoding="utf-8"))
        size_in = float(side["size_in"])
        odir = out / d.name
        odir.mkdir(parents=True, exist_ok=True)

        done = all((odir / f"mesh_mc{r}.obj").exists() for r in res_list)
        if done and not a.force:
            print(f"  [{n+1}/{len(dirs)}] {d.name}  already present")
            rows.append(json.loads((odir / "mesh.json").read_text(encoding="utf-8")))
            continue

        img, alpha = prepare(d / f"{a.protocol}_matte.png", a.foreground_ratio)
        img.save(odir / "input.png")
        inp = TM.rasterise_alpha(alpha)

        t1 = time.time()
        with torch.no_grad():
            codes = model([img], device=device)
        t_fwd = time.time() - t1

        rec = {"dir": d.name, "protocol": a.protocol,
               "point_mode": side["point_mode"], "reserved": side["reserved"],
               "x": side["x"], "y": side["y"],
               "plan_x_in": side["plan_x_in"], "plan_y_in": side["plan_y_in"],
               "size_in": size_in, "size_ft": side["size_ft"],
               "clash": side["clash"], "periods": side["periods"],
               "forward_s": round(t_fwd, 2), "by_resolution": {}}

        for r in res_list:
            t2 = time.time()
            mesh = model.extract_mesh(codes, True, resolution=r)[0]
            t_ex = time.time() - t2
            path = odir / f"mesh_mc{r}.obj"
            mesh.export(path)
            tm = trimesh.load(path, process=False)
            # TripoSR's frame is fixed and known, so the projection is
            # pinned rather than searched -- see tile_measure.AXES
            sil = TM.mesh_silhouette(tm, TM.TRIPOSR_AXES)
            m = TM.compare(inp, sil, size_in)
            m.update(faces=int(len(tm.faces)), verts=int(len(tm.vertices)),
                     watertight=bool(tm.is_watertight),
                     bodies=int(tm.body_count),
                     volume=round(float(tm.volume), 6),
                     extract_s=round(t_ex, 2),
                     mb=round(path.stat().st_size / 1e6, 2))
            rec["by_resolution"][str(r)] = m
            Image.fromarray(np.concatenate(
                [inp, sil], 1).astype(np.uint8) * 255).save(
                    odir / f"silhouette_mc{r}.png")
            print(f"  [{n+1}/{len(dirs)}] {d.name} mc{r}: "
                  f"iou {m['iou']:.3f}  thin_recall {m['thin_recall']:.3f} "
                  f"(thin {100*m['thin_frac']:.0f}%)  {m['faces']} faces  "
                  f"wt {m['watertight']}  bodies {m['bodies']}  {t_ex:.1f}s",
                  flush=True)

        (odir / "mesh.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
        del codes
        torch.cuda.empty_cache()

    total = time.time() - t_run
    summary = {
        "protocol": a.protocol, "resolutions": res_list,
        "n": len(rows), "device": device,
        "model_load_s": round(t_load, 1),
        "total_s": round(total, 1),
        "per_object_s": round(total / max(len(rows), 1), 1),
        "chunk_size": a.chunk_size, "foreground_ratio": a.foreground_ratio,
        "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)
                        if device.startswith("cuda") else None,
        "scale": man["scale"], "in_per_unit": man["in_per_unit"],
        "meshes": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for r in res_list:
        v = [m["by_resolution"][str(r)] for m in rows if str(r) in m["by_resolution"]]
        if not v:
            continue
        print(f"\nmc {r}: iou {np.mean([x['iou'] for x in v]):.3f}  "
              f"thin_recall {np.mean([x['thin_recall'] for x in v]):.3f}  "
              f"faces {int(np.mean([x['faces'] for x in v]))}  "
              f"watertight {sum(x['watertight'] for x in v)}/{len(v)}  "
              f"single-body {sum(x['bodies']==1 for x in v)}/{len(v)}")
    print(f"\n{len(rows)} meshes in {total/60:.1f}m "
          f"({total/max(len(rows),1):.1f}s each), peak VRAM "
          f"{summary['vram_peak_mb']} MB\nwrote {out/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
