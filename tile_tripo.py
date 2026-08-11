r"""
tile_tripo.py -- volumes from the paid Tripo API, as a drop-in for tile_triposr.

Same inputs, same output layout, same measurements, so the two backends can be
read against each other and `tile_model.py` does not care which produced the
meshes. Local TripoSR loses the gaps between fragments and erodes the surfaces;
the hosted models keep the planes and the fragment boundaries. This asks the
hosted one, and scores it with the same ruler so that claim becomes a number.

RUN IT WITH THE 3D-TOOLS VENV, which has the SDK and trimesh:

    "../tools/triposr/.venv/Scripts/python.exe" tile_tripo.py --probe 3

API credits are NOT Studio credits. `platform.tripo3d.ai` (this) and
`tripo3d.ai` (the web app) are separate business lines with separate balances,
and a Studio subscription buys nothing here -- a Studio Pro account with 3,000
credits a month still reads `balance: 0` on this endpoint and every task comes
back 403 / code 2010. API credits are pay-as-you-go, bought on the platform site.

THIS SPENDS MONEY. Image-to-model is charged per task, so:

  * nothing runs without `--spend`, which is a deliberate second gate
  * `--probe N` does N points only, for deciding a model version cheaply
  * completed points are skipped on re-run, so a resumed run costs nothing for
    work already paid for
  * the account balance is read before and after and recorded

The API key is read from the TRIPO_API_KEY environment variable and is never
written to disk or into a sidecar. Set it yourself:

    setx TRIPO_API_KEY "your-key"        (then reopen the shell)

Texture and PBR are OFF by default. The tile is printed in white and the
geometry is booleaned; textures cost extra per task and would be thrown away.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import tile_measure as TM

MODELS = ["P1-20260311", "v3.1-20260211", "v3.0-20250812", "Turbo-v1.0-20250506",
          "v2.5-20250123", "v2.0-20240919"]

# deliberately never a command-line argument: it would land in shell history and
# in the process list, where anyone on the machine can read it
NO_KEY = (
    "No Tripo key found. Either of these works:\n"
    '  1. setx TRIPO_API_KEY "your-key"  -- then RESTART the app. setx writes\n'
    "     the registry, and a process already running keeps its old copy.\n"
    "  2. put the key on the first line of ~/.tripo_key  -- takes effect at once.\n"
    "This script never accepts a key as an argument and never writes one out."
)


def read_key() -> tuple[str, str]:
    """The key, and where it came from. Never printed, never logged.

    `setx` writes to the registry, and a process that is already running does not
    see it -- so on Windows the environment variable only works after the parent
    application is restarted. A key file avoids that, which is why both are
    supported. Whichever is found first wins.
    """
    v = os.environ.get("TRIPO_API_KEY", "").strip()
    if v:
        return v, "TRIPO_API_KEY"
    p = os.environ.get("TRIPO_API_KEY_FILE", "").strip()
    cands = ([Path(p)] if p else []) + [
        Path.home() / ".tripo_key",
        Path(__file__).resolve().parent / ".tripo_key",
    ]
    for c in cands:
        try:
            if c.is_file():
                v = c.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                if v:
                    return v, str(c)
        except Exception:
            continue
    return "", ""


def prepare(matte_path: Path, out_png: Path, pad=0.06) -> float:
    """Trim the matte to its own bounds with a little air and save RGBA.

    The API takes a transparent PNG, which is the whole advantage of having
    sparse_fit's alpha: no background removal has to guess at an assemblage.
    Framing is normalised so every point arrives at the model the same way.
    """
    im = Image.open(matte_path).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.nonzero(a > 127)
    if len(ys) == 0:
        raise ValueError(f"empty matte: {matte_path}")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    side = int(max(crop.size) * (1 + 2 * pad))
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))
    if side > 1024:
        sq = sq.resize((1024, 1024), Image.LANCZOS)
    sq.save(out_png)
    return float((np.asarray(sq)[:, :, 3] > 127).mean())


def convert_only(args) -> int:
    """Turn already-downloaded GLBs into scored OBJs, paying nothing.

    Generation and conversion are separable, and they broke apart the hard way:
    TripoSR pins trimesh==4.0.5, which calls `ndarray.ptp()`, removed in NumPy 2 --
    so the first paid run generated and downloaded fine and then died on the
    conversion. The GLBs were already on disk. This recovers them, and it runs in
    the project environment, whose trimesh is current.
    """
    out = Path(args.out)
    src = Path(args.assemblages)
    done, failed = 0, []
    for odir in sorted(d for d in out.iterdir() if d.is_dir()):
        if (odir / "mesh_tripo.obj").exists() and not args.force:
            continue
        glbs = sorted(odir.glob("*.glb"))
        # tripo.json is written only after the download returns, so it is the
        # signal that the GLB is complete. Converting alongside a running
        # generation would otherwise pick up a half-written file.
        if not glbs or not (odir / "tripo.json").exists():
            continue
        glb = next((g for g in glbs if "pbr" in g.name), glbs[0])
        a = src / odir.name
        try:
            meta = json.loads((a / f"{args.protocol}.json").read_text(encoding="utf-8"))
            alpha = np.asarray(Image.open(a / f"{args.protocol}_matte.png")
                               .convert("RGBA"), np.float32)[:, :, 3] / 255.0
            inp = TM.rasterise_alpha(alpha)
            # never write the raw mesh out: 1.5 M faces is a 112 MB OBJ and
            # 95 of those is 10 GB of intermediate nobody reads. The GLB beside
            # it is the archival copy, at 43 MB.
            raw = to_obj(glb, None)
            m = raw
            if args.solidify:
                import trimesh
                m = solidify(raw, args.solidify, args.decimate)
                m.export(odir / "mesh_tripo.obj")
            else:
                raw.export(odir / "mesh_tripo.obj")
            sil, axes, _ = TM.best_silhouette(m, inp)
            rec = TM.compare(inp, sil, float(meta["size_in"]))
            rec.update(TM.mesh_stats(m))
            rec.update(dir=odir.name, ok=True, projection=axes, glb=glb.name,
                       recovered=True)
            Image.fromarray(np.concatenate([inp, sil], 1).astype(np.uint8) * 255)\
                 .save(odir / "silhouette.png")
            (odir / "tripo.json").write_text(json.dumps(rec, indent=2),
                                             encoding="utf-8")
            done += 1
            print(f"  {odir.name:34s} iou {rec['iou']:.3f}  "
                  f"thin {rec['thin_recall']:.3f}  "
                  f"gaps kept {1-rec['interstice_fill']:.2f}  "
                  f"{rec['faces']:7d} faces  bodies {rec['bodies']:3d}  "
                  f"wt {rec['watertight']}")
        except Exception as ex:
            failed.append((odir.name, f"{type(ex).__name__}: {ex}"))
    print(f"\nconverted {done}, failed {len(failed)}")
    for f in failed:
        print("  ", *f)
    return 0


def solidify(m, divisions=256, target_faces=40000, log=print):
    """Open shell soup -> one watertight solid, at a resolution finer than the
    nozzle, with the gaps between fragments intact.

    The hosted models return roughly 1.5 M faces in 600-800 *open* components --
    none of them watertight, so `manifold3d` refuses the lot and hole filling
    does nothing. Voxelising and re-surfacing closes them by construction. On a
    1.25 in printed object, 256 divisions is a 0.12 mm voxel against a 0.4 mm
    nozzle, so nothing printable is lost, and any gap wider than a voxel
    survives -- which is the entire reason for using this model over TripoSR.

    Decimation then brings 587 k faces back to something the boolean can hold.
    `preservetopology=True` is not optional: without it quadric collapse
    introduces non-manifold edges with no open edges, so `is_watertight` reports
    False and hole filling silently does nothing. That trap is already recorded
    in this project and it cost an afternoon once.
    """
    import trimesh
    v = m.voxelized(pitch=float(max(m.extents)) / divisions).fill().marching_cubes
    v.update_faces(v.nondegenerate_faces())
    v.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(v)
    if not target_faces or len(v.faces) <= target_faces:
        return v
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(v.vertices, v.faces), "m")
    ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                    targetfacenum=int(target_faces), preservetopology=True,
                    preservenormal=True, planarquadric=True)
    g = ms.current_mesh()
    d = trimesh.Trimesh(g.vertex_matrix().copy(), g.face_matrix().copy(),
                        process=True)
    trimesh.repair.fix_normals(d)
    if not d.is_watertight:
        log("    decimation opened the mesh; keeping the undecimated solid")
        return v
    return d


def to_obj(src: Path, dst: Path):
    """GLB -> a single triangulated OBJ in this project's frame.

    Two corrections, both learned by looking at a mesh that seemed shattered and
    was not:

    * glTF is y-up and everything here is z-up, so the model arrives lying on its
      side. Rotating +90 degrees about x puts the composition upright, matching
      what TripoSR hands back and what tile_model expects.
    * `process=False` leaves every triangle with its own vertices, so nothing
      shares an edge, `body_count` reports hundreds of pieces and the boolean
      refuses the mesh. Merging is required, not tidying.
    """
    import trimesh
    s = trimesh.load(src, force="mesh")          # process=True: merge vertices
    if hasattr(s, "geometry"):
        s = trimesh.util.concatenate(list(s.geometry.values()))
    s.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    trimesh.repair.fix_normals(s)
    if dst is not None:
        s.export(dst)
    return s


async def one(client, sem, d, args, inp, size_in, odir):
    from tripo3d import TaskStatus
    async with sem:
        t0 = time.time()
        src = odir / "input.png"
        cov = prepare(d / f"{args.protocol}_matte.png", src)

        kw = dict(model_version=args.model_version, texture=args.texture,
                  pbr=args.pbr, quad=False)
        if args.face_limit:
            kw["face_limit"] = args.face_limit
        if args.geometry_quality:
            kw["geometry_quality"] = args.geometry_quality
        if args.generate_parts:
            kw["generate_parts"] = True
        if args.orientation:
            kw["orientation"] = args.orientation

        task_id = await client.image_to_model(image=str(src), **kw)
        task = await client.wait_for_task(task_id, polling_interval=3.0)
        if task.status != TaskStatus.SUCCESS:
            return {"dir": d.name, "ok": False, "task_id": task_id,
                    "status": str(task.status), "error": task.error_msg}

        files = await client.download_task_models(task, str(odir))
        glb = files.get("pbr_model") or files.get("model") or files.get("base_model")
        if not glb:
            return {"dir": d.name, "ok": False, "task_id": task_id,
                    "status": "no model file", "files": files}

        if args.download_only:
            rec = {"dir": d.name, "ok": True, "task_id": task_id, "glb": Path(glb).name,
                   "model_version": args.model_version, "downloaded": True,
                   "seconds": round(time.time() - t0, 1)}
            (odir / "tripo.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"  {d.name:34s} downloaded  {rec['seconds']:.0f}s", flush=True)
            return rec
        m = to_obj(Path(glb), odir / "mesh_tripo.obj")
        sil, axes, _ = TM.best_silhouette(m, inp)
        rec = TM.compare(inp, sil, size_in)
        rec.update(TM.mesh_stats(m))
        rec.update(dir=d.name, ok=True, task_id=task_id,
                   model_version=args.model_version, projection=axes,
                   matte_coverage=round(cov, 4), seconds=round(time.time() - t0, 1))
        Image.fromarray(np.concatenate([inp, sil], 1).astype(np.uint8) * 255)\
             .save(odir / "silhouette.png")
        (odir / "tripo.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  {d.name:34s} iou {rec['iou']:.3f}  thin {rec['thin_recall']:.3f}  "
              f"gaps kept {1-rec['interstice_fill']:.2f}  {rec['faces']:6d} faces  "
              f"wt {str(rec['watertight']):5s}  {rec['seconds']:.0f}s", flush=True)
        return rec


async def run(args) -> int:
    from tripo3d import TripoClient

    key, key_src = read_key()
    src = Path(args.assemblages)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(args.points).read_text(encoding="utf-8"))

    dirs = sorted(d for d in src.iterdir()
                  if d.is_dir() and (d / f"{args.protocol}_matte.png").exists())
    if not args.include_reserved:
        # Site A is a cleared plot. No volume is placed there, so generating one
        # is 20 credits spent on something tile_model throws away.
        keep = []
        for d in dirs:
            try:
                if json.loads((d / f"{args.protocol}.json")
                              .read_text(encoding="utf-8")).get("reserved"):
                    continue
            except Exception:
                pass
            keep.append(d)
        if len(keep) != len(dirs):
            print(f"skipping {len(dirs)-len(keep)} reserved point(s) -- nothing "
                  f"is built on them")
        dirs = keep
    if args.probe:
        # spread the probe across the point modes rather than taking the first N
        # alphabetically, which would be all clash points
        pick, seen = [], set()
        for d in dirs:
            mode = d.name.split("_")[0]
            if mode not in seen:
                seen.add(mode)
                pick.append(d)
        dirs = (pick + [d for d in dirs if d not in pick])[:args.probe]

    def generated(d):
        # a downloaded GLB counts: --download-only leaves no OBJ, and a resumed
        # run must never pay twice for a task that already succeeded
        o = out / d.name
        return (o / "mesh_tripo.obj").exists() or any(o.glob("*.glb"))

    todo = [d for d in dirs if args.force or not generated(d)]
    done = len(dirs) - len(todo)
    if args.max_tasks:
        todo = todo[:args.max_tasks]

    est = len(todo) * args.credits_per_task
    print(f"{len(dirs)} points, {done} already generated, {len(todo)} to run")
    print(f"model {args.model_version}, texture {args.texture}, pbr {args.pbr}")
    print(f"estimated cost: {len(todo)} x {args.credits_per_task} credits "
          f"= {est} credits (~${est/100:.2f} at 100 credits = $1)")

    if not todo:
        print("nothing to do")
        return 0
    if not args.spend:
        print("\nDRY RUN. Nothing was sent and nothing was charged.\n"
              "Re-run with --spend to actually generate."
              + (f"\nkey found via {key_src}" if key else "\n" + NO_KEY))
        return 0
    if not key:
        print(NO_KEY, file=sys.stderr)
        return 2

    async with TripoClient(api_key=key) as client:
        try:
            b0 = (await client.get_balance()).balance
            print(f"balance before: {b0:.0f} credits")
        except Exception as ex:
            b0 = None
            print(f"(balance unavailable: {type(ex).__name__})")

        sem = asyncio.Semaphore(args.concurrency)
        jobs, t0 = [], time.time()
        for d in todo:
            side = json.loads((d / f"{args.protocol}.json").read_text(encoding="utf-8"))
            odir = out / d.name
            odir.mkdir(parents=True, exist_ok=True)
            alpha = np.asarray(Image.open(d / f"{args.protocol}_matte.png")
                               .convert("RGBA"), np.float32)[:, :, 3] / 255.0
            inp = TM.rasterise_alpha(alpha)
            jobs.append(one(client, sem, d, args, inp,
                            float(side["size_in"]), odir))

        rows = []
        for r in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(r, Exception):
                print(f"  FAILED: {type(r).__name__}: {r}")
                rows.append({"ok": False, "error": f"{type(r).__name__}: {r}"})
            else:
                rows.append(r)
        elapsed = time.time() - t0

        b1 = None
        try:
            b1 = (await client.get_balance()).balance
        except Exception:
            pass

    ok = [r for r in rows if r.get("ok")]
    rejected = [r for r in rows if not r.get("ok")]
    summary = {
        "backend": "tripo-api", "model_version": args.model_version,
        "protocol": args.protocol, "texture": args.texture, "pbr": args.pbr,
        "face_limit": args.face_limit, "geometry_quality": args.geometry_quality,
        "generate_parts": args.generate_parts,
        "n_requested": len(todo), "n_ok": len(ok),
        "n_rejected": len(rejected), "rejected": rejected,
        "already_present": done,
        "elapsed_s": round(elapsed, 1),
        "per_object_s": round(elapsed / max(len(todo), 1), 1),
        "concurrency": args.concurrency,
        "balance_before": b0, "balance_after": b1,
        "credits_spent": (round(b0 - b1, 2) if (b0 is not None and b1 is not None)
                          else None),
        "scale": man["scale"], "in_per_unit": man["in_per_unit"],
        "meshes": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    scored = [r for r in ok if "iou" in r]
    if scored:
        def mean(k):
            return float(np.mean([r[k] for r in scored]))
        print(f"\n{len(ok)}/{len(todo)} succeeded in {elapsed/60:.1f}m "
              f"({elapsed/max(len(todo),1):.0f}s each at concurrency {args.concurrency})")
        print(f"  iou {mean('iou'):.3f}   thin_recall {mean('thin_recall'):.3f}   "
              f"gaps kept {1-mean('interstice_fill'):.3f}")
        print(f"  faces {int(mean('faces')):,}   bodies {mean('bodies'):.1f}   "
              f"fill {mean('fill'):.3f}   "
              f"watertight {sum(r['watertight'] for r in scored)}/{len(scored)}")
    if summary["credits_spent"] is not None:
        print(f"  credits spent: {summary['credits_spent']:.0f} "
              f"(~${summary['credits_spent']/100:.2f}), "
              f"{b1:.0f} left")
    print(f"wrote {out/'summary.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assemblages", default="data/gap_space/site_tile/assemblages")
    ap.add_argument("--points", default="data/gap_space/site_tile/points.json")
    ap.add_argument("--out", default="data/gap_space/site_tile/meshes_tripo")
    ap.add_argument("--protocol", default="drift")
    ap.add_argument("--model-version", default="P1-20260311", choices=MODELS)
    ap.add_argument("--texture", action="store_true",
                    help="off by default: the tile prints in white and the "
                         "geometry is booleaned, so a texture is paid for and "
                         "thrown away")
    ap.add_argument("--pbr", action="store_true")
    ap.add_argument("--face-limit", type=int, default=None)
    ap.add_argument("--geometry-quality", default=None,
                    choices=["standard", "detailed"])
    ap.add_argument("--orientation", default=None,
                    choices=["default", "align_image"])
    ap.add_argument("--generate-parts", action="store_true",
                    help="ask for the model split into parts. Worth testing: the "
                         "whole complaint against TripoSR is that it fuses the "
                         "fragments into one rock.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--probe", type=int, default=0,
                    help="run only N points, one per point mode, for choosing a "
                         "model version cheaply")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--credits-per-task", type=int, default=20,
                    help="20 = image-to-3D with no texture, 30 with standard "
                         "texture, per developers.tripo3d.ai/en/pricing. For the "
                         "estimate only; the real figure comes from the balance "
                         "before and after, so a wrong rate shows up on run one")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--spend", action="store_true",
                    help="REQUIRED to send anything. Without it this is a dry run.")
    ap.add_argument("--include-reserved", action="store_true",
                    help="also generate the reserved plot, which is left empty")
    ap.add_argument("--download-only", action="store_true",
                    help="generate and download the GLB and stop. Conversion "
                         "needs pymeshlab, which lives in the project "
                         "environment, so it is a separate --convert-only pass")
    ap.add_argument("--solidify", type=int, default=256,
                    help="voxel divisions used to close the shell soup into a "
                         "solid. 0 disables. 256 is a 0.12 mm voxel on a 1.25 in "
                         "object, finer than the nozzle")
    ap.add_argument("--decimate", type=int, default=40000,
                    help="target faces after solidifying, so 95 of these can "
                         "still be booleaned. 0 disables")
    ap.add_argument("--convert-only", action="store_true",
                    help="score GLBs already downloaded and skip the API "
                         "entirely. Costs nothing, needs no key, and runs in the "
                         "project environment")
    a = ap.parse_args()
    if a.convert_only:
        return convert_only(a)
    return asyncio.run(run(a))


if __name__ == "__main__":
    raise SystemExit(main())
