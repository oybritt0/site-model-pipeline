r"""
tile_assemblages.py -- compose one sparse_fit assemblage per tile coordinate.

Reads data/gap_space/site_tile/points.json and runs sparse_fit's own placement
at each coordinate, saving both readings:

    <slug>/<protocol>.png         the finished composition, on its vignette
    <slug>/<protocol>_matte.png   the same composition, straight-alpha, no ground
    <slug>/<protocol>.json        the sidecar

The matte is the whole point of the pass. TripoSR wants a clean foreground and
the vignette is a gradient, so it cannot be thresholded off after the fact --
but compose_fitted draws onto a transparent layer and the alpha was simply never
saved. sparse_fit now returns it.

Parameters match the existing corpus exactly (out_size 1080, work_res 1024,
margin 0.055, k 10, emphasis even) so these compositions are the same species as
everything already on disk. The reserved point is composed too: the void needs a
drawing even though no volume is placed there.

    python tile_assemblages.py
    python tile_assemblages.py --protocols drift,vestige,swarm --only 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import gap_space_core as G
import rift_atlas as R
import sparse_fit as SF

OUT = G.OUT / "site_tile"


def slug(p) -> str:
    return f"{p['mode']}_{p['i']:02d}__x{SF.slug(p['x'])}_y{SF.slug(p['y'])}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(OUT / "points.json"))
    ap.add_argument("--out", default=str(OUT / "assemblages"))
    ap.add_argument("--protocols", default="drift",
                    help="composition protocols from sparse_fit.PROTOCOLS")
    ap.add_argument("--out-size", type=int, default=1080)
    ap.add_argument("--work-res", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=0.055)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--emphasis", default="even")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, default=0, help="first N points, for a probe")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    man = json.loads(Path(a.points).read_text(encoding="utf-8"))
    pts = man["points"]
    for i, p in enumerate(pts):
        p["i"] = i
    if a.only:
        pts = pts[:a.only]
    names = [n.strip() for n in a.protocols.split(",") if n.strip()]
    for n in names:
        if n not in SF.PROTOCOLS:
            raise SystemExit(f"unknown protocol {n}; choose from {list(SF.PROTOCOLS)}")

    fld = G.load_field(man["protocol"])
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)

    made, skipped, failed, t0 = 0, 0, [], time.time()
    for p in pts:
        cdir = root / slug(p)
        cdir.mkdir(parents=True, exist_ok=True)
        lf = SF.emphasise(fld, G.lift(fld, p["x"], p["y"], k=a.k), a.emphasis)
        forms = None
        for proto in names:
            png = cdir / f"{proto}.png"
            if png.exists() and not a.force:
                skipped += 1
                continue
            cfg = SF.PROTOCOLS[proto]
            try:
                if forms is None:
                    rng = np.random.default_rng(a.seed + p["i"])
                    forms = R.forms_at(fld, lf, a.work_res, a.out_size,
                                       0.012, 0.45, 12, rng, edge="hard")
                if not forms:
                    failed.append((slug(p), proto, "no forms"))
                    break
                canvas, placed, fit, matte = SF.compose_fitted(
                    fld, lf, forms, a.out_size, a.seed + p["i"],
                    margin_frac=a.margin,
                    n_forms=cfg["n_forms"], scale=cfg["scale"],
                    jitter=cfg["jitter"], coherence=cfg["coherence"],
                    spin=cfg["spin"], position=cfg["position"])
            except Exception as ex:
                failed.append((slug(p), proto, f"{type(ex).__name__}: {ex}"))
                continue

            Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).save(png)
            Image.fromarray((np.clip(matte, 0, 1) * 255).astype(np.uint8),
                            "RGBA").save(cdir / f"{proto}_matte.png")
            cov = float((matte[:, :, 3] > 0.5).mean())
            (cdir / f"{proto}.json").write_text(json.dumps({
                "method": proto, "note": cfg["note"],
                "protocol": man["protocol"], "category": "all",
                "point_mode": p["mode"], "point_index": p["i"],
                "reserved": p["reserved"],
                "x": p["x"], "y": p["y"],
                "plan_x_in": p["plan_x_in"], "plan_y_in": p["plan_y_in"],
                "size_in": p["size_in"], "size_ft": p["size_ft"],
                "steps_out": p["steps_out"], "diversity": p["diversity"],
                "clash": p["clash"], "periods": p["periods"],
                "matte": f"{proto}_matte.png", "matte_coverage": round(cov, 4),
                "contained": True, "fit_scale": fit, "margin_frac": a.margin,
                "edge": "hard", "k": a.k, "emphasis": a.emphasis,
                "n_forms": cfg["n_forms"], "scale": cfg["scale"],
                "jitter": cfg["jitter"], "coherence": cfg["coherence"],
                "spin": cfg["spin"], "position_from": cfg["position"],
                "influence": SF.INFLUENCE,
                "forms_available": len(forms), "forms_placed": placed,
                "out_size": a.out_size, "work_res": a.work_res,
                "seed": a.seed + p["i"],
                "sources": sorted({f["id"] for f in forms}),
            }, indent=2), encoding="utf-8")
            made += 1
            print(f"  [{made}] {slug(p)}/{proto}  {placed} forms  "
                  f"fit={fit}  matte {100*cov:.1f}%  "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)

    print(f"\n{made} made, {skipped} already present, {len(failed)} failed "
          f"in {(time.time()-t0)/60:.1f}m")
    for f in failed:
        print("  FAIL", *f)
    (root / "index.json").write_text(json.dumps({
        "points": str(Path(a.points)), "protocols": names,
        "made": made, "skipped": skipped,
        "failed": [list(f) for f in failed],
        "dirs": [slug(p) for p in pts],
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
