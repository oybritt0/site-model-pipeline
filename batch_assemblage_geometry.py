r"""
batch_assemblage_geometry.py — a run of sparse_fit compositions as tagged geometry.

For each composition it writes ONE SVG and ONE OBJ holding both readings of every
fragment, in the same place, so they can be toggled against each other:

    true__frag_NN__<tag>       the full-resolution traced silhouette
    curated__frag_NN__<tag>    simplified, with radial edges and sharp detail held back

Each fragment carries a semantic tag from the project's own apparatus: its period,
plus its strongest term in each of the four facets of verbal_vocab.json, read off
that work's row of verbal_profile.npy. The vocabulary file's comment says the term
list "IS the projection apparatus", so the tags stay inside the method rather than
importing outside description. Column order is the facet lists concatenated, the
same convention gap_prompt.py pins down.

    python batch_assemblage_geometry.py --protocol dinov2_base -n 6
    python batch_assemblage_geometry.py -n 4 --compositions drift,vestige,swarm
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon

import gap_space_core as G
import gap_prompt as GP
import rift_atlas as R
import sparse_fit as SF
import assemblage_geometry as AG

# the curated reading, from the sweep: arcs carry the long moves, tight corners
# and small incidents stay sharp so source detail still reads
CURATED = dict(smooth=0.015, tol=0.035, fillet_rel=0.45, sharp_edge=0.0,
               sharp_angle=75.0, min_turn=12.0)


def slug(s: str, n=28) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:n]


def esc(v) -> str:
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def fragment_terms(work_id: str) -> dict:
    """Strongest authored term per facet for one work."""
    c = GP._load()
    ids = c["ids"] or []
    row = {i: r for r, i in enumerate(ids)}.get(work_id)
    if row is None or row >= len(c["prof"]):
        return {}
    v = c["prof"][row]
    return {name: terms[int(np.argmax(v[a:b]))]
            for name, (a, b, terms) in c["facets"].items()}


def write_svg(path, frags, size, info, image_b64=None):
    """Both readings, as switchable layer groups, over the composition they trace."""
    def d(ring):
        return ("M " + " L ".join(f"{p[0]*size:.2f},{p[1]*size:.2f}" for p in ring) + " Z")

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
           f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
           f'<g id="ground"><rect width="{size}" height="{size}" fill="#f4f4f2"/></g>']
    if image_b64:
        # embedded rather than linked, so the sheet stays readable on its own;
        # xlink:href as well as href for older renderers
        uri = f"data:image/png;base64,{image_b64}"
        out.append(f'<g id="source" inkscape:label="source" inkscape:groupmode="layer">'
                   f'<image x="0" y="0" width="{size}" height="{size}" '
                   f'href="{uri}" xlink:href="{uri}" /></g>')
    for key, opacity, width in (("true", 0.0, 0.7), ("curated", 0.38, 2.0)):
        out.append(f'<g id="{key}" inkscape:label="{key}" inkscape:groupmode="layer">')
        for i, f in enumerate(frags):
            c = AG.PALETTE[i % len(AG.PALETTE)]
            fill = f"rgb{c}" if opacity else "none"
            t = f["terms"]
            # the layer name can only carry so much, so every facet also goes on
            # the group as data-* and into title/desc, which survive Inkscape and
            # Illustrator round-trips
            attrs = " ".join(
                f'data-{k}="{esc(v)}"' for k, v in (
                    ("reading", key), ("work-id", f["work_id"]),
                    ("period", f["period"]), ("coseg-part", f["coseg_part"]),
                    ("material", t.get("material", "")),
                    ("organization", t.get("organization", "")),
                    ("gestalt", t.get("gestalt", "")),
                    ("surface", t.get("surface", ""))))
            desc = (f'{key} reading | work {f["work_id"]} | period {f["period"]} | '
                    f'coseg part {f["coseg_part"]} | material: {t.get("material","")} | '
                    f'organization: {t.get("organization","")} | '
                    f'gestalt: {t.get("gestalt","")} | surface: {t.get("surface","")} | '
                    f'{len(f[key])} points')
            out.append(f'<g id="{key}__{f["name"]}" inkscape:label="{key}__{f["name"]}" '
                       f'inkscape:groupmode="layer" {attrs}>'
                       f'<title>{esc(key)}__{esc(f["name"])}</title>'
                       f'<desc>{esc(desc)}</desc>'
                       f'<path d="{d(f[key])}" fill="{fill}" fill-opacity="{opacity}" '
                       f'stroke="rgb{c}" stroke-width="{width}"/></g>')
        out.append("</g>")
    out.append(f'<g id="annotation"><text x="12" y="{size-12}" font-size="13" '
               f'font-family="monospace">{info}</text></g></svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def write_obj(path, frags, size, depth, stagger):
    """Both readings as named objects, sharing each fragment's z so they overlay."""
    blocks, closed = [], 0
    for i, f in enumerate(frags):
        z = i * stagger * size
        for key in ("true", "curated"):
            poly = AG._clean(Polygon(np.column_stack(
                [f[key][:, 0] * size, (1.0 - f[key][:, 1]) * size])))
            if poly is None:
                continue
            m = trimesh.creation.extrude_polygon(poly, height=max(depth * size, 1e-6))
            m.apply_translation([0, 0, z])
            closed += bool(m.is_watertight)
            t = f["terms"]
            # OBJ has no metadata mechanism, so the facets ride in comments
            # directly above the object they describe
            head = "\n".join([
                f"# {key} reading of frag_{i:02d}",
                f"# work_id={f['work_id']}  period={f['period']}  "
                f"coseg_part={f['coseg_part']}",
                f"# material={t.get('material','')}",
                f"# organization={t.get('organization','')}",
                f"# gestalt={t.get('gestalt','')}",
                f"# surface={t.get('surface','')}",
                f"# points={len(f[key])}  arcs={f['arcs']}  "
                f"deviation_px={f['deviation_px']}",
            ])
            blocks.append((f"{key}__{f['name']}", m, head))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {path.name} - {len(blocks)} prisms, true and curated\n")
        off = 1
        for name, m, head in blocks:
            fh.write(f"\n{head}\no {name}\ng {name}\n")
            for v in m.vertices:
                fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for a, b, c in m.faces + off:
                fh.write(f"f {a} {b} {c}\n")
            off += len(m.vertices)
    return len(blocks), closed


def build_one(fld, x, y, comp, args, out: Path, lf, forms):
    cfg = SF.PROTOCOLS[comp]
    traced, fit = AG.compose_traced(fld, lf, forms, args.out_size, args.seed, cfg,
                                    tol_px=0.0)
    if not traced:
        return None
    # the raster sparse_fit would have made from the same placement, so the
    # curves can be read against the composition they were taken from
    canvas, _, _, _ = SF.compose_fitted(
        fld, lf, forms, args.out_size, args.seed, n_forms=cfg["n_forms"],
        scale=cfg["scale"], jitter=cfg["jitter"], coherence=cfg["coherence"],
        spin=cfg["spin"], position=cfg["position"])
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))

    frags = []
    for i, t in enumerate(traced):
        true_ring = AG.simplify(t["placed"], "outline", 0.0)
        if true_ring is None:
            continue
        r = AG.simplify(AG.smooth_ring(t["placed"], CURATED["smooth"]),
                        "outline", CURATED["tol"])
        if r is None:
            continue
        span = math.sqrt(Polygon(r).area)
        segs = AG.fillet_ring(r, CURATED["fillet_rel"] * span, CURATED["min_turn"],
                              sharp_edge=CURATED["sharp_edge"],
                              sharp_angle=CURATED["sharp_angle"], span=span)
        cur = AG.segs_to_ring(segs)
        terms = fragment_terms(t["id"])
        period = str(t["period"]).split("_", 1)[-1]
        name = (f"frag_{i:02d}__{slug(period, 16)}"
                f"__{slug(terms.get('material', 'na'), 32)}"
                f"__{slug(terms.get('gestalt', 'na'), 32)}")
        tp, sp = Polygon(true_ring).buffer(0), Polygon(cur).buffer(0)
        frags.append({
            "name": name, "true": true_ring, "curated": cur,
            "work_id": t["id"], "period": t["period"], "coseg_part": int(t["part"]),
            "terms": terms, "arcs": sum(1 for s in segs if s[0] == "A"),
            "straight": sum(1 for s in segs if s[0] == "L"),
            "true_points": len(true_ring), "curated_points": len(cur),
            "deviation_px": round(tp.exterior.hausdorff_distance(sp.exterior)
                                  * args.out_size, 1),
            "area_err_pct": round(abs(sp.area - tp.area) / tp.area * 100, 2),
        })
    if not frags:
        return None

    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "composition.png")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    info = f"{fld.display_name}  ({x:.3f}, {y:.3f})  {comp}  {len(frags)} fragments"
    write_svg(out / "composition.svg", frags, args.size, info, image_b64=b64)
    nobj, closed = write_obj(out / "solids.obj", frags, args.size, args.depth, args.stagger)

    with open(out / "fragments.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "work_id", "period", "coseg_part", "material",
                    "organization", "gestalt", "surface", "true_points",
                    "curated_points", "arcs", "straight", "deviation_px",
                    "area_err_pct"])
        for f in frags:
            t = f["terms"]
            w.writerow([f["name"], f["work_id"], f["period"], f["coseg_part"],
                        t.get("material", ""), t.get("organization", ""),
                        t.get("gestalt", ""), t.get("surface", ""),
                        f["true_points"], f["curated_points"], f["arcs"],
                        f["straight"], f["deviation_px"], f["area_err_pct"]])
    (out / "fragments.json").write_text(json.dumps({
        "protocol": fld.protocol_id if hasattr(fld, "protocol_id") else args.protocol,
        "x": x, "y": y, "composition": comp, "fit_scale": fit,
        "curated": CURATED, "size_mm": args.size, "depth": args.depth,
        "fragments": [{k: v for k, v in f.items() if k not in ("true", "curated")}
                      for f in frags],
    }, indent=2), encoding="utf-8")
    return len(frags), nobj, closed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="dinov2_base")
    ap.add_argument("-n", "--candidates", type=int, default=6)
    ap.add_argument("--compositions", default="drift")
    ap.add_argument("--points", default="empty")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-size", type=int, default=1080, dest="out_size")
    ap.add_argument("--work-res", type=int, default=512, dest="work_res")
    ap.add_argument("--size", type=float, default=1000.0)
    ap.add_argument("--depth", type=float, default=0.04)
    ap.add_argument("--stagger", type=float, default=0.006)
    ap.add_argument("--tol", type=float, default=None,
                    help="curated simplification; lower keeps more corners")
    ap.add_argument("--smooth", type=float, default=None)
    ap.add_argument("--fillet-rel", type=float, default=None, dest="fillet_rel")
    ap.add_argument("--sharp-angle", type=float, default=None, dest="sharp_angle")
    ap.add_argument("--sharp-edge", type=float, default=None, dest="sharp_edge")
    ap.add_argument("--coords", type=Path, default=None,
                    help="a points.json from tile_points.py: build these exact "
                         "coordinates instead of searching for candidates, so "
                         "the geometry route can be read against the "
                         "image-to-3D route on the same ground")
    ap.add_argument("--max-per-work", type=int, default=None, dest="max_per_work",
                    help="forms cut per work; sparse_fit fixes this at 12 "
                         "independently of k, so matching it matters when the "
                         "point is to replay the same placement")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    for k in ("tol", "smooth", "fillet_rel", "sharp_angle", "sharp_edge"):
        v = getattr(a, k)
        if v is not None:
            CURATED[k] = v
    print(f"curated: {CURATED}")

    comps = [c.strip() for c in a.compositions.split(",") if c.strip()]
    bad = [c for c in comps if c not in SF.PROTOCOLS]
    if bad:
        raise SystemExit(f"unknown composition(s) {bad}; choose from {list(SF.PROTOCOLS)}")

    if a.coords:
        man = json.loads(a.coords.read_text(encoding="utf-8"))
        a.protocol = man["protocol"]
        cands = [dict(p, label=f"{p['mode']}_{i:02d}")
                 for i, p in enumerate(man["points"])]
        if a.candidates:
            cands = cands[:a.candidates]
    else:
        cands = None
    fld = G.load_field(a.protocol)
    if cands is None:
        cands = G.gap_candidates(fld, n=a.candidates, samples=8000, seed=a.seed,
                                 sort=a.points)
    root = a.out or (G.OUT / "assemblage_geometry_batch" /
                     f"{a.protocol}_{time.strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)

    base_seed, mpw = a.seed, a.max_per_work or a.k
    made, t0 = 0, time.time()
    for ci, c in enumerate(cands):
        # --coords replays a named point, so the seed has to advance with the
        # point exactly as sparse_fit's does or the two routes diverge
        a.seed = base_seed + ci if a.coords else base_seed
        # cutting the forms is the expensive step, so do it once per coordinate
        # and reuse it across every composition protocol, as sparse_fit does
        lf = G.lift(fld, c["x"], c["y"], k=a.k)
        forms = R.forms_at(fld, lf, a.work_res, a.out_size, 0.012, 0.45, mpw,
                           np.random.default_rng(a.seed), edge="hard")
        if not forms:
            print(f"  skip candidate {ci}: no forms cut", flush=True)
            continue
        for comp in comps:
            tag = c.get("label", f"{a.points}_{ci:02d}")
            d = root / f"{tag}__x{c['x']:.2f}_y{c['y']:.2f}__{comp}"
            try:
                got = build_one(fld, c["x"], c["y"], comp, a, d, lf, forms)
            except Exception as ex:
                print(f"  FAIL {d.name}: {type(ex).__name__}: {ex}", flush=True)
                continue
            if not got:
                print(f"  skip {d.name}: nothing composed", flush=True)
                continue
            nf, nobj, closed = got
            made += 1
            print(f"  [{made}] {d.name}  {nf} fragments, {nobj} prisms, "
                  f"{closed} watertight  ({(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"\n{made} compositions -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
