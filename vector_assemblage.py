r"""
vector_assemblage.py — the gap assemblage as a drawing, not a photo-collage.

The raster assemblage cuts pixels out of photographs. That reads as photomontage
-- a representational convention roughly a century old, and pictorial rather
than architectural. The parts it cuts already exist as VECTOR outlines:
coseg_to_geometry.py wrote closed part silhouettes for all 4449 works into
content_geometry_coseg.json, in its words "a real outline you can boolean-union
or quote by hand across media. Nothing is synthesised."

So this composes the same coordinate out of the same parts, chosen by the same
lift, placed by the same logic -- position from where the part sat in its source,
scale from the source work's lift weight, rotation from the part's own principal
axis -- but emits polygons. The output is an editable SVG with organised layers,
which is native to architectural drawing, goes to Rhino through the existing
export path, and carries no photographic surface at all.

Layers, in draw order:
    ground              a light figure-ground wash of everything, for reading
    part_<cluster>      one layer per co-segmentation cluster, so the discovered
                        vocabulary stays separable
    cuts                the heaviest outlines, poche-weight
    annotation          coordinate, period mix, scale bar

    python vector_assemblage.py --protocol dinov2_base --at 9.71 3.35
    python vector_assemblage.py --sort mixed -n 3 --mode section
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import gap_space_core as G

COSEG_GEOM = G.E / "content_geometry_coseg.json"
_GEOM: dict = {}


def geom():
    if not _GEOM:
        _GEOM.update(json.load(open(COSEG_GEOM, encoding="utf-8"))["items"])
    return _GEOM


def _poly_stats(poly: np.ndarray):
    """Centroid, area, principal axis and elongation of a closed polygon."""
    x, y = poly[:, 0], poly[:, 1]
    a = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    c = poly.mean(axis=0)
    d = poly - c
    cov = np.cov(d.T) if len(poly) > 2 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    ang = float(np.degrees(np.arctan2(evec[1, -1], evec[0, -1])))
    elong = float(np.sqrt(max(ev[-1], 1e-12) / max(ev[0], 1e-12)))
    return c, a, ang, elong


def collect(fld: G.Field, lf: G.Lift, min_area=0.004, max_area=0.45,
            max_per_work=8, part_bias=True, rng=None) -> list:
    """Every usable part polygon from the neighbourhood, with its placement data."""
    rng = rng or np.random.default_rng(0)
    gm = geom()
    meta = G.coseg_parts()
    pw = None
    if part_bias:
        acc = np.zeros(meta["k"])
        pidx = {p: i for i, p in enumerate(meta["periods"])}
        for i, w in zip(lf.idx, lf.w):
            r = pidx.get(fld.period[i])
            if r is not None:
                acc += w * meta["hist"][r]
        if acc.sum() > 0:
            pw = 0.35 + 0.65 * (acc / acc.max())

    out = []
    for i, w in zip(lf.idx, lf.w):
        rec = gm.get(fld.ids[i])
        if not rec:
            continue
        got = []
        for part in rec.get("coseg", []):
            poly = np.asarray(part["poly"], dtype=float)
            if len(poly) < 4:
                continue
            c, area, ang, elong = _poly_stats(poly)
            if area < min_area or area > max_area:
                continue
            got.append({
                "poly": poly, "c": c, "area": area, "ang": ang, "elong": elong,
                "part": int(part["c"]), "col": part.get("col", [90, 90, 95]),
                "w": float(w) * (1.0 if pw is None else float(pw[int(part["c"])])),
                "id": fld.ids[i], "period": fld.period[i],
            })
        if len(got) > max_per_work:
            pr = np.array([g["area"] for g in got])
            pr = pr / pr.sum()
            got = [got[j] for j in rng.choice(len(got), max_per_work,
                                              replace=False, p=pr)]
        out.extend(got)
    return out


def place(forms, n_forms=22, scale=1.6, influence=0.7, jitter=0.05,
          coherence=0.0, spin=0.0, mode="fused", seed=0) -> list:
    """Same placement argument as the raster painter, applied to polygons."""
    rng = np.random.default_rng(seed)
    if not forms:
        return []
    pr = np.array([f["w"] for f in forms], dtype=float)
    pr = pr / pr.sum()
    n = int(min(n_forms, len(forms)))
    chosen = [forms[i] for i in rng.choice(len(forms), n, replace=False, p=pr)]
    chosen.sort(key=lambda f: -f["area"])
    wmean = float(np.mean([f["w"] for f in chosen])) or 1.0

    placed = []
    for f in chosen:
        gain = (f["w"] / wmean) ** influence
        s = float(np.clip(scale * math.sqrt(gain), 0.15, 4.0))
        ang = f["ang"] * (1.0 - coherence) + (rng.uniform(-spin, spin) if spin else 0.0)
        cx = f["c"][0] + rng.normal(0, jitter)
        cy = f["c"][1] + rng.normal(0, jitter)
        if mode == "plan":
            # a plan reads horizontally: collapse the source's vertical position
            # into depth so parts spread across the sheet rather than stacking
            cy = 0.5 + (cy - 0.5) * 0.45 + rng.normal(0, jitter)
        elif mode == "section":
            # a section keeps the source's height exactly; that stratification
            # is the one piece of spatial information a photograph really carries
            cy = f["c"][1]
        p = (f["poly"] - f["c"])
        th = math.radians(ang)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        p = (p * s) @ R.T + np.array([cx, cy])
        placed.append({**f, "placed": p, "scale": s, "angle": ang})
    return placed


def to_svg(placed, info, size=1000, margin=40) -> str:
    """Editable SVG, one layer per co-segmentation cluster."""
    S, M = size, margin
    def pts(p):
        return " ".join(f"{M + v[0]*(S-2*M):.2f},{M + v[1]*(S-2*M):.2f}" for v in p)

    by_part: dict = {}
    for f in placed:
        by_part.setdefault(f["part"], []).append(f)

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
         f'width="{S}" height="{S}" viewBox="0 0 {S} {S}">',
         '<rect width="100%" height="100%" fill="#ffffff"/>']

    L.append('<g id="ground" inkscape:groupmode="layer" inkscape:label="ground">')
    for f in placed:
        L.append(f'<polygon points="{pts(f["placed"])}" fill="#000" '
                 f'fill-opacity="0.045" stroke="none"/>')
    L.append('</g>')

    for part in sorted(by_part):
        r, g, b = by_part[part][0]["col"]
        L.append(f'<g id="part_{part:02d}" inkscape:groupmode="layer" '
                 f'inkscape:label="part {part:02d}" '
                 f'stroke="rgb({r},{g},{b})" fill="none" stroke-width="1.1">')
        for f in by_part[part]:
            L.append(f'<polygon points="{pts(f["placed"])}" '
                     f'data-source="{f["id"]}" data-period="{f["period"]}"/>')
        L.append('</g>')

    # the largest forms again, heavier: these read as the cut
    L.append('<g id="cuts" inkscape:groupmode="layer" inkscape:label="cuts" '
             'fill="none" stroke="#111" stroke-width="2.4">')
    for f in sorted(placed, key=lambda f: -f["area"])[:max(1, len(placed)//4)]:
        L.append(f'<polygon points="{pts(f["placed"])}"/>')
    L.append('</g>')

    per = " · ".join(f"{p}" for p in info["periods"])
    L.append('<g id="annotation" inkscape:groupmode="layer" inkscape:label="annotation" '
             'font-family="Helvetica,Arial" fill="#222">')
    L.append(f'<text x="{M}" y="{S-24}" font-size="13">{info["protocol"]} '
             f'({info["x"]:.2f}, {info["y"]:.2f})  ·  {info["steps_out"]:.0f} steps out '
             f'·  {info["mode"]}</text>')
    L.append(f'<text x="{M}" y="{S-8}" font-size="11" fill="#666">{per}</text>')
    L.append('</g></svg>')
    return "\n".join(L)


def build(fld: G.Field, px, py, *, k=10, power=2.0, mode="fused", seed=0,
          size=1000, **kw) -> tuple:
    lf = G.lift(fld, px, py, k=k, power=power)
    forms = collect(fld, lf, rng=np.random.default_rng(seed))
    if not forms:
        raise RuntimeError("no part polygons in this neighbourhood")
    placed = place(forms, mode=mode, seed=seed, **kw)
    acc: dict = {}
    for i, w in zip(lf.idx, lf.w):
        acc[fld.period[i]] = acc.get(fld.period[i], 0.0) + float(w)
    info = {
        "protocol": fld.protocol, "x": float(px), "y": float(py),
        "steps_out": lf.steps_out, "mode": mode,
        "forms_available": len(forms), "forms_placed": len(placed),
        "parts_used": sorted({f["part"] for f in placed}),
        "periods": [f"{p.split('_',1)[-1]} {w:.0%}" for p, w in
                    sorted(acc.items(), key=lambda kv: -kv[1])[:3]],
        "sources": sorted({f["id"] for f in placed}),
    }
    return to_svg(placed, info, size=size), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="dinov2_base")
    ap.add_argument("--at", nargs=2, type=float)
    ap.add_argument("--sort", default="mixed")
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("--mode", default="fused", choices=["fused", "plan", "section"])
    ap.add_argument("--n-forms", type=int, default=22)
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fld = G.load_field(a.protocol)
    pts = ([tuple(a.at)] if a.at else
           [(c["x"], c["y"]) for c in G.gap_candidates(
               fld, n=a.n, samples=8000, min_steps=8.0, sort=a.sort)])
    outdir = Path(a.out) if a.out else G.OUT / "vector"
    outdir.mkdir(parents=True, exist_ok=True)
    for x, y in pts:
        svg, info = build(fld, x, y, mode=a.mode, seed=a.seed,
                          n_forms=a.n_forms, scale=a.scale)
        name = f"{a.protocol}_{x:.2f}_{y:.2f}_{a.mode}".replace(".", "p").replace("-", "m")
        (outdir / f"{name}.svg").write_text(svg, encoding="utf-8")
        (outdir / f"{name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"{info['forms_placed']}/{info['forms_available']} forms, "
              f"parts {info['parts_used']} -> {name}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
