r"""
assemblage_geometry.py — a sparse_fit composition as simple closed geometry.

sparse_fit composes by cutting raster fragments and shrinking the whole layer to
fit. The pieces it cuts already exist as vector outlines in
content_geometry_coseg.json, so the same composition can be built as polygons
instead of pixels, and then simplified until it is workable: every fragment one
closed, non-self-intersecting curve, and one closed solid.

Placement is sparse_fit's, verbatim -- source centroid plus jitter, size from
influence weight, rotation from the form's own principal axis -- followed by the
same single rigid shrink of the whole arrangement, which in vector space is just
a scale about the composition's bounding box. Relative positions, overlaps and
stranded outliers survive exactly.

    --shape outline   simplified silhouette, the general form kept   (default)
    --shape hull      convex hull, the simplest closed reading
    --shape rect      minimum-area rectangle
    --shape ellipse   fitted ellipse

Outputs, all closed: a layered SVG, DXF closed polylines, and extruded solids as
OBJ meshes and STEP polysurfaces (bottom + top + one planar face per segment).

    python assemblage_geometry.py --protocol dinov2_base --at 4.79 10.00
    python assemblage_geometry.py --protocol dinov2_base --shape hull --depth 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon

import gap_space_core as G
import rift_atlas as R
import sparse_fit as SF
import vector_assemblage as VA

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207)]


# ---------------------------------------------------------------- simplify ---

def _clean(poly: Polygon) -> Polygon | None:
    """Force a valid, single-ring, non-self-intersecting polygon or give up."""
    if poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)   # keep the main lobe
    if poly.is_empty or poly.area <= 0:
        return None
    return Polygon(poly.exterior)                      # drop interior rings


def simplify(pts: np.ndarray, shape: str, tol: float) -> np.ndarray | None:
    """One closed, simple ring per fragment. `tol` is a fraction of its size."""
    poly = _clean(Polygon(pts))
    if poly is None:
        return None
    span = math.sqrt(poly.area)

    if shape == "hull":
        poly = poly.convex_hull
    elif shape == "rect":
        poly = poly.minimum_rotated_rectangle
    elif shape == "ellipse":
        c = np.asarray(poly.centroid.coords[0])
        d = np.asarray(poly.exterior.coords) - c
        cov = np.cov(d.T)
        ev, evec = np.linalg.eigh(cov)
        t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        # 2*sqrt(eigenvalue) matches the spread of the outline it replaces
        e = np.stack([2 * math.sqrt(max(ev[1], 1e-12)) * np.cos(t),
                      2 * math.sqrt(max(ev[0], 1e-12)) * np.sin(t)], 1)
        poly = Polygon(e @ np.array([evec[:, 1], evec[:, 0]]) + c)
    else:                                              # outline
        poly = poly.simplify(tol * span, preserve_topology=True)

    poly = _clean(poly)
    if poly is None or poly.area <= 0:
        return None
    ring = np.asarray(poly.exterior.coords)[:-1]       # open ring, no repeat
    return ring if len(ring) >= 3 else None


def smooth_ring(ring: np.ndarray, frac: float) -> np.ndarray:
    """Periodic moving average, so simplification picks real corners rather than
    pixel stair-steps. `frac` is a fraction of the ring's point count."""
    n = len(ring)
    w = max(1, int(round(frac * n)))
    if w < 2 or n < 8:
        return ring
    k = np.ones(2 * w + 1) / (2 * w + 1)
    pad = np.vstack([ring[-w:], ring, ring[:w]])
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in (0, 1)], 1)


# ------------------------------------------------------------------ radial ---

def fillet_ring(ring: np.ndarray, radius: float, min_turn_deg=18.0,
                sharp_edge=0.0, sharp_angle=0.0, span=None):
    """Replace corners with tangent arcs, giving real radial geometry.

    Filleting a solid's edges is what OpenCASCADE could not do on these shapes;
    doing it in 2D before the extrusion is exact, and the arcs become cylindrical
    faces on the prism. Returns segments: ("L", p0, p1) or ("A", p0, p1, centre).

    Rounding every corner flattens a shape into one continuous gesture. Two rules
    hold detail back out of that: a corner tight enough to read as a point
    (`sharp_angle`), and a corner between short edges, which is a small incident
    on a longer run (`sharp_edge`, a fraction of the shape's span). Both stay
    sharp, so the arcs carry the long moves and the detail interrupts them.
    """
    n = len(ring)
    if n < 3 or radius <= 0:
        return [("L", ring[i], ring[(i + 1) % n]) for i in range(n)]
    if span is None:
        span = math.sqrt(max(Polygon(ring).area, 1e-12))

    min_turn = math.radians(min_turn_deg)
    sharp_a = math.radians(sharp_angle) if sharp_angle > 0 else 0.0
    entry, exit_, arcs = [None] * n, [None] * n, [None] * n
    for i in range(n):
        v, a, b = ring[i], ring[i - 1], ring[(i + 1) % n]
        d1, d2 = a - v, b - v
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if l1 < 1e-9 or l2 < 1e-9:
            entry[i] = exit_[i] = v
            continue
        u1, u2 = d1 / l1, d2 / l2
        theta = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        # near-straight stays a corner-free join; a near-fold has no room for an arc
        if theta > math.pi - min_turn or theta < 1e-3:
            entry[i] = exit_[i] = v
            continue
        if sharp_a and theta < sharp_a:                      # a real point
            entry[i] = exit_[i] = v
            continue
        if sharp_edge and min(l1, l2) < sharp_edge * span:   # a small incident
            entry[i] = exit_[i] = v
            continue
        half = theta / 2.0
        t = min(radius / math.tan(half), 0.5 * l1, 0.5 * l2)
        r_eff = t * math.tan(half)
        if r_eff < 1e-9:
            entry[i] = exit_[i] = v
            continue
        entry[i], exit_[i] = v + u1 * t, v + u2 * t
        bis = u1 + u2
        nb = np.linalg.norm(bis)
        if nb < 1e-9:
            entry[i] = exit_[i] = v
            continue
        arcs[i] = v + (bis / nb) * (r_eff / math.sin(half))

    segs = []
    for i in range(n):
        if arcs[i] is not None:
            segs.append(("A", entry[i], exit_[i], arcs[i]))
        p_from = exit_[i]
        p_to = entry[(i + 1) % n]
        if np.linalg.norm(p_to - p_from) > 1e-9:
            segs.append(("L", p_from, p_to))
    return segs


def _arc_points(p0, p1, c, steps=None):
    """Sample the minor arc from p0 to p1 about c, endpoints included."""
    a0 = math.atan2(*(p0 - c)[::-1])
    a1 = math.atan2(*(p1 - c)[::-1])
    d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi     # shortest signed sweep
    r = float(np.linalg.norm(p0 - c))
    if steps is None:
        steps = max(2, int(abs(d) / 0.12) + 1)
    t = a0 + d * np.linspace(0, 1, steps + 1)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1), d, r


def segs_to_ring(segs) -> np.ndarray:
    """Polyline approximation, for meshing and preview."""
    pts = []
    for s in segs:
        if s[0] == "A":
            pts.append(_arc_points(s[1], s[2], s[3])[0][:-1])
        else:
            pts.append(s[1][None, :])
    return np.concatenate(pts)


# ------------------------------------------------------------ raster trace ---

def compose_traced(fld, lf, forms, size, seed, cfg, margin_frac=0.055,
                   influence=SF.INFLUENCE, pad_mult=2.2, tol_px=1.2):
    """sparse_fit.compose_fitted, but recording outlines instead of pixels.

    The coseg polygons top out at ~24 points, which is well below the fragment
    shapes the raster composition actually draws. These are the same masks the
    painter composites, traced at full resolution and then carried through the
    identical crop-and-shrink, so the curves land on the drawn silhouettes.
    """
    import cv2
    rng = np.random.default_rng(seed)
    W = int(size * pad_mult)
    off = (W - size) // 2

    pr = np.array([f["w"] for f in forms], dtype=float)
    pr = pr / pr.sum()
    n = int(min(cfg["n_forms"], len(forms)))
    chosen = [forms[i] for i in rng.choice(len(forms), size=n, replace=False, p=pr)]
    chosen.sort(key=lambda f: -f["area"])
    wmean = float(np.mean([f["w"] for f in chosen])) or 1.0

    emb, bear = {}, {}
    if cfg["position"] in ("embedding", "bearing"):
        d = fld.xy[lf.idx] - np.array([lf.x, lf.y])
        mx = float(np.abs(d).max()) or 1.0
        for j, i in enumerate(lf.idx):
            emb[fld.ids[i]] = (0.5 + 0.42 * d[j][0] / mx, 0.5 - 0.42 * d[j][1] / mx)
            bear[fld.ids[i]] = math.degrees(math.atan2(-d[j][1], d[j][0]))

    traced = []
    for f in chosen:
        h0, w0 = f["rgba"].shape[:2]
        if h0 < 2 or w0 < 2:
            continue
        gain = (f["w"] / wmean) ** influence
        target = f["area"] * (size * size) * (cfg["scale"] ** 2) * gain
        cur = max(1.0, float((f["rgba"][:, :, 3] > 0.3).sum()))
        sc = float(np.clip(np.sqrt(target / cur), 0.15, 3.0))
        if max(h0, w0) * sc > 1.5 * size:
            sc = 1.5 * size / max(h0, w0)
        nh, nw = max(2, int(h0 * sc)), max(2, int(w0 * sc))
        if nh > 4 * size or nw > 4 * size:
            continue
        tile = Image.fromarray((np.clip(f["rgba"], 0, 1) * 255).astype(np.uint8),
                               "RGBA").resize((nw, nh), Image.LANCZOS)
        ang = f["angle"] * (1.0 - cfg["coherence"])
        if cfg["position"] == "bearing" and f["id"] in bear:
            ang = bear[f["id"]]
        if cfg["spin"]:
            ang += float(rng.uniform(-cfg["spin"], cfg["spin"]))
        if abs(ang) > 0.5:
            tile = tile.rotate(ang, expand=True, resample=Image.BICUBIC)
        ta = np.asarray(tile, dtype=np.float32) / 255.0
        th, tw = ta.shape[:2]
        if cfg["position"] in ("embedding", "bearing") and f["id"] in emb:
            bx, by = emb[f["id"]]
        else:
            bx, by = f["cx"], f["cy"]
        cx = bx + float(rng.normal(0, cfg["jitter"]))
        cy = by + float(rng.normal(0, cfg["jitter"]))
        x0 = int(round(cx * size - tw / 2)) + off
        y0 = int(round(cy * size - th / 2)) + off

        alpha = (ta[:, :, 3] > 0.3).astype(np.uint8)
        if alpha.sum() < 16:
            continue
        cs, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if tol_px > 0:
            c = cv2.approxPolyDP(c, tol_px, True)
        pts = c.reshape(-1, 2).astype(float) + np.array([x0, y0])
        if len(pts) >= 3:
            traced.append({**{k: v for k, v in f.items() if k != "rgba"},
                           "layer": pts, "src_points": int(alpha.sum() and len(cs[0]))})

    if not traced:
        return [], 1.0
    allp = np.concatenate([t["layer"] for t in traced])
    lo, hi = allp.min(0), allp.max(0)
    m = margin_frac * size
    safe = size - 2 * m
    k = float(min(safe / max(hi[0] - lo[0], 1e-9), safe / max(hi[1] - lo[1], 1e-9)))
    nw2, nh2 = (hi - lo) * k
    px, py = m + (safe - nw2) / 2, m + (safe - nh2) / 2
    for t in traced:
        t["placed"] = ((t["layer"] - lo) * k + np.array([px, py])) / size
    return traced, round(k, 4)


# ------------------------------------------------------------------- place ---

def place_fitted(forms, cfg, seed, margin_frac=0.055, influence=SF.INFLUENCE):
    """sparse_fit's placement, then its one rigid shrink -- in vector space."""
    rng = np.random.default_rng(seed)
    if not forms:
        return []
    pr = np.array([f["w"] for f in forms], dtype=float)
    pr = pr / pr.sum()
    n = int(min(cfg["n_forms"], len(forms)))
    chosen = [forms[i] for i in rng.choice(len(forms), n, replace=False, p=pr)]
    chosen.sort(key=lambda f: -f["area"])              # big behind, small in front
    wmean = float(np.mean([f["w"] for f in chosen])) or 1.0

    placed = []
    for f in chosen:
        gain = (f["w"] / wmean) ** influence
        s = float(np.clip(cfg["scale"] * math.sqrt(gain), 0.15, 4.0))
        ang = f["ang"] * (1.0 - cfg["coherence"])
        if cfg["spin"]:
            ang += float(rng.uniform(-cfg["spin"], cfg["spin"]))
        cx = f["c"][0] + float(rng.normal(0, cfg["jitter"]))
        cy = f["c"][1] + float(rng.normal(0, cfg["jitter"]))
        th = math.radians(ang)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        p = ((f["poly"] - f["c"]) * s) @ rot.T + np.array([cx, cy])
        placed.append({**f, "placed": p, "scale": s, "angle": ang})

    # the rigid shrink: one scale about the whole arrangement's bounding box
    allp = np.concatenate([p["placed"] for p in placed])
    lo, hi = allp.min(0), allp.max(0)
    safe = 1.0 - 2 * margin_frac
    k = float(min(safe / max(hi[0] - lo[0], 1e-9), safe / max(hi[1] - lo[1], 1e-9)))
    ctr = (lo + hi) / 2
    for p in placed:
        p["placed"] = (p["placed"] - ctr) * k + 0.5
    return placed, round(k, 4)


# ------------------------------------------------------------------ writers ---

def write_svg(path, rings, size, info):
    """One group per fragment, in draw order, plus a ground wash."""
    def d(r, segs):
        if not segs:
            return "M " + " L ".join(f"{v[0]*size:.2f},{v[1]*size:.2f}" for v in r) + " Z"
        out = [f"M {segs[0][1][0]*size:.2f},{segs[0][1][1]*size:.2f}"]
        for s in segs:
            if s[0] == "A":
                _, sweep, rad = _arc_points(s[1], s[2], s[3], steps=1)
                # always the minor arc, so large-arc-flag stays 0; SVG shares the
                # y-down convention these points are in, so the sign carries over
                out.append(f"A {rad*size:.3f},{rad*size:.3f} 0 0,{1 if sweep > 0 else 0} "
                           f"{s[2][0]*size:.2f},{s[2][1]*size:.2f}")
            else:
                out.append(f"L {s[2][0]*size:.2f},{s[2][1]*size:.2f}")
        return " ".join(out) + " Z"

    # the inkscape: attributes are what make these read as named layers rather
    # than anonymous groups, so the namespace has to be declared for valid XML
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
           f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
           f'<g id="ground"><rect width="{size}" height="{size}" fill="#f4f4f2"/></g>']
    for i, (name, r, segs) in enumerate(rings):
        c = PALETTE[i % len(PALETTE)]
        out.append(f'<g id="{name}" inkscape:label="{name}" inkscape:groupmode="layer">'
                   f'<path d="{d(r, segs)}" fill="rgb{c}" fill-opacity="0.35" '
                   f'stroke="rgb{c}" stroke-width="1.6"/></g>')
    out.append(f'<g id="annotation"><text x="12" y="{size-12}" font-size="13" '
               f'font-family="monospace">{info}</text></g></svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def write_dxf(path, rings, size):
    """Closed LWPOLYLINEs, one layer per fragment. Y flipped so it reads upright."""
    import ezdxf
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    for i, (name, r, segs) in enumerate(rings):
        c = PALETTE[i % len(PALETTE)]
        doc.layers.add(name, true_color=ezdxf.rgb2int(c))
        if segs:
            # bulge belongs to the segment leaving each vertex; flipping Y
            # reverses the sweep, so the sign flips with it
            pts = []
            for s in segs:
                b = 0.0
                if s[0] == "A":
                    _, sweep, _ = _arc_points(s[1], s[2], s[3], steps=1)
                    b = -math.tan(sweep / 4.0)
                pts.append((float(s[1][0] * size), float((1.0 - s[1][1]) * size), b))
            msp.add_lwpolyline(pts, format="xyb", close=True, dxfattribs={"layer": name})
        else:
            pts = [(float(v[0] * size), float((1.0 - v[1]) * size)) for v in r]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": name})
    doc.saveas(path)


def extrude_obj(path, rings, size, depth, stagger):
    """Watertight prisms, one per fragment, as named objects in one OBJ."""
    meshes = []
    for i, (name, r, _segs) in enumerate(rings):
        pts = np.column_stack([r[:, 0] * size, (1.0 - r[:, 1]) * size])
        poly = _clean(Polygon(pts))
        if poly is None:
            continue
        m = trimesh.creation.extrude_polygon(poly, height=max(depth * size, 1e-6))
        m.apply_translation([0, 0, i * stagger * size])
        meshes.append((name, m))
    with open(path, "w") as f:
        f.write(f"# {path.name} - {len(meshes)} closed prisms\n")
        off = 1
        for name, m in meshes:
            f.write(f"\no {name}\ng {name}\n")
            for v in m.vertices:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for a, b, c in m.faces + off:
                f.write(f"f {a} {b} {c}\n")
            off += len(m.vertices)
    return meshes


def extrude_step(path, rings, size, depth, stagger):
    """True polysurfaces: bottom + top + one planar face per polygon segment."""
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire)
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.TopExp import TopExp_Explorer
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.TDocStd import TDocStd_Document
    from OCP.TDataStd import TDataStd_Name
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.Interface import Interface_Static

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ctool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    def P(v, z):
        return gp_Pnt(float(v[0] * size), float((1.0 - v[1]) * size), z)

    stats = []
    for i, (name, r, segs) in enumerate(rings):
        z = i * stagger * size
        if segs:
            # three-point arcs: the midpoint fixes the sweep, so no direction
            # convention has to survive the Y flip
            wire = BRepBuilderAPI_MakeWire()
            for s in segs:
                if s[0] == "A":
                    pts, _, _ = _arc_points(s[1], s[2], s[3], steps=2)
                    arc = GC_MakeArcOfCircle(P(s[1], z), P(pts[1], z), P(s[2], z)).Value()
                    wire.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
                else:
                    wire.Add(BRepBuilderAPI_MakeEdge(P(s[1], z), P(s[2], z)).Edge())
            face = BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()
        else:
            poly = BRepBuilderAPI_MakePolygon()
            for v in r:
                poly.Add(P(v, z))
            poly.Close()
            face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
        solid = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, max(depth * size, 1e-6))).Shape()
        e, nf, ncyl = TopExp_Explorer(solid, TopAbs_FACE), 0, 0
        while e.More():
            nf += 1
            if (BRepAdaptor_Surface(TopoDS.Face_s(e.Current())).GetType()
                    == GeomAbs_SurfaceType.GeomAbs_Cylinder):
                ncyl += 1
            e.Next()
        lbl = tool.AddShape(solid, False)
        TDataStd_Name.Set_s(lbl, TCollection_ExtendedString(name))
        c = PALETTE[i % len(PALETTE)]
        ctool.SetColor(lbl, Quantity_Color(c[0]/255, c[1]/255, c[2]/255,
                                           Quantity_TOC_RGB), XCAFDoc_ColorSurf)
        stats.append((name, len(r), nf, BRepCheck_Analyzer(solid).IsValid(), ncyl))
    Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPCAFControl_Writer()
    w.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs)
    w.Write(str(path))
    return stats


def write_preview(path, rings, meshes, info):
    """Contact sheet: the closed curves, and the solids they extrude into."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13, 6.5), facecolor="white")
    ax = fig.add_subplot(1, 2, 1)
    for i, (name, r, _segs) in enumerate(rings):
        c = np.array(PALETTE[i % len(PALETTE)]) / 255
        ring = np.vstack([r, r[:1]])
        ax.fill(ring[:, 0], ring[:, 1], color=c, alpha=0.35)
        ax.plot(ring[:, 0], ring[:, 1], color=c, lw=1.6)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_axis_off()
    ax.set_title(f"{len(rings)} closed curves", fontsize=9)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    light = np.array([0.4, -0.7, 0.6]); light /= np.linalg.norm(light)
    allv = []
    for i, (name, m) in enumerate(meshes):
        tris = m.vertices[m.faces]
        allv.append(tris.reshape(-1, 3))
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        sh = (0.4 + 0.6 * np.clip(np.abs(n @ light), 0, 1))[:, None]
        c = np.array(PALETTE[i % len(PALETTE)]) / 255
        ax.add_collection3d(Poly3DCollection(tris, facecolors=np.clip(c * sh, 0, 1),
                                             edgecolors="black", linewidths=0.25))
    allv = np.concatenate(allv)
    lo, hi = allv.min(0), allv.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.view_init(elev=32, azim=-62); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"{len(meshes)} closed solids", fontsize=9)

    fig.suptitle(info, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="dinov2_base")
    ap.add_argument("--at", nargs=2, type=float, default=None,
                    metavar=("X", "Y"), help="coordinate; omit to take the emptiest gap")
    ap.add_argument("--composition", default="drift", choices=list(SF.PROTOCOLS))
    ap.add_argument("--shape", default="outline",
                    choices=["outline", "hull", "rect", "ellipse"])
    ap.add_argument("--source", default="raster", choices=["raster", "coseg"],
                    help="raster traces the drawn silhouettes (high resolution); "
                         "coseg uses the stored part polygons (~20 points)")
    ap.add_argument("--trace-tol", type=float, default=1.2, dest="trace_tol",
                    help="contour tracing tolerance in pixels; 0 keeps every pixel step")
    ap.add_argument("--out-size", type=int, default=1080, dest="out_size")
    ap.add_argument("--work-res", type=int, default=512, dest="work_res")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="extra simplification, as a fraction of each fragment's size")
    ap.add_argument("--fillet-rel", type=float, default=0.0, dest="fillet_rel",
                    help="corner radius as a fraction of each fragment's own size; "
                         "large values give the flat cut-out reading")
    ap.add_argument("--sharp-edge", type=float, default=0.0, dest="sharp_edge",
                    help="corners between edges shorter than this fraction of the "
                         "shape's span stay sharp, so small detail survives")
    ap.add_argument("--sharp-angle", type=float, default=0.0, dest="sharp_angle",
                    help="corners tighter than this many degrees stay sharp")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="pre-smooth the traced contour, as a fraction of its points")
    ap.add_argument("--fillet", type=float, default=0.0,
                    help="corner radius as a fraction of the frame; arcs are real, "
                         "and become cylindrical faces on the extrusion")
    ap.add_argument("--min-turn", type=float, default=18.0, dest="min_turn",
                    help="degrees of turn a corner needs before it gets rounded")
    ap.add_argument("--depth", type=float, default=0.04, help="extrusion, fraction of frame")
    ap.add_argument("--stagger", type=float, default=0.006,
                    help="z step per fragment in draw order, fraction of frame")
    ap.add_argument("--size", type=float, default=1000.0, help="frame size in mm")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    fld = G.load_field(a.protocol)
    if a.at:
        x, y = a.at
    else:
        c = G.gap_candidates(fld, n=1, samples=8000, seed=a.seed, sort="empty")[0]
        x, y = c["x"], c["y"]
    lf = G.lift(fld, x, y, k=a.k)

    cfg = SF.PROTOCOLS[a.composition]
    if a.source == "raster":
        forms = R.forms_at(fld, lf, a.work_res, a.out_size, 0.012, 0.45, 12,
                           np.random.default_rng(a.seed), edge="hard")
        if not forms:
            raise SystemExit("no forms cut at this coordinate")
        placed, fit = compose_traced(fld, lf, forms, a.out_size, a.seed, cfg,
                                     tol_px=a.trace_tol)
    else:
        forms = VA.collect(fld, lf, rng=np.random.default_rng(a.seed))
        if not forms:
            raise SystemExit("no part polygons in this neighbourhood")
        placed, fit = place_fitted(forms, cfg, a.seed)
    print(f"{a.protocol} at ({x:.3f}, {y:.3f}) — {a.source} source, {len(forms)} "
          f"available, {len(placed)} placed, fit={fit}")

    rings, rows = [], []
    for i, p in enumerate(placed):
        pts = p["placed"]
        if a.smooth > 0 and a.shape == "outline":
            pts = smooth_ring(pts, a.smooth)
        r = simplify(pts, a.shape, a.tol)
        if r is None:
            print(f"  frag_{i:02d}: dropped, no valid ring")
            continue
        name = f"frag_{i:02d}_part{p['part']}"
        segs = None
        # a radius fixed against the frame over- or under-rounds depending on how
        # big the fragment is; --fillet-rel scales it to each shape's own size
        span = math.sqrt(Polygon(r).area)
        radius = a.fillet_rel * span if a.fillet_rel > 0 else a.fillet
        if radius > 0:
            segs = fillet_ring(r, radius, a.min_turn, sharp_edge=a.sharp_edge,
                               sharp_angle=a.sharp_angle, span=span)
            r = segs_to_ring(segs)
        rings.append((name, r, segs))
        rows.append([name, p["id"], p["period"], p["part"],
                     len(p.get("poly", p["placed"])), len(r),
                     sum(1 for s in segs if s[0] == "A") if segs else 0,
                     f"{Polygon(r).area:.5f}"])

    tag = f"{a.protocol}__x{x:.2f}_y{y:.2f}__{a.composition}__{a.source}_{a.shape}"
    out = a.out or (G.OUT / "assemblage_geometry" / tag)
    out.mkdir(parents=True, exist_ok=True)

    info = f"{a.protocol}  ({x:.3f}, {y:.3f})  {a.composition}/{a.shape}  {len(rings)} fragments"
    write_svg(out / "composition.svg", rings, a.size, info)
    write_dxf(out / "composition.dxf", rings, a.size)
    meshes = extrude_obj(out / "solids.obj", rings, a.size, a.depth, a.stagger)
    stats = extrude_step(out / "solids.step", rings, a.size, a.depth, a.stagger)
    write_preview(out / "preview.png", rings, meshes, info)

    closed = sum(1 for _, m in meshes if m.is_watertight)
    valid = sum(1 for s in stats if s[3])
    print(f"  {len(rings)} closed curves, mean {np.mean([len(r) for _, r, _ in rings]):.1f} "
          f"points (from {np.mean([len(p.get('poly', p['placed'])) for p in placed]):.0f})")
    print(f"  OBJ:  {closed}/{len(meshes)} watertight prisms")
    ncyl = sum(s[4] for s in stats)
    print(f"  STEP: {valid}/{len(stats)} valid solids, "
          f"{np.mean([s[2] for s in stats]):.1f} faces each"
          + (f", {ncyl} cylindrical" if ncyl else ""))

    with open(out / "fragments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "work_id", "period", "coseg_part", "src_points",
                    "curve_points", "arcs", "area_frac"])
        w.writerows(rows)
    (out / "meta.json").write_text(json.dumps({
        "protocol": a.protocol, "x": x, "y": y, "composition": a.composition,
        "shape": a.shape, "tol": a.tol, "fit_scale": fit, "k": a.k, "seed": a.seed,
        "depth": a.depth, "stagger": a.stagger, "size_mm": a.size,
        "source": a.source, "trace_tol": a.trace_tol,
        "fillet": a.fillet, "fillet_rel": a.fillet_rel,
        "smooth": a.smooth, "min_turn": a.min_turn,
        "sharp_edge": a.sharp_edge, "sharp_angle": a.sharp_angle,
        "fragments": len(rings),
    }, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
