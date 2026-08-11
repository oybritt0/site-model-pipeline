r"""
tile_model.py -- one 12 x 12 in tile of the site model, at 1/64" = 1'.

Takes the volumes made from each tile coordinate, stands them on a ground
surface derived from the layout itself, unions the lot into one solid, clears the
plot at Site A, clips to the tile, and writes an STL.

  ground        The relief driver is a decision the tile had to make and the
                evidence went against the plan. `tile_stability_check.py`
                measured whether clash ground sits low in cross-protocol
                stability, and in CLIP -- the protocol this tile is built in --
                it does not: the correlation is POSITIVE (spearman +0.21) and
                Site A sits at the 63rd percentile, above the median. So
                stability-driven topography is dropped rather than defended.

                What replaces it is the layout's own occupancy: ground height
                falls with distance to the nearest work. Territory the model has
                filled is plateau; the gaps are basins. Since a site in this
                project is *defined* as a gap, the four sites then sit in the low
                ground by construction rather than by hope, and the relief is
                measured rather than asserted. `--relief stability` still runs,
                for the record.

  standing      TripoSR is given a frontal view, so its canonical +z is the
                composition's own vertical. Kept as up, each mass stands with the
                assemblage as its elevation. `--stand flat` lays the picture
                plane down instead and makes the composition a plan.

  turning       Each mass turns on the bearing from its coordinate to the work
                that claims it hardest -- sparse_fit's `bearing` protocol, one
                level up. Not decoration: it is the same vector the compositions
                already use.

  the void      Site A is a 200 ft plot, which at 1/64" is a 3.1 in circle. It is
                cut clear of volumes and its ground is levelled. Nothing is built
                there; that is what "held in reserve" means.

    python tile_model.py
    python tile_model.py --relief flat --stand flat --out .../variant
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

import gap_space_core as G

OUT = G.OUT / "site_tile"

# 1/64" = 1' -- one model inch is 64 feet. Not 64 x 12; that error put the
# Site A plot at 2381 ft in the first build log.
FT_PER_IN = 64.0


# ---------------------------------------------------------------------------
# ground
# ---------------------------------------------------------------------------

def relief_field(fld, man, n, mode, smooth_in, gamma=1.0):
    """An n x n height field over the tile, in 0..1."""
    w = man["window"]
    ipu = man["in_per_unit"]
    gx = np.linspace(0, man["tile_in"], n)
    gy = np.linspace(0, man["tile_in"], n)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    lx = w["x0"] + X / ipu
    ly = w["y0"] + Y / ipu
    P = np.stack([lx.ravel(), ly.ravel()], 1)

    if mode == "flat":
        return np.zeros((n, n)), {"driver": "flat"}

    if mode == "density":
        d, _ = fld.tree().query(P, k=1)
        v = d.reshape(n, n) / G.step_unit(fld)          # steps to nearest work
        lo, hi = np.percentile(v, [5, 95])
        h = 1.0 - np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        meta = {"driver": "density", "units": "steps to nearest work",
                "p5": float(lo), "p95": float(hi),
                "note": "occupied ground is plateau, gaps are basins"}
    elif mode == "stability":
        d, nn = fld.tree().query(P, k=12)
        s = fld.stability[nn]
        wq = 1.0 / (d ** 2 + 1e-9)
        with np.errstate(invalid="ignore"):
            v = (np.nan_to_num(s, nan=0.0) * wq).sum(1) / wq.sum(1)
        v = v.reshape(n, n)
        lo, hi = np.percentile(v, [5, 95])
        h = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        meta = {"driver": "stability", "units": "IDW cross-protocol agreement",
                "p5": float(lo), "p95": float(hi),
                "note": "kept runnable, but the correlation with clash ground "
                        "is positive in CLIP -- see stability_check.json"}
    else:
        raise SystemExit(f"unknown relief mode {mode}")

    if gamma and gamma != 1.0:
        # drama without thickness: push the field towards plateau-and-escarpment
        # instead of a gentle swell, at the same peak height
        h = np.clip(h, 0, 1) ** gamma
        h = (h - h.min()) / max(h.max() - h.min(), 1e-9)
        meta["gamma"] = gamma
    if smooth_in > 0:
        from scipy import ndimage as ndi
        sig = smooth_in / (man["tile_in"] / n)
        h = ndi.gaussian_filter(h, sig, mode="nearest")
        h = (h - h.min()) / max(h.max() - h.min(), 1e-9)
        meta["smooth_in"] = smooth_in
    return h, meta


def ground_solid(h, tile_in, base_in, relief_in):
    """Height field -> a closed slab. Top from the field, flat bottom, four walls."""
    n = h.shape[0]
    g = np.linspace(0, tile_in, n)
    X, Y = np.meshgrid(g, g, indexing="xy")
    Z = base_in + relief_in * h

    top = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    bot = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], 1)
    V = np.vstack([top, bot])
    off = n * n

    idx = np.arange(n * n).reshape(n, n)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]                 # top
    F += [np.stack([a + off, c + off, b + off], 1),                      # bottom
          np.stack([a + off, d + off, c + off], 1)]

    def wall(i0, i1, flip):
        p, q = i0[:-1], i0[1:]
        f = [np.stack([p, q, q + off], 1), np.stack([p, q + off, p + off], 1)]
        return [x[:, ::-1] if flip else x for x in f]

    F += wall(idx[0, :], None, False)
    F += wall(idx[-1, :], None, True)
    F += wall(idx[:, 0], None, True)
    F += wall(idx[:, -1], None, False)

    m = trimesh.Trimesh(V, np.vstack(F), process=True)
    m.fix_normals()
    if not m.is_watertight:
        raise RuntimeError("ground slab did not close")
    return m


def built_coverage(solid, tile_in, ground_max, grid=600, samples=4_000_000):
    """Fraction of the tile plan standing above the highest possible ground.

    The number that matters for whether the tile reads as a city, and it is not
    the one the footprint circles predict: the masses are lacy, filling only
    about a ninth of their own bounding boxes, so nominal coverage overstates
    built coverage roughly fivefold.
    """
    pts, fid = trimesh.sample.sample_surface(solid, int(samples))
    up = solid.face_normals[fid][:, 2] > 0.2
    if up.sum() < 1000:
        return 0.0
    ix = np.clip((pts[up, 0] / tile_in * (grid - 1)).astype(int), 0, grid - 1)
    iy = np.clip((pts[up, 1] / tile_in * (grid - 1)).astype(int), 0, grid - 1)
    z = np.full((grid, grid), -1.0)
    np.maximum.at(z, (iy, ix), pts[up, 2])
    return float((z > ground_max + 0.02).mean())


def resize(man, band, power):
    """Recompute footprints from the lift mass already recorded per point.

    The first tile clamped 86 of 95 masses against a 0.9-1.6 in band -- 58 on the
    floor, 28 on the ceiling -- while the underlying claim spans 62x. So the
    hierarchy was in the data all along and the clamp was flattening it. Widening
    the band is free: the meshes are unchanged, only their scale.

    Coordinates are untouched on purpose. They key the mesh directories, and the
    volumes are paid for.
    """
    P = [p for p in man["points"] if not p["reserved"]]
    cl = np.array([p["claim"] for p in P], float)
    g = (cl / cl.mean()) ** power
    lo, hi = band
    s = np.clip(0.5 * (lo + hi) * g / g.mean(), lo, hi)
    for p, v in zip(P, s):
        p["size_in"] = float(v)
        p["size_ft"] = round(float(v) * FT_PER_IN, 1)
    return float(s.min()), float(s.max()), int((s <= lo + 1e-6).sum() + (s >= hi - 1e-6).sum())


def network(man, h, tile_in, amp, width_in, min_jaccard, per_node, max_in):
    """Raise causeways between sites that recruit the same works.

    An edge is not drawn because two points are near each other -- that would be
    invented. It is drawn when their lifts overlap: the model itself says these
    two pieces of ground draw on the same constituency. `site_survey.py` already
    uses exactly this Jaccard to decide whether protocols agree about a site, so
    the network is the layout's own account of what is connected to what.
    """
    P = [p for p in man["points"] if not p["reserved"]]
    sets = [set(p["neighbours"]) for p in P]
    xy = np.array([[p["plan_x_in"], p["plan_y_in"]] for p in P])

    cand = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = float(np.hypot(*(xy[i] - xy[j])))
            if d > max_in:
                continue
            u = len(sets[i] | sets[j])
            jac = len(sets[i] & sets[j]) / u if u else 0.0
            if jac > 0:
                cand.append((jac, d, i, j))
    cand.sort(key=lambda t: -t[0])

    deg = {i: 0 for i in range(len(P))}
    edges = []
    for jac, d, i, j in cand:
        if jac < min_jaccard and (deg[i] >= 1 and deg[j] >= 1):
            continue
        if deg[i] >= per_node and deg[j] >= per_node:
            continue
        edges.append((i, j, jac, d))
        deg[i] += 1
        deg[j] += 1

    n = h.shape[0]
    mask = np.zeros((n, n), bool)
    px = xy / tile_in * (n - 1)
    for i, j, _, _ in edges:
        a, b = px[i], px[j]
        steps = max(2, int(np.hypot(*(b - a)) * 1.5))
        t = np.linspace(0, 1, steps)[:, None]
        q = np.round(a + (b - a) * t).astype(int)
        q = np.clip(q, 0, n - 1)
        mask[q[:, 1], q[:, 0]] = True

    from scipy import ndimage as ndi
    d = ndi.distance_transform_edt(~mask) * (tile_in / n)
    ridge = np.clip(1.0 - d / max(width_in, 1e-6), 0, 1) ** 2
    return np.maximum(h, np.clip(h, 0, 1) * (1 - ridge) + amp * ridge), len(edges)


def height_at(h, man, x_in, y_in, base_in, relief_in):
    n = h.shape[0]
    i = int(np.clip(round(y_in / man["tile_in"] * (n - 1)), 0, n - 1))
    j = int(np.clip(round(x_in / man["tile_in"] * (n - 1)), 0, n - 1))
    return base_in + relief_in * float(h[i, j])


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------

def bearing(fld, x, y) -> float:
    """Direction from the coordinate to the work that claims it hardest.
    sparse_fit's `bearing` protocol uses exactly this vector one level down."""
    lf = G.lift(fld, x, y, k=12)
    i = int(lf.idx[int(np.argmax(lf.w))])
    dx, dy = fld.xy[i] - np.array([x, y])
    return float(np.degrees(np.arctan2(dy, dx)))


def place(path, p, fld, h, man, args):
    m = trimesh.load(path, process=False)
    if len(m.faces) == 0:
        return None
    m = trimesh.Trimesh(m.vertices.copy(), m.faces.copy(), process=False)

    if args.stand == "flat":
        # picture plane down: the composition becomes the plan, depth the height
        m.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 1, 0]))

    ext = m.extents
    span = float(max(ext[0], ext[1]))
    if span <= 0:
        return None
    m.apply_scale(p["size_in"] / span)

    if args.rotate == "bearing":
        m.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(bearing(fld, p["x"], p["y"])), [0, 0, 1]))

    # manifold3d rejects anything not closed, and does it quietly enough that a
    # part can vanish from the union without a word. A couple of meshes per run
    # come back open, so repair here and say so rather than find out later.
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        m.repaired = True

    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    z = height_at(h, man, p["plan_x_in"], p["plan_y_in"],
                  args.base_in, args.relief_in)
    m.apply_translation([p["plan_x_in"], p["plan_y_in"],
                         z - args.sink * float(m.extents[2])])
    return m


# ---------------------------------------------------------------------------
# boolean
# ---------------------------------------------------------------------------

def union_all(parts, engine, log=print):
    """Union in a balanced tree. Sequential accumulation makes every step pay
    for the whole accumulated mass; pairing keeps the operands small for longer."""
    lvl = [p for p in parts if p is not None and len(p.faces)]
    r = 0
    while len(lvl) > 1:
        nxt, t0 = [], time.time()
        for i in range(0, len(lvl), 2):
            if i + 1 >= len(lvl):
                nxt.append(lvl[i])
                continue
            nxt.append(_union2(lvl[i], lvl[i + 1], engine))
        r += 1
        log(f"    union round {r}: {len(lvl)} -> {len(nxt)} "
            f"({time.time()-t0:.1f}s, {max(len(x.faces) for x in nxt):,} "
            f"faces largest)")
        lvl = nxt
    return lvl[0]


def _union2(a, b, engine):
    if engine == "manifold":
        try:
            return trimesh.boolean.union([a, b], engine="manifold")
        except Exception as ex:
            # manifold3d refuses non-manifold input outright; pymeshlab's libigl
            # path is slower but tolerates it, so the run does not die here
            print(f"    manifold union failed ({type(ex).__name__}), "
                  f"falling back to pymeshlab")
    return _pymeshlab_boolean(a, b, "generate_boolean_union")


def _pymeshlab_boolean(a, b, filt):
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(a.vertices, a.faces), "a")
    ms.add_mesh(ml.Mesh(b.vertices, b.faces), "b")
    ms.apply_filter(filt, first_mesh=0, second_mesh=1)
    m = ms.current_mesh()
    return trimesh.Trimesh(m.vertex_matrix().copy(), m.face_matrix().copy(),
                           process=False)


def difference(a, b, engine):
    if engine == "manifold":
        return trimesh.boolean.difference([a, b], engine="manifold")
    return _pymeshlab_boolean(a, b, "generate_boolean_difference")


def intersection(a, b, engine):
    if engine == "manifold":
        return trimesh.boolean.intersection([a, b], engine="manifold")
    return _pymeshlab_boolean(a, b, "generate_boolean_intersection")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(OUT / "points.json"))
    ap.add_argument("--meshes", default=str(OUT / "meshes"))
    ap.add_argument("--mc", type=int, default=256)
    ap.add_argument("--mesh-file", default=None, dest="mesh_file",
                    help="filename inside each mesh dir. Defaults to "
                         "mesh_mc<mc>.obj, which is TripoSR's naming; the paid "
                         "API writes mesh_tripo.obj, where a marching-cubes "
                         "resolution means nothing")
    ap.add_argument("--out", default=str(OUT / "tile"))
    ap.add_argument("--relief", default="density",
                    choices=["density", "stability", "flat"])
    ap.add_argument("--size-band", default="1.2,3.6",
                    help="footprint range in inches. The first tile used 0.9-1.6 "
                         "and clamped 86 of 95 masses, flattening a 62x spread "
                         "of lift mass into 1.8x")
    ap.add_argument("--size-power", type=float, default=0.7,
                    help="exponent on lift mass. Higher exaggerates hierarchy")
    ap.add_argument("--relief-gamma", type=float, default=1.6,
                    help="contrast on the height field. Above 1 flattens the "
                         "plateaus and steepens the slopes at the same peak "
                         "height -- drama without thickness")
    ap.add_argument("--network", type=float, default=0.30,
                    help="height of the causeways raised between sites that "
                         "recruit the same works, as a fraction of relief. "
                         "0 disables")
    ap.add_argument("--network-width-in", type=float, default=0.16)
    ap.add_argument("--network-jaccard", type=float, default=0.12)
    ap.add_argument("--network-per-node", type=int, default=3)
    ap.add_argument("--network-max-in", type=float, default=3.5)
    ap.add_argument("--relief-in", type=float, default=0.90,
                    help="0.90 in = 58 ft at 1/64 inch, against masses now much "
                         "taller. Total thickness stays near 1 in / 25 mm, "
                         "which is still castable")
    ap.add_argument("--base-in", type=float, default=0.10,
                    help="2.5 mm. Thin, because the ground was "
                         "89 percent of the first tile material")
    ap.add_argument("--grid", type=int, default=193)
    ap.add_argument("--smooth-in", type=float, default=0.22)
    ap.add_argument("--stand", default="upright", choices=["upright", "flat"])
    ap.add_argument("--rotate", default="bearing", choices=["bearing", "none"])
    ap.add_argument("--sink", type=float, default=0.18,
                    help="fraction of each mass's height buried, so the field "
                         "unions into one solid instead of separate islands")
    ap.add_argument("--void-in", type=float, default=3.1,
                    help="200 ft at 1/64\"")
    ap.add_argument("--no-void-cut", action="store_true",
                    help="do not remove the plot. The site is scribed into the "
                         "ground as a ring instead, so where it goes stays an "
                         "open decision")
    ap.add_argument("--site-mark-in", type=float, default=0.035,
                    help="depth of the scribed ring in inches. 0.035 in is "
                         "0.9 mm, about two nozzle widths, so it survives the "
                         "print")
    ap.add_argument("--site-mark-width-in", type=float, default=0.07)
    ap.add_argument("--engine", default="manifold", choices=["manifold", "pymeshlab"])
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--keep-floaters", action="store_true",
                    help="keep bodies left unattached to the ground; they are "
                         "unprintable, and are dropped by default")
    a = ap.parse_args()

    man = json.loads(Path(a.points).read_text(encoding="utf-8"))
    fld = G.load_field(man["protocol"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = man["tile_in"]
    t0 = time.time()

    band = tuple(float(v) for v in a.size_band.split(","))
    lo, hi, clamped = resize(man, band, a.size_power)
    print(f"footprints {lo:.2f}-{hi:.2f} in "
          f"({lo*FT_PER_IN:.0f}-{hi*FT_PER_IN:.0f} ft), {clamped} at a limit")

    h, hmeta = relief_field(fld, man, a.grid, a.relief, a.smooth_in, a.relief_gamma)
    if a.network > 0 and a.relief != "flat":
        h, nedges = network(man, h, T, a.network, a.network_width_in,
                            a.network_jaccard, a.network_per_node, a.network_max_in)
        hmeta["network"] = {"edges": nedges, "amp_frac": a.network,
                            "width_in": a.network_width_in,
                            "min_jaccard": a.network_jaccard,
                            "per_node": a.network_per_node,
                            "rule": "sites joined where their lifts recruit the "
                                    "same works"}
        print(f"  network: {nedges} causeways between sites sharing constituency")

    # Site A's plot: levelled ground, and later cleared of volumes
    site = next(p for p in man["points"] if p["reserved"])
    gx = np.linspace(0, T, a.grid)
    X, Y = np.meshgrid(gx, gx, indexing="xy")
    R = np.hypot(X - site["plan_x_in"], Y - site["plan_y_in"])
    disc = R <= (a.void_in / 2)
    if a.no_void_cut:
        # The site is marked, not made. Scribing a ring into the height field
        # leaves the terrain and the surrounding masses exactly as the layout
        # put them, so the plot can be drawn by hand later without rebuilding
        # anything -- where the boolean cylinder had already destroyed nine
        # volumes and levelled the ground under them.
        if a.site_mark_in > 0 and a.relief != "flat":
            ring = np.clip(1.0 - np.abs(R - a.void_in / 2)
                           / max(a.site_mark_width_in / 2, 1e-6), 0, 1) ** 2
            h = np.clip(h - (a.site_mark_in / max(a.relief_in, 1e-6)) * ring, 0, 1)
    elif disc.any() and a.relief != "flat":
        h = h.copy()
        h[disc] = float(np.median(h[disc]))
        from scipy import ndimage as ndi
        h = ndi.gaussian_filter(h, 0.6, mode="nearest")

    ground = ground_solid(h, T, a.base_in, a.relief_in)
    print(f"ground: {len(ground.faces)} faces, watertight {ground.is_watertight}, "
          f"relief {a.relief} {a.relief_in} in "
          f"({a.relief_in*FT_PER_IN:.0f} ft), base {a.base_in} in")

    parts, placed, failed_open = [], [], []
    for i, p in enumerate(man["points"]):
        if p["reserved"]:
            continue
        d = Path(a.meshes) / f"{p['mode']}_{i:02d}__x{_s(p['x'])}_y{_s(p['y'])}"
        f = d / (a.mesh_file or f"mesh_mc{a.mc}.obj")
        if not f.exists():
            print(f"  missing {f.name} for {d.name}")
            continue
        m = place(f, p, fld, h, man, a)
        if m is None:
            continue
        if not m.is_watertight:
            print(f"  DROPPED {d.name}: still open after repair "
                  f"({len(m.faces)} faces) -- it would corrupt the union")
            failed_open.append(d.name)
            continue
        parts.append(m)
        placed.append({"i": i, "mode": p["mode"], "dir": d.name,
                       "plan_x_in": p["plan_x_in"], "plan_y_in": p["plan_y_in"],
                       "size_in": p["size_in"], "size_ft": p["size_ft"],
                       "height_in": round(float(m.extents[2]), 3),
                       "height_ft": round(float(m.extents[2]) * FT_PER_IN, 1),
                       "faces": int(len(m.faces)),
                       "watertight": bool(m.is_watertight)})
    if not parts:
        raise SystemExit("no volumes placed -- run tile_triposr.py first")
    nominal = float(sum(np.pi * (p["size_in"] / 2) ** 2 for p in man["points"]
                        if not p["reserved"]) / (T * T))
    print(f"{len(parts)} volumes placed, "
          f"{sum(x['watertight'] for x in placed)} watertight, "
          f"{sum(x['faces'] for x in placed):,} faces total")
    print(f"  heights {min(x['height_in'] for x in placed):.2f}"
          f"-{max(x['height_in'] for x in placed):.2f} in "
          f"({min(x['height_ft'] for x in placed):.0f}"
          f"-{max(x['height_ft'] for x in placed):.0f} ft)")

    print("  massing union ...")
    mass = union_all(parts, a.engine)
    print(f"  mass: {len(mass.faces)} faces, watertight {mass.is_watertight}, "
          f"bodies {mass.body_count}")

    if a.void_in > 0 and not a.no_void_cut:
        top = a.base_in + a.relief_in + 4.0
        cyl = trimesh.creation.cylinder(radius=a.void_in / 2, height=top * 2,
                                        sections=192)
        cyl.apply_translation([site["plan_x_in"], site["plan_y_in"], top])
        mass = difference(mass, cyl, a.engine)
        print(f"  Site A cleared: {a.void_in} in plot "
              f"({a.void_in*FT_PER_IN:.0f} ft) -> {len(mass.faces)} faces")

    print("  ground union ...")
    solid = _union2(mass, ground, a.engine)
    print(f"  solid: {len(solid.faces)} faces, watertight {solid.is_watertight}, "
          f"bodies {solid.body_count}")

    if not a.no_clip:
        zt = solid.bounds[1][2] + 1.0
        box = trimesh.creation.box(extents=[T, T, zt * 2])
        box.apply_translation([T / 2, T / 2, 0])
        solid = intersection(solid, box, a.engine)
        print(f"  clipped to the tile: {len(solid.faces)} faces, "
              f"watertight {solid.is_watertight}, bodies {solid.body_count}")

    # TripoSR leaves debris: every mesh arrives as one mass plus tens of small
    # disconnected specks (9-57 bodies each, measured in the resolution sweep).
    # Anything that ends up still floating free of the ground cannot be printed,
    # so it goes -- but only after the union, because a speck that lands on a
    # neighbour or on the ground is legitimately part of the field, and
    # sparse_fit's stranded outliers are meant to survive.
    dropped = {"bodies": 0, "volume_in3": 0.0}
    if not a.keep_floaters:
        comps = solid.split(only_watertight=False)
        if len(comps) > 1:
            vols = [abs(float(c.volume)) for c in comps]
            j = int(np.argmax(vols))
            dropped = {"bodies": len(comps) - 1,
                       "volume_in3": round(sum(vols) - vols[j], 4),
                       "fraction": round((sum(vols) - vols[j]) / max(sum(vols), 1e-9), 5)}
            solid = comps[j]
            print(f"  dropped {dropped['bodies']} unattached bodies, "
                  f"{100*dropped['fraction']:.2f}% of volume")

    cov = built_coverage(solid, T, a.base_in + a.relief_in)
    print(f"  built coverage {100*cov:.1f}% of the tile plan "
          f"(nominal footprint circles were {100*nominal:.0f}%)")

    tag = Path(a.mesh_file).stem.replace("mesh_", "") if a.mesh_file else f"mc{a.mc}"
    stl = out / f"tile_{a.relief}_{a.stand}_{tag}.stl"
    solid.export(stl)
    # STL stores loose triangles. Re-merging 2.4 M of them on load drops a few
    # degenerate faces and leaves the solid a handful of broken faces short of
    # closed, which manifold3d then refuses -- so the downstream steps read this
    # PLY instead, which keeps the vertex indices exactly as built.
    ply = stl.with_suffix(".ply")
    solid.export(ply)

    vol_in3 = float(solid.volume)
    rec = {
        "scale": man["scale"], "tile_in": T,
        "represents_ft": man["tile_represents_ft"],
        "relief": hmeta, "relief_in": a.relief_in, "base_in": a.base_in,
        "grid": a.grid, "stand": a.stand, "rotate": a.rotate, "sink": a.sink,
        "mc_resolution": a.mc, "engine": a.engine,
        "void": {"centre_in": [site["plan_x_in"], site["plan_y_in"]],
                 "diameter_in": a.void_in, "diameter_ft": round(a.void_in * FT_PER_IN, 1),
                 "cut": not a.no_void_cut,
                 "scribed_depth_in": a.site_mark_in if a.no_void_cut else 0,
                 "site": "A -- medieval x industrial, on labour"},
        "n_volumes": len(parts), "volumes": placed,
        "opened_and_dropped": failed_open,
        "dropped_unattached": dropped,
        "built_coverage": round(cov, 4),
        "nominal_footprint_coverage": round(nominal, 4),
        "faces": int(len(solid.faces)), "vertices": int(len(solid.vertices)),
        "watertight": bool(solid.is_watertight), "bodies": int(solid.body_count),
        "volume_in3": round(vol_in3, 3),
        "volume_cm3": round(vol_in3 * 16.3871, 2),
        "bounds_in": np.round(solid.bounds, 4).tolist(),
        "build_s": round(time.time() - t0, 1),
        "stl": stl.name, "ply": ply.name,
    }
    (out / f"{stl.stem}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"\n{stl}  {vol_in3:.1f} in3 / {vol_in3*16.3871:.0f} cm3  "
          f"in {(time.time()-t0)/60:.1f}m")
    try:
        import tile_render as TR
        TR.contact(solid, out / f"{stl.stem}_views.png")
        print(f"wrote {out / (stl.stem + '_views.png')}")
    except Exception as ex:
        print(f"(render skipped: {type(ex).__name__}: {ex})")
    return 0


def _s(v: float) -> str:
    return f"{v:+.2f}".replace(".", "p").replace("+", "")


if __name__ == "__main__":
    raise SystemExit(main())
