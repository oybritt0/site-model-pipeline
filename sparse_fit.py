r"""
sparse_fit.py — the original sparse assemblage, shrunk so nothing leaves the frame.

Everything I tried in sparse_atlas.py rearranged the composition: repulsion,
anchors, growth-to-fill. All of it traded the reference's character for an even
scatter. The reference images get their quality from placement rules that are
already right -- source centroid plus a little jitter, size from influence,
rotation from the form's own principal axis -- which produce a connected run of
overlapping pieces with one or two fragments stranded off on their own, and
whole regions of frame left empty. Nothing about that needs improving.

The only actual problem was that pieces ran off the edge.

So this changes nothing about placement. It runs the ORIGINAL rules verbatim,
onto an oversized transparent layer so pieces that would have fallen off are
kept whole, measures the bounding box of what was actually drawn, and scales
that entire layer down as ONE rigid object until it clears the margin. Relative
positions, overlaps, gaps and stranded outliers all survive exactly; the whole
composition just gets smaller and sits inside the frame.

    python sparse_fit.py --protocols dinov2_base --candidates 1
    python sparse_fit.py --all-protocols --out-size 1080
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import gap_space_core as G
import rift_atlas as R

# the sparse preset, verbatim from batch_assemblage.PRESETS
SPARSE = dict(n_forms=10, scale=1.3, jitter=0.05, coherence=0.0, spin=0.0)
INFLUENCE = 0.7

# Named composition protocols. The invariant across all of them: the whole
# composition is drawn on an oversized layer and then shrunk rigidly, so
# nothing ever touches the frame. What varies is density, looseness, how much
# the layout is taken from model space rather than from the source pictures,
# and orientation.
#
#   position  "source"    where the part sat in its own picture (the reference)
#             "embedding" where its work sits relative to the gap in the model
#             "bearing"   embedding direction for position AND rotation
PROTOCOLS = {
    "drift":    dict(n_forms=10, scale=1.30, jitter=0.05, coherence=0.0, spin=0.0,
                     position="source",
                     note="the reference composition, fitted. Source positions, "
                          "own axis, connected runs with stranded outliers."),
    "vestige":  dict(n_forms=5, scale=1.75, jitter=0.04, coherence=0.0, spin=0.0,
                     position="source",
                     note="very sparse. A few large fragments and a lot of ground."),
    "scatter":  dict(n_forms=10, scale=1.15, jitter=0.15, coherence=0.0, spin=0.0,
                     position="source",
                     note="looser. The same pieces shaken apart, runs broken up."),
    "swarm":    dict(n_forms=24, scale=0.90, jitter=0.09, coherence=0.0, spin=0.0,
                     position="source",
                     note="dense. Many small pieces, overlap doing the work."),
    "survey":   dict(n_forms=12, scale=1.20, jitter=0.03, coherence=0.0, spin=0.0,
                     position="embedding",
                     note="layout taken from model space: each fragment sits "
                          "where its work sits relative to the gap."),
    "bearing":  dict(n_forms=12, scale=1.20, jitter=0.02, coherence=0.0, spin=0.0,
                     position="bearing",
                     note="model space twice over: position AND rotation both "
                          "from the vector between the work and the gap."),
    "levelled": dict(n_forms=12, scale=1.25, jitter=0.05, coherence=1.0, spin=0.0,
                     position="source",
                     note="every form turned upright. Orientation removed, so "
                          "only position and size carry the argument."),
    "tumble":   dict(n_forms=12, scale=1.20, jitter=0.07, coherence=0.0, spin=75.0,
                     position="source",
                     note="orientation randomised past the form's own axis."),
}


_SIG = {}


def signals():
    """Three per-work scores used to bias WHICH neighbours contribute.

      typical      low own-period reconstruction error from vibe_autoencoder --
                   the works that most embody their era's style, because the
                   period's own autoencoder finds them unsurprising
      memetic      how common this work's co-segmentation parts are across the
                   corpus. High = built from widely repeated visual conventions
      architectural  how much its parts are those the architecture category is
                   made of, via the period x part matrix
    """
    if _SIG:
        return _SIG
    import collections
    idx = json.load(open(G.AE_PERIOD / "ae_index.json", encoding="utf-8"))
    sur = {i: v for i, v in zip(idx["ids"], idx["ae_surprise"]) if v is not None}
    vals = np.array(list(sur.values()), dtype=float)
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
    typical = {i: float(np.clip(1.0 - (v - lo) / max(hi - lo, 1e-9), 0, 1))
               for i, v in sur.items()}

    # part frequency across the corpus
    freq = collections.Counter()
    n_work = 0
    parts_of = {}
    for f in (G.COSEG / "labels").glob("*.npy"):
        u = [int(x) for x in np.unique(np.load(f))]
        parts_of[f.stem] = u
        freq.update(u)
        n_work += 1
    fmax = max(freq.values()) if freq else 1
    memetic = {}
    for wid, u in parts_of.items():
        if u:
            memetic[wid] = float(np.mean([freq[x] / fmax for x in u]))

    # which parts architecture is made of
    meta = G.coseg_parts()
    manifest = json.load(open(G.CAT / "manifest.json", encoding="utf-8"))
    arch_parts = collections.Counter()
    for m in manifest:
        if m.get("canonical") and m.get("category") == "architecture":
            arch_parts.update(parts_of.get(m["id"], []))
    amax = max(arch_parts.values()) if arch_parts else 1
    architectural = {}
    for wid, u in parts_of.items():
        if u:
            architectural[wid] = float(np.mean([arch_parts[x] / amax for x in u]))

    _SIG.update(typical=typical, memetic=memetic, architectural=architectural)
    return _SIG


EMPHASES = {
    "even":          "no bias -- distance alone decides who contributes",
    "typical":       "works their own period's autoencoder finds least surprising",
    "memetic":       "works built from the most widely repeated parts",
    "architectural": "works made of the parts architecture is made of",
    "blend":         "typical x memetic x architectural, all three at once",
}


def emphasise(fld, lf, kind, strength=1.6):
    """Reweight the lift so the chosen kind of work contributes more."""
    if kind == "even":
        return lf
    S = signals()
    w = np.array(lf.w, dtype=float)
    for n, i in enumerate(lf.idx):
        wid = fld.ids[i]
        if kind == "blend":
            v = (S["typical"].get(wid, .5) * S["memetic"].get(wid, .5)
                 * S["architectural"].get(wid, .5)) ** (1 / 3)
        else:
            v = S[kind].get(wid, 0.5)
        w[n] *= (0.15 + v) ** strength
    if w.sum() <= 0:
        return lf
    lf.w = w / w.sum()
    return lf


def clash_points(fld, n=6, samples=9000, seed=0, min_steps=4.0, k=12,
                 min_sep=None, mask=None):
    """Gaps where two ideologically OPPOSED periods both have real presence.

    rift_atlas scores polarisation along a single axis. This is more specific,
    and closer to what the collision actually is: take the two heaviest periods
    in the neighbourhood and score their distance across all five axes, times
    the product of their weights so BOTH must genuinely be there. A point that
    is 80% neoclassical and 5% High Tech is not a collision, however far apart
    those two sit.

    Also relaxes min_steps: a clash lives where two territories meet, which is
    often nearer the data than an empty void is.
    """
    import itertools
    A = json.load(open(G.E / "affordance_vocab.json", encoding="utf-8"))["axes"]
    POS = {kk: np.array(v, float) for kk, v in A["positions"].items()}
    names = A["order"]

    rng = np.random.default_rng(seed)
    lo, hi = fld.xy.min(0), fld.xy.max(0)
    pts = rng.uniform(lo, hi, size=(samples, 2))
    try:
        from scipy.spatial import Delaunay
        pts = pts[Delaunay(fld.xy).find_simplex(pts) >= 0]
    except Exception:
        pass
    u = G.step_unit(fld)
    dd, _ = fld.tree().query(pts, k=1)
    pts = pts[dd >= min_steps * u]
    if len(pts) == 0:
        return []

    scored = []
    for q in pts[:3000]:
        lf = G.lift(fld, q[0], q[1], k=k, mask=mask)
        acc = {}
        for i, w in zip(lf.idx, lf.w):
            acc[fld.period[i]] = acc.get(fld.period[i], 0.0) + float(w)
        top = sorted(((pp, w) for pp, w in acc.items() if pp in POS),
                     key=lambda t: -t[1])[:2]
        if len(top) < 2:
            continue
        (pa, wa), (pb, wb) = top
        dist = float(np.linalg.norm(POS[pa] - POS[pb]))
        score = dist * (wa * wb)
        axis = names[int(np.argmax(np.abs(POS[pa] - POS[pb])))]
        scored.append((score, q, pa, pb, wa, wb, dist, axis,
                       sorted(acc.items(), key=lambda t: -t[1])[:4], lf))
    scored.sort(key=lambda t: -t[0])

    if min_sep is None:
        ext = float(np.hypot(*(fld.xy.max(0) - fld.xy.min(0))))
        min_sep = 0.045 * ext
    out = []
    for sc, q, pa, pb, wa, wb, dist, axis, per, lf in scored:
        if any(np.hypot(*(q - o["xy"])) < min_sep for o in out):
            continue
        out.append({"xy": q, "x": float(q[0]), "y": float(q[1]),
                    "steps_out": lf.steps_out,
                    "periods": [[pp, round(float(w), 4)] for pp, w in per],
                    "clash": {"a": pa, "b": pb,
                              "weight_a": round(wa, 4), "weight_b": round(wb, 4),
                              "axis_distance": round(dist, 3),
                              "widest_axis": axis, "score": round(sc, 4)}})
        if len(out) >= n:
            break
    return out


def edge_points(fld, n=6, seed=0, k=12, min_sep=None, mask=None):
    """Edge conditions: gaps just outside the occupied territory.

    Sampled beyond the convex hull but close to it, so the neighbourhood is the
    outermost works in the arrangement rather than the interior. Where the space
    stops having anything to say.
    """
    from scipy.spatial import ConvexHull, Delaunay
    rng = np.random.default_rng(seed)
    hull = ConvexHull(fld.xy)
    tri = Delaunay(fld.xy)
    lo, hi = fld.xy.min(0), fld.xy.max(0)
    span = hi - lo
    pts = rng.uniform(lo - 0.06 * span, hi + 0.06 * span, size=(12000, 2))
    outside = pts[tri.find_simplex(pts) < 0]
    if len(outside) == 0:
        return []
    d, _ = fld.tree().query(outside, k=1)
    u = G.step_unit(fld)
    keep = outside[(d > 2.0 * u) & (d < 14.0 * u)]      # just beyond the edge
    if len(keep) == 0:
        return []
    if min_sep is None:
        min_sep = 0.06 * float(np.hypot(*span))
    order = rng.permutation(len(keep))
    out = []
    for i in order:
        q = keep[i]
        if any(np.hypot(*(q - o["xy"])) < min_sep for o in out):
            continue
        lf = G.lift(fld, q[0], q[1], k=k, mask=mask)
        acc = {}
        for j, w in zip(lf.idx, lf.w):
            acc[fld.period[j]] = acc.get(fld.period[j], 0.0) + float(w)
        out.append({"xy": q, "x": float(q[0]), "y": float(q[1]),
                    "steps_out": lf.steps_out,
                    "periods": [[pp, round(float(w), 4)]
                                for pp, w in sorted(acc.items(),
                                                    key=lambda t: -t[1])[:4]],
                    "clash": None})
        if len(out) >= n:
            break
    return out


def slug(v: float) -> str:
    return f"{v:+.2f}".replace(".", "p").replace("+", "")


def compose_fitted(fld, lf, forms, size, seed, margin_frac=0.055,
                   n_forms=10, scale=1.3, jitter=0.05, coherence=0.0, spin=0.0,
                   influence=INFLUENCE, pad_mult=2.2, position="source"):
    """Original placement on an oversized layer, then one rigid shrink to fit."""
    rng = np.random.default_rng(seed)
    W = int(size * pad_mult)               # room for pieces that would fall off
    off = (W - size) // 2
    layer = np.zeros((W, W, 4), dtype=np.float32)

    pr = np.array([f["w"] for f in forms], dtype=float)
    pr = pr / pr.sum()
    n = int(min(n_forms, len(forms)))
    chosen = [forms[i] for i in rng.choice(len(forms), size=n, replace=False, p=pr)]
    chosen.sort(key=lambda f: -f["area"])          # big behind, small in front
    wmean = float(np.mean([f["w"] for f in chosen])) or 1.0

    # model-space placement: where each work sits relative to the gap
    emb, bear = {}, {}
    if position in ("embedding", "bearing"):
        d = fld.xy[lf.idx] - np.array([lf.x, lf.y])
        mx = float(np.abs(d).max()) or 1.0
        for j, i in enumerate(lf.idx):
            emb[fld.ids[i]] = (0.5 + 0.42 * d[j][0] / mx,
                               0.5 - 0.42 * d[j][1] / mx)
            bear[fld.ids[i]] = math.degrees(math.atan2(-d[j][1], d[j][0]))

    placed = 0
    for f in chosen:
        h0, w0 = f["rgba"].shape[:2]
        if h0 < 2 or w0 < 2:
            continue
        # --- identical to paint_assemblage: influence -> size -----------------
        gain = (f["w"] / wmean) ** influence
        target = f["area"] * (size * size) * (scale ** 2) * gain
        cur = max(1.0, float((f["rgba"][:, :, 3] > 0.3).sum()))
        sc = float(np.clip(np.sqrt(target / cur), 0.15, 3.0))
        if max(h0, w0) * sc > 1.5 * size:
            sc = 1.5 * size / max(h0, w0)
        nh, nw = max(2, int(h0 * sc)), max(2, int(w0 * sc))
        if nh > 4 * size or nw > 4 * size:
            continue
        tile = Image.fromarray((np.clip(f["rgba"], 0, 1) * 255).astype(np.uint8),
                               "RGBA").resize((nw, nh), Image.LANCZOS)
        ang = f["angle"] * (1.0 - coherence)
        if position == "bearing" and f["id"] in bear:
            ang = bear[f["id"]]
        if spin:
            ang += float(rng.uniform(-spin, spin))
        if abs(ang) > 0.5:
            tile = tile.rotate(ang, expand=True, resample=Image.BICUBIC)
        ta = np.asarray(tile, dtype=np.float32) / 255.0
        th, tw = ta.shape[:2]
        if position in ("embedding", "bearing") and f["id"] in emb:
            bx, by = emb[f["id"]]
        else:
            bx, by = f["cx"], f["cy"]
        cx = bx + float(rng.normal(0, jitter))
        cy = by + float(rng.normal(0, jitter))
        x0 = int(round(cx * size - tw / 2)) + off
        y0 = int(round(cy * size - th / 2)) + off
        # ----------------------------------------------------------------------
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(W, x0 + tw), min(W, y0 + th)
        if xs1 <= xs0 or ys1 <= ys0:
            continue
        sub = ta[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        a = sub[:, :, 3:4]
        dst = layer[ys0:ys1, xs0:xs1]
        dst[:, :, :3] = sub[:, :, :3] * a + dst[:, :, :3] * (1 - a)
        dst[:, :, 3:4] = np.maximum(dst[:, :, 3:4], a)
        placed += 1

    ys, xs = np.nonzero(layer[:, :, 3] > 0.02)
    if len(ys) == 0:
        raise RuntimeError("nothing drawn")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = layer[y0:y1, x0:x1]

    # one rigid shrink of the whole composition
    m = int(round(margin_frac * size))
    safe = size - 2 * m
    k = min(safe / crop.shape[1], safe / crop.shape[0])
    nw2, nh2 = max(2, int(crop.shape[1] * k)), max(2, int(crop.shape[0] * k))
    fitted = np.asarray(
        Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8), "RGBA")
        .resize((nw2, nh2), Image.LANCZOS), dtype=np.float32) / 255.0

    canvas = R.background(fld, lf, size, tone=True)
    px = m + (safe - nw2) // 2
    py = m + (safe - nh2) // 2
    a = fitted[:, :, 3:4]
    canvas[py:py + nh2, px:px + nw2] = (
        fitted[:, :, :3] * a + canvas[py:py + nh2, px:px + nw2] * (1 - a))

    # The same composition again, on nothing. Image-to-3D wants a clean matte and
    # the background here is a gradient, so it cannot be recovered by thresholding
    # the finished PNG -- but the alpha was there all along, it was just never
    # saved. `fitted` carries colour composited over black, i.e. premultiplied, so
    # un-premultiply it: otherwise every fragment edge keeps a dark fringe when it
    # is later laid over TripoSR's grey.
    matte = np.zeros((size, size, 4), dtype=np.float32)
    sub = matte[py:py + nh2, px:px + nw2]
    sub[:, :, :3] = fitted[:, :, :3] / np.maximum(a, 1e-4)
    sub[:, :, 3:4] = a
    np.clip(matte, 0, 1, out=matte)
    return canvas, placed, round(float(k), 4), matte


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--categories",
                    default="all,painting_sculpture,architecture,drawing_print")
    ap.add_argument("--points", default="mixed",
                    choices=["mixed", "empty", "clash", "edge"])
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--out-size", type=int, default=1080)
    ap.add_argument("--work-res", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=0.055)
    ap.add_argument("--protocol-set", default="drift",
                    help="comma list of names, or 'all'")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--min-steps", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emphasis", default="even",
                    help="comma list from " + str(list(EMPHASES)))
    ap.add_argument("--k-set", default=None,
                    help="comma list of neighbour counts, e.g. 5,12,28")
    ap.add_argument("--subset", default=None,
                    help="json with an 'ids' list; fragments are cut only from "
                         "those works. Combines with --categories.")
    ap.add_argument("--subset-label", default=None,
                    help="folder tag for the subset; defaults to its filename")
    ap.add_argument("--stamp", default=None,
                    help="reuse an existing run stamp to resume into "
                         "the same folders instead of starting a new set")
    ap.add_argument("--matte", action="store_true",
                    help="also save <name>_matte.png, straight-alpha RGBA of the "
                         "composition with no background. The alpha already "
                         "exists inside compose_fitted; the finished PNG's "
                         "vignette cannot be thresholded back off. Image-to-3D "
                         "needs this.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    protos = ([p.strip() for p in a.protocols.split(",")] if a.protocols
              else [p["id"] for p in G.list_protocols()])
    cats = [c.strip() for c in a.categories.split(",")]

    # A subset restricts which works fragments may be cut from, without
    # touching where the gap points are chosen: the coordinate still means
    # what it means in the full layout, and only the material changes. Checked
    # before use -- a mask this narrow is only worth having if different gap
    # points still lift different works.
    subset_ids = None
    subset_tag = ""
    if a.subset:
        sj = json.loads(Path(a.subset).read_text(encoding="utf-8"))
        subset_ids = set(sj["ids"] if isinstance(sj, dict) else sj)
        subset_tag = (a.subset_label or Path(a.subset).stem
                      .replace("subset_", "")) + "-"
        print(f"subset: {len(subset_ids)} ids from {a.subset}")
    kset = ([int(x) for x in a.k_set.split(",")] if a.k_set else [a.k])
    emphs = [x.strip() for x in a.emphasis.split(",") if x.strip() in EMPHASES]
    if not emphs:
        raise SystemExit(f"emphasis must be from {list(EMPHASES)}")
    names = (list(PROTOCOLS) if a.protocol_set == "all"
             else [x.strip() for x in a.protocol_set.split(",")
                   if x.strip() in PROTOCOLS])
    if not names:
        raise SystemExit(f"no protocol matched; choose from {list(PROTOCOLS)}")
    # --stamp resumes into an existing run's folders; without it every run makes
    # a fresh set beside the old one, which is why the first resume attempt
    # could never have filled the gaps
    stamp = a.stamp or time.strftime('%Y%m%d_%H%M%S')
    root = Path(a.out) if a.out else G.OUT / "sparsefit"
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# sparse, fitted — composition protocols\n\n"
        "The invariant: the composition is drawn on an oversized layer and then "
        "shrunk rigidly, so **nothing ever touches the frame**. Relative "
        "positions, overlaps, gaps and stranded outliers survive exactly — the "
        "whole arrangement simply gets smaller.\n\n"
        "Files are named `<model>__<sort><rank>__<category>__<protocol>`.\n\n"
        + "\n".join(f"- **{k}** — {v['note']}" for k, v in PROTOCOLS.items()
                    if k in names) + "\n",
        encoding="utf-8")

    made, t0 = 0, time.time()
    for pid in protos:
        fld = G.load_field(pid)
        if a.points == "clash":
            cands = clash_points(fld, n=a.candidates, seed=a.seed,
                                 min_steps=a.min_steps, k=a.k)
        elif a.points == "edge":
            cands = edge_points(fld, n=a.candidates, seed=a.seed, k=a.k)
        else:
            cands = G.gap_candidates(fld, n=a.candidates, samples=8000,
                                     seed=a.seed, min_steps=a.min_steps,
                                     sort=a.points)
        for ci, c in enumerate(cands):
            for cat in cats:
                mask = None
                if cat != "all":
                    mask = np.array([q == cat for q in fld.category], bool)
                if subset_ids is not None:
                    sm = np.array([i in subset_ids for i in fld.ids], bool)
                    mask = sm if mask is None else (mask & sm)
                if mask is not None and not mask.any():
                    continue
                # same nesting as the original atlases:
                #   <category root>/<model>/<sort>_<rank>__x<x>_y<y>/<method>.png
                croot = root / f"sparsefit_{subset_tag}{cat}_{stamp}"
                cdir = croot / pid / (f"{a.points}_{ci:02d}__"
                                      f"x{slug(c['x'])}_y{slug(c['y'])}")
                cdir.mkdir(parents=True, exist_ok=True)
                pj = cdir / "point.json"
                if not pj.exists():
                    lf0 = G.lift(fld, c["x"], c["y"], k=a.k, mask=mask)
                    pj.write_text(json.dumps({
                        "protocol": pid, "protocol_display": fld.display_name,
                        "category": cat, "sort": a.points, "rank": ci,
                        "x": c["x"], "y": c["y"],
                        "steps_out": c["steps_out"],
                        "periods": c["periods"],
                        "clash": c.get("clash"),
                        "neighbours": G._neighbour_rows(fld, lf0),
                    }, indent=2), encoding="utf-8")
                for kk in kset:
                 for emph in emphs:
                  lf = emphasise(fld,
                                 G.lift(fld, c["x"], c["y"], k=kk, mask=mask),
                                 emph)
                  forms = None
                  for proto in names:
                   cfg = PROTOCOLS[proto]
                   name = f"k{kk}_{emph}_{proto}"
                   png = cdir / f"{name}.png"
                   if png.exists():
                     continue
                   try:
                     if forms is None:      # cut once, reuse for every protocol
                         rng = np.random.default_rng(a.seed + ci)
                         forms = R.forms_at(fld, lf, a.work_res, a.out_size,
                                            0.012, 0.45, 12, rng, edge="hard")
                     if not forms:
                         break
                     canvas, placed, k, matte = compose_fitted(
                         fld, lf, forms, a.out_size, a.seed + ci,
                         margin_frac=a.margin,
                         n_forms=cfg["n_forms"], scale=cfg["scale"],
                         jitter=cfg["jitter"], coherence=cfg["coherence"],
                         spin=cfg["spin"], position=cfg["position"])
                     Image.fromarray((np.clip(canvas, 0, 1) * 255)
                                     .astype(np.uint8)).save(png)
                     if a.matte:
                         Image.fromarray(
                             (np.clip(matte, 0, 1) * 255).astype(np.uint8),
                             "RGBA").save(cdir / f"{name}_matte.png")
                     (cdir / f"{name}.json").write_text(json.dumps({
                         "method": proto, "note": cfg["note"],
                         "k": kk, "emphasis": emph,
                         "emphasis_note": EMPHASES[emph],
                         "position_from": cfg["position"], "contained": True,
                         "fit_scale": k, "margin_frac": a.margin,
                         "protocol": pid, "category": cat,
                         "point_sort": a.points, "rank": ci,
                         "x": c["x"], "y": c["y"], "steps_out": c["steps_out"],
                         "periods": c["periods"], "clash": c.get("clash"),
                         "matte": f"{name}_matte.png" if a.matte else None,
                         "edge": "hard", "n_forms": cfg["n_forms"],
                         "scale": cfg["scale"], "jitter": cfg["jitter"],
                         "coherence": cfg["coherence"], "spin": cfg["spin"],
                         "influence": INFLUENCE,
                         "forms_available": len(forms), "forms_placed": placed,
                         "out_size": a.out_size, "work_res": a.work_res,
                         "sources": sorted({f["id"] for f in forms}),
                     }, indent=2), encoding="utf-8")
                     made += 1
                     print(f"  [{made}] {cat}/{pid}/{cdir.name}/{name}  "
                           f"{placed} forms  fit={k}  "
                           f"({(time.time()-t0)/60:.1f}m)", flush=True)
                   except Exception as ex:
                     print(f"  FAIL {name}: {type(ex).__name__}: {ex}", flush=True)
    for d in sorted(root.glob(f"sparsefit_*_{stamp}")):
        notes = "\n".join(f"- **{k2}** - {v['note']}"
                          for k2, v in PROTOCOLS.items() if k2 in names)
        (d / "README.md").write_text(
            "# sparse, fitted\n\n"
            "Nothing touches the frame: the composition is drawn on an "
            "oversized layer, then shrunk rigidly until it clears the margin. "
            "Relative positions, overlaps, gaps and stranded outliers survive "
            "exactly.\n\n"
            "Structure matches the original atlases:\n"
            "`<model>/<sort>_<rank>__x_y/<protocol>.png`\n\n"
            + notes + "\n", encoding="utf-8")
    print(f"{made} images in {(time.time()-t0)/60:.1f} min -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
