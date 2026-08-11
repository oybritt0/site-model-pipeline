r"""
rift_atlas.py — gap zones chosen and composed by ideological rift, at print size.

Four compositional methods, each a different argument about what a gap IS. All
four select coordinates the same way: not "empty", not merely "period-mixed",
but TORN -- a neighbourhood where two codifications pull opposite ways on one
axis of who architecture is for (affordance_vocab.json -> axes).

    rift        the drawing is split by the axis that divides it. The two poles'
                parts are placed on opposite sides of a seam whose angle is the
                axis itself. The rift is the composition, not a caption on it.

    stratum     one part-type only, cut from every work in the neighbourhood and
                stacked oldest to newest. A core sample: how a single element is
                re-codified as the values around it change.

    palimpsest  memetic weighting. A part that recurs across many neighbours is
                convention and is drawn heavy; a part appearing once is a
                mutation and is drawn faint. What survives copying is what the
                period agreed to keep.

    negative    the void. Parts are used as apertures cut from a field rather
                than as figures placed on one, so what is drawn is what the
                neighbourhood WITHHOLDS -- the field minus its own vocabulary.

Resolution. The old atlases were 768px, which is 32 ppi at 24 inches -- unusable
in print. Masks are computed at `work_res` (cheap, and the coseg grid is only
52x52 so more is wasted) while pixels are sampled from the source at full size
and composited at `out_size`. 3072px is 128 ppi at 24 inches, which holds up for
a large print viewed at arm's length; --out-size 4096 gives 170 ppi.

    python rift_atlas.py --method rift --n 40 --out-size 3072
    python rift_atlas.py --all --candidates 6
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

OUT = G.OUT
_AX: dict = {}


def axes():
    if not _AX:
        d = json.load(open(G.E / "affordance_vocab.json", encoding="utf-8"))
        _AX.update(names=d["axes"]["order"],
                   defs=d["axes"]["definitions"],
                   pos={k: np.array(v, dtype=float)
                        for k, v in d["axes"]["positions"].items()},
                   periods=d["periods"])
    return _AX


# ---------------------------------------------------------------------------
# Selecting torn ground
# ---------------------------------------------------------------------------

def rift_score(fld: G.Field, lf: G.Lift):
    """Weighted polarisation per axis. High when the neighbourhood's weight sits
    at BOTH ends of one axis rather than spread or clumped in the middle."""
    A = axes()
    P, W = [], []
    for i, w in zip(lf.idx, lf.w):
        v = A["pos"].get(fld.period[i])
        if v is not None:
            P.append(v)
            W.append(w)
    if len(P) < 2:
        return None
    P = np.stack(P)
    W = np.array(W)
    W = W / W.sum()
    mean = (P * W[:, None]).sum(0)
    var = (W[:, None] * (P - mean) ** 2).sum(0)
    # bimodality: weight actually AT the poles, not merely spread
    pole = np.array([float(W[P[:, a] > 0.3].sum() * W[P[:, a] < -0.3].sum())
                     for a in range(P.shape[1])])
    score = np.sqrt(var) * (0.25 + 4.0 * pole)
    a = int(np.argmax(score))
    return {"axis": A["names"][a], "axis_index": a,
            "score": float(score[a]), "spread": float(np.sqrt(var[a])),
            "pole_product": float(pole[a]),
            "neg_pole": A["defs"][A["names"][a]]["neg"],
            "pos_pole": A["defs"][A["names"][a]]["pos"]}


def torn_points(fld: G.Field, n=8, samples=7000, seed=0, min_steps=6.0,
                k=12, min_sep=None) -> list:
    """Gap coordinates ranked by how torn they are, spatially thinned."""
    rng = np.random.default_rng(seed)
    lo, hi = fld.xy.min(0), fld.xy.max(0)
    pts = rng.uniform(lo, hi, size=(samples, 2))
    try:
        from scipy.spatial import Delaunay
        pts = pts[Delaunay(fld.xy).find_simplex(pts) >= 0]
    except Exception:
        pass
    u = G.step_unit(fld)
    d, _ = fld.tree().query(pts, k=1)
    pts = pts[d >= min_steps * u]
    if len(pts) == 0:
        return []
    scored = []
    for p in pts[:2500]:
        lf = G.lift(fld, p[0], p[1], k=k)
        r = rift_score(fld, lf)
        if r:
            scored.append((r["score"], p, r, lf))
    scored.sort(key=lambda t: -t[0])
    if min_sep is None:
        ext = float(np.hypot(*(fld.xy.max(0) - fld.xy.min(0))))
        min_sep = 0.05 * ext
    out = []
    for s, p, r, lf in scored:
        if any(np.hypot(*(p - o["xy"])) < min_sep for o in out):
            continue
        per: dict = {}
        for i, w in zip(lf.idx, lf.w):
            per[fld.period[i]] = per.get(fld.period[i], 0.0) + float(w)
        out.append({"xy": p, "x": float(p[0]), "y": float(p[1]), "rift": r,
                    "steps_out": lf.steps_out,
                    "periods": sorted(per.items(), key=lambda kv: -kv[1])[:4]})
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Selective form collection
# ---------------------------------------------------------------------------

def forms_at(fld, lf, work_res, out_size, min_frac, max_frac, max_per_work,
             rng, edge="hard", min_contrast=0.06, part_filter=None):
    """Cut forms once, at work_res for the mask and out_size for the pixels."""
    from scipy import ndimage as ndi
    meta = G.coseg_parts()
    kept = [p for p in meta["kept"] if part_filter is None or p in part_filter]
    out = []
    for i, w in zip(lf.idx, lf.w):
        wid, path = fld.ids[i], fld.path[i]
        f = G.COSEG / "labels" / f"{wid}.npy"
        if not path or not f.exists():
            continue
        lab = np.load(f).astype(np.int32)
        g = lab.shape[0]
        try:
            hi = G._load_square(path, out_size)          # full-res material
        except Exception:
            continue
        sp = sp_ids = None
        if edge == "hard":
            from skimage.segmentation import slic
            small = np.asarray(Image.fromarray((hi * 255).astype(np.uint8))
                               .resize((work_res, work_res), Image.LANCZOS),
                               dtype=np.float32) / 255.0
            sp = slic(small, n_segments=1200, compactness=9.0, sigma=1.0,
                      start_label=0, channel_axis=2)
            sp_ids = np.unique(sp)
        got = []
        for p in kept:
            m = lab == p
            if not m.any():
                continue
            m = ndi.binary_opening(ndi.binary_closing(m, np.ones((2, 2))), np.ones((2, 2)))
            cc, nc = ndi.label(m)
            for ci in range(1, nc + 1):
                comp = cc == ci
                frac = float(comp.sum()) / (g * g)
                if frac < min_frac or frac > max_frac:
                    continue
                ys, xs = np.nonzero(comp)
                pts = np.stack([xs.astype(float), ys.astype(float)])
                cov = np.cov(pts) if pts.shape[1] > 2 else np.eye(2)
                ev, evec = np.linalg.eigh(cov)
                ang = float(np.degrees(np.arctan2(evec[1, -1], evec[0, -1])))
                cxg, cyg = float(xs.mean() / g), float(ys.mean() / g)

                up = np.asarray(Image.fromarray((comp * 255).astype(np.uint8), "L")
                                .resize((work_res, work_res), Image.BILINEAR),
                                dtype=np.float32) / 255.0
                if edge == "hard":
                    cov_sp = ndi.mean(up, labels=sp, index=sp_ids)
                    keep = np.zeros(int(sp_ids.max()) + 1, dtype=bool)
                    keep[sp_ids] = cov_sp >= 0.45
                    hard = ndi.binary_fill_holes(
                        ndi.binary_closing(keep[sp], np.ones((3, 3))))
                    c2, n2 = ndi.label(hard)
                    if n2 > 1:
                        sz = ndi.sum(hard, c2, range(1, n2 + 1))
                        hard = c2 == (int(np.argmax(sz)) + 1)
                    a_small = ndi.gaussian_filter(hard.astype(np.float32), 0.6)
                else:
                    a_small = np.clip((ndi.gaussian_filter(
                        up, max(1.0, work_res / (g * 1.6))) - 0.30) / 0.45, 0, 1)

                alpha = np.asarray(Image.fromarray((a_small * 255).astype(np.uint8), "L")
                                   .resize((out_size, out_size), Image.BILINEAR),
                                   dtype=np.float32) / 255.0
                ay, ax = np.nonzero(alpha > 0.02)
                if len(ay) < 64:
                    continue
                y0, y1, x0, x1 = ay.min(), ay.max() + 1, ax.min(), ax.max() + 1
                al = alpha[y0:y1, x0:x1]
                crop = hi[y0:y1, x0:x1]
                inside = al > 0.5
                if inside.sum() < 64:
                    continue
                gl = crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
                contrast = float(gl[inside].std())
                if contrast < min_contrast:
                    continue
                got.append({"rgba": np.dstack([crop, al]), "part": int(p),
                            "w": float(w), "cx": cxg, "cy": cyg, "area": frac,
                            "angle": ang, "contrast": contrast, "id": wid,
                            "period": fld.period[i]})
        if len(got) > max_per_work:
            pr = np.array([o["area"] * (0.25 + o["contrast"]) for o in got])
            pr = pr / pr.sum()
            got = [got[j] for j in rng.choice(len(got), max_per_work,
                                              replace=False, p=pr)]
        out.extend(got)
    return out


def paste(canvas, rgba, cx, cy, scale, angle, size):
    h0, w0 = rgba.shape[:2]
    nh, nw = max(2, int(h0 * scale)), max(2, int(w0 * scale))
    if nh > 4 * size or nw > 4 * size:
        return 0
    t = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA") \
        .resize((nw, nh), Image.LANCZOS)
    if abs(angle) > 0.5:
        t = t.rotate(angle, expand=True, resample=Image.BICUBIC)
    ta = np.asarray(t, dtype=np.float32) / 255.0
    th, tw = ta.shape[:2]
    x0, y0 = int(round(cx * size - tw / 2)), int(round(cy * size - th / 2))
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(size, x0 + tw), min(size, y0 + th)
    if xs1 <= xs0 or ys1 <= ys0:
        return 0
    sub = ta[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    a = sub[:, :, 3:4]
    canvas[ys0:ys1, xs0:xs1] = sub[:, :, :3] * a + canvas[ys0:ys1, xs0:xs1] * (1 - a)
    return 1


def background(fld, lf, size, tone=True):
    base, wsum = np.zeros((size, size, 3), np.float32), 0.0
    for i, w in zip(lf.idx, lf.w):
        if not fld.path[i]:
            continue
        try:
            base += float(w) * G._load_square(fld.path[i], size)
            wsum += float(w)
        except Exception:
            pass
    from scipy import ndimage as ndi
    c = base / max(wsum, 1e-6)
    c = ndi.gaussian_filter(c, (size / 24.0, size / 24.0, 0))
    if tone:
        g = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        c = np.repeat(g[:, :, None], 3, 2)
    return 0.60 + 0.40 * c


# ---------------------------------------------------------------------------
# The four methods
# ---------------------------------------------------------------------------

def compose(method, fld, lf, forms, pt, size, seed, n_forms=26):
    rng = np.random.default_rng(seed)
    A = axes()
    ax = pt["rift"]["axis_index"]
    canvas = background(fld, lf, size, tone=(method != "palimpsest"))
    placed = 0

    if method == "classic":
        placed = _classic(canvas, forms, rng, size, n_forms)

    elif method == "rift":
        # each form is pushed to the side of the seam its period sits on
        theta = math.radians(20 + 140 * (pt["rift"]["score"] % 1.0))
        nx, ny = math.cos(theta), math.sin(theta)
        pool = [f for f in forms if A["pos"].get(f["period"]) is not None]
        if not pool:
            pool = forms
        pr = np.array([f["w"] for f in pool]); pr = pr / pr.sum()
        pick = [pool[i] for i in rng.choice(len(pool), min(n_forms, len(pool)),
                                            replace=False, p=pr)]
        pick.sort(key=lambda f: -f["area"])
        for f in pick:
            v = A["pos"].get(f["period"])
            side = float(np.sign(v[ax])) if v is not None else 0.0
            off = 0.19 * side
            cx = np.clip(f["cx"] + off * nx + rng.normal(0, 0.03), 0.05, 0.95)
            cy = np.clip(f["cy"] + off * ny + rng.normal(0, 0.03), 0.05, 0.95)
            s = 1.7 * math.sqrt(max(f["w"], 1e-6) / max(np.mean([g["w"] for g in pick]), 1e-9))
            placed += paste(canvas, f["rgba"], cx, cy, np.clip(s, .2, 3.), f["angle"], size)

    elif method == "stratum":
        # one part-type, ordered oldest to newest across the sheet
        by = {}
        for f in forms:
            by.setdefault(f["part"], []).append(f)
        part = max(by, key=lambda p: sum(g["w"] for g in by[p]))
        pool = sorted(by[part], key=lambda f: f["period"])
        pick = pool[:n_forms]
        for j, f in enumerate(pick):
            cy = (j + 0.5) / max(1, len(pick))
            cx = 0.5 + rng.normal(0, 0.06)
            s = 2.0 * math.sqrt(max(f["w"], 1e-6) /
                                max(np.mean([g["w"] for g in pick]), 1e-9))
            placed += paste(canvas, f["rgba"], cx, cy, np.clip(s, .2, 3.), 0.0, size)

    elif method == "palimpsest":
        # frequency of a part across the neighbourhood decides its weight
        freq: dict = {}
        for f in forms:
            freq[f["part"]] = freq.get(f["part"], 0) + 1
        mx = max(freq.values()) if freq else 1
        pr = np.array([f["w"] * (freq[f["part"]] / mx) ** 1.5 for f in forms])
        pr = pr / pr.sum()
        pick = [forms[i] for i in rng.choice(len(forms), min(n_forms, len(forms)),
                                             replace=False, p=pr)]
        pick.sort(key=lambda f: freq[f["part"]])
        for f in pick:
            rel = freq[f["part"]] / mx
            r = f["rgba"].copy()
            r[:, :, 3] *= 0.18 + 0.82 * rel        # rare = faint, common = solid
            s = (1.2 + 1.4 * rel)
            placed += paste(canvas, r, f["cx"] + rng.normal(0, .05),
                            f["cy"] + rng.normal(0, .05),
                            np.clip(s, .2, 3.), f["angle"] * (1 - rel), size)

    elif method == "negative":
        # forms become apertures cut from a field: what the neighbourhood withholds
        field = np.zeros((size, size, 3), np.float32)
        field[:] = 0.08
        mask = np.zeros((size, size), np.float32)
        pr = np.array([f["w"] for f in forms]); pr = pr / pr.sum()
        pick = [forms[i] for i in rng.choice(len(forms), min(n_forms, len(forms)),
                                             replace=False, p=pr)]
        for f in pick:
            tmp = np.zeros((size, size, 3), np.float32)
            a = np.zeros((size, size), np.float32)
            r = f["rgba"].copy()
            r[:, :, :3] = 1.0
            c2 = np.dstack([tmp, a])
            buf = np.zeros((size, size, 3), np.float32)
            if paste(buf, r, f["cx"], f["cy"], 1.9, f["angle"], size):
                mask = np.maximum(mask, buf[:, :, 0])
                placed += 1
        canvas = field * (1 - mask[:, :, None]) + canvas * mask[:, :, None]

    return canvas, placed


def _classic(canvas, forms, rng, size, n_forms, scale=1.7, influence=0.7,
             jitter=0.07):
    """The original assemblage placement: position from where the part sat in
    its source, scale from the source work's lift weight, rotation from the
    part's own principal axis. Reproduced here only so the existing atlases can
    be re-made at print resolution through the high-res path."""
    pr = np.array([f["w"] for f in forms]); pr = pr / pr.sum()
    pick = [forms[i] for i in rng.choice(len(forms), min(n_forms, len(forms)),
                                         replace=False, p=pr)]
    pick.sort(key=lambda f: -f["area"])
    wm = float(np.mean([f["w"] for f in pick])) or 1.0
    placed = 0
    for f in pick:
        s_ = scale * math.sqrt((f["w"] / wm) ** influence)
        placed += paste(canvas, f["rgba"], f["cx"] + rng.normal(0, jitter),
                        f["cy"] + rng.normal(0, jitter),
                        float(np.clip(s_, .2, 3.)), f["angle"], size)
    return placed


METHODS = {
    "classic":    "the original assemblage placement, re-made at print resolution",
    "rift":       "split by the ideological axis that divides the neighbourhood",
    "stratum":    "one part-type across the periods contesting this point",
    "palimpsest": "weighted by how often a part recurs: convention against mutation",
    "negative":   "the void -- what the neighbourhood withholds, not what it holds",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default=None, choices=list(METHODS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--categories", default="all,painting_sculpture,architecture,drawing_print")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--out-size", type=int, default=3072)
    ap.add_argument("--work-res", type=int, default=1024)
    ap.add_argument("--edges", default="hard")
    ap.add_argument("-k", type=int, default=12)
    ap.add_argument("--n-forms", type=int, default=26)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-steps", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--points", default="rift",
                    choices=["rift", "mixed", "empty"],
                    help="rift = torn ground; mixed/empty = the original "
                         "gap_candidates sorts")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    methods = list(METHODS) if a.all else [a.method or "rift"]
    protos = ([p.strip() for p in a.protocols.split(",")] if a.protocols
              else [p["id"] for p in G.list_protocols()])
    cats = [c.strip() for c in a.categories.split(",")]
    edges = [e.strip() for e in a.edges.split(",")]

    for method in methods:
        root = Path(a.out) if a.out else OUT / f"rift_{method}_{time.strftime('%Y%m%d_%H%M%S')}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "README.md").write_text(
            f"# rift atlas — {method}\n\n**{METHODS[method]}**\n\n"
            f"Coordinates are chosen by ideological rift: a neighbourhood where two\n"
            f"codifications pull opposite ways on one axis of who architecture is for\n"
            f"(`affordance_vocab.json` -> `axes`), not merely where periods differ.\n\n"
            f"{a.out_size}px = {a.out_size/24:.0f} ppi at 24 inches.\n"
            f"Masks computed at {a.work_res}px; pixels sampled from source at full size.\n\n"
            f"Every image has a `.json` beside it with the coordinate, the torn axis,\n"
            f"its two poles, the period mix and the source works.\n", encoding="utf-8")
        made, t0 = 0, time.time()
        print(f"=== {method} -> {root.name} ===", flush=True)

        for pid in protos:
            fld = G.load_field(pid)
            if a.points == "rift":
                pts = torn_points(fld, n=a.candidates, seed=a.seed,
                                  min_steps=a.min_steps, k=a.k)
            else:
                pts = []
                for c in G.gap_candidates(fld, n=a.candidates, samples=8000,
                                          seed=a.seed, min_steps=a.min_steps,
                                          sort=a.points):
                    lf0 = G.lift(fld, c["x"], c["y"], k=a.k)
                    r = rift_score(fld, lf0) or {"axis": "n/a", "axis_index": 0,
                                                 "score": 0.0, "spread": 0.0,
                                                 "pole_product": 0.0,
                                                 "neg_pole": "", "pos_pole": ""}
                    pts.append({"xy": np.array([c["x"], c["y"]]), "x": c["x"],
                                "y": c["y"], "rift": r,
                                "steps_out": c["steps_out"],
                                "periods": c["periods"]})
            if not pts:
                print(f"  [{pid}] no torn point found"); continue
            for ci, pt in enumerate(pts):
                for cat in cats:
                    mask = None
                    if cat != "all":
                        mask = np.array([c == cat for c in fld.category], bool)
                        if not mask.any():
                            continue
                    lf = G.lift(fld, pt["x"], pt["y"], k=a.k, mask=mask)
                    for edge in edges:
                        # --points MUST be in the name: without it a
                        # second pass over the same protocol/rank/category
                        # collides with the first and is silently skipped
                        # as "already exists".
                        name = f"{pid}__{a.points}{ci:02d}__{cat}__{edge}"
                        png = root / f"{name}.png"
                        if png.exists():
                            continue
                        if a.dry_run:
                            print(f"  would make {name}"); continue
                        try:
                            rng = np.random.default_rng(a.seed + ci)
                            forms = forms_at(fld, lf, a.work_res, a.out_size,
                                             0.010, 0.45, 10, rng, edge=edge)
                            if not forms:
                                continue
                            canvas, placed = compose(method, fld, lf, forms, pt,
                                                     a.out_size, a.seed + ci,
                                                     n_forms=a.n_forms)
                            Image.fromarray((np.clip(canvas, 0, 1) * 255)
                                            .astype(np.uint8)).save(png)
                            rec = {"method": method, "framing": METHODS[method],
                                   "protocol": pid, "category": cat, "edge": edge,
                                   "x": pt["x"], "y": pt["y"],
                                   "steps_out": pt["steps_out"],
                                   "rift": pt["rift"],
                                   "periods": [[p, round(w, 4)] for p, w in pt["periods"]],
                                   "forms_available": len(forms),
                                   "forms_placed": placed,
                                   "out_size": a.out_size, "work_res": a.work_res,
                                   "ppi_at_24in": round(a.out_size / 24.0, 1),
                                   "sources": sorted({f["id"] for f in forms})}
                            (root / f"{name}.json").write_text(
                                json.dumps(rec, indent=2), encoding="utf-8")
                            made += 1
                            print(f"  [{made}] {name}  {placed} forms  "
                                  f"axis={pt['rift']['axis']}  "
                                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
                        except Exception as ex:
                            print(f"  FAIL {name}: {type(ex).__name__}: {ex}", flush=True)
        print(f"  {made} images in {(time.time()-t0)/60:.1f} min -> {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
