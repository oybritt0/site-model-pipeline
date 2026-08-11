r"""
tile_points.py -- choose the coordinates for one 12 x 12 in tile of the site model.

The site model is a model of the field of a model space. Its plan IS the
protocol's 2D layout, so nothing here invents a composition: it picks
coordinates inside a window of that layout and converts them to inches.

Scale, fixed by EXHIBITION_SITES.md 4:

    1/64" = 1'      2.7586 in per layout unit      12 in tile = 4.35 units
                    the tile therefore represents 768 x 768 ft

The window is centred on Site A (medieval x industrial, on labour) at CLIP
x 8.30 / y 8.01, so the tile contains the anchor site and tests the void
condition as well as the field.

Point modes are sparse_fit's, restricted to the window:

    clash    two ideologically opposed periods both really present
    mixed    the most period-mixed neighbourhood
    empty    the biggest vacancies

`edge` is not available here and is not a bug: edge points are sampled outside
the convex hull, and 95% of this window is inside it. A tile cut from the
interior of the field has no edge condition in it.

Footprint follows the same rule that sizes fragments inside a composition, one
level up. There, a fragment's size comes from its work's lift weight against the
mean. Here, a site's size comes from its own lift mass -- the unnormalised
sum of 1/d^2 over the k works that claim it -- against the mean, raised to the
same INFLUENCE, then clamped into the 1.0-1.5 in band the tile was specified at.

    python tile_points.py
    python tile_points.py -n 32 --seed 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import gap_space_core as G
import sparse_fit as SF

# Defaults are the anchor tile: CLIP, centred on Site A. Both are arguments now,
# because the first tile landed in architecture territory -- 724 works in the
# window and not one drawing -- and the fix is a different protocol and centre,
# not a different method. See protocol_mix.json.
PROTOCOL = "clip_vitl14_laion2b"
SITE_A = (8.30, 8.01)

SCALE_DEN = 64.0                       # 1/64" = 1'
TILE_IN = 12.0
TILE_UNITS = 4.35                      # 9% of the ~14.5-unit field, both axes
IN_PER_UNIT = TILE_IN / TILE_UNITS     # 2.7586
FT_PER_IN = 12.0 * SCALE_DEN           # 768 ft across the tile

OUT = G.OUT / "site_tile"


def window(centre=SITE_A, span=TILE_UNITS):
    h = span / 2.0
    return (centre[0] - h, centre[0] + h, centre[1] - h, centre[1] + h)


def claim(fld, lf) -> float:
    """The lift's mass before it is normalised: how hard this ground is claimed.

    lf.w sums to 1 by construction and so says nothing about a point's absolute
    standing. The kernel it was built from does -- 1/d^2 over the k nearest
    works, exactly as gap_space_core.lift computes it.
    """
    return float((1.0 / (np.asarray(lf.d, float) ** 2.0 + 1e-9)).sum())


def diversity(fld, lf) -> float:
    """Period entropy of the neighbourhood, on gap_candidates' fixed denominator
    so a point mixing five periods outranks one mixing two."""
    per = np.array(fld.period, dtype=object)[lf.idx]
    _, cnt = np.unique(per, return_counts=True)
    if len(cnt) < 2:
        return 0.0
    p = cnt / cnt.sum()
    denom = math.log(max(2, min(len(lf.idx), len(set(fld.period)))))
    return float(-(p * np.log(p)).sum() / denom)


def clash_score(fld, lf, POS, names):
    acc = {}
    for i, w in zip(lf.idx, lf.w):
        acc[fld.period[i]] = acc.get(fld.period[i], 0.0) + float(w)
    top = sorted(((pp, w) for pp, w in acc.items() if pp in POS),
                 key=lambda t: -t[1])[:2]
    if len(top) < 2:
        return 0.0, None, sorted(acc.items(), key=lambda t: -t[1])[:4]
    (pa, wa), (pb, wb) = top
    dist = float(np.linalg.norm(POS[pa] - POS[pb]))
    axis = names[int(np.argmax(np.abs(POS[pa] - POS[pb])))]
    return (dist * wa * wb,
            {"a": pa, "b": pb, "weight_a": round(wa, 4), "weight_b": round(wb, 4),
             "axis_distance": round(dist, 3), "widest_axis": axis,
             "score": round(dist * wa * wb, 4)},
            sorted(acc.items(), key=lambda t: -t[1])[:4])


def survey(fld, win, samples, seed, min_steps, k):
    """Score every admissible point in the window, once."""
    A = json.load(open(G.E / "affordance_vocab.json", encoding="utf-8"))["axes"]
    POS = {kk: np.array(v, float) for kk, v in A["positions"].items()}
    names = A["order"]

    rng = np.random.default_rng(seed)
    pts = rng.uniform([win[0], win[2]], [win[1], win[3]], size=(samples, 2))
    try:
        from scipy.spatial import Delaunay
        pts = pts[Delaunay(fld.xy).find_simplex(pts) >= 0]
    except Exception:
        pass
    u = G.step_unit(fld)
    d, _ = fld.tree().query(pts, k=1)
    pts = pts[d >= min_steps * u]

    rows = []
    for q in pts:
        lf = G.lift(fld, q[0], q[1], k=k)
        sc, cl, per = clash_score(fld, lf, POS, names)
        rows.append({"x": float(q[0]), "y": float(q[1]),
                     "steps_out": round(lf.steps_out, 3),
                     "claim": claim(fld, lf),
                     "diversity": round(diversity(fld, lf), 4),
                     "clash_score": round(sc, 4), "clash": cl,
                     "periods": [[p, round(float(w), 4)] for p, w in per],
                     "neighbours": [fld.ids[i] for i in lf.idx]})
    return rows


def described(fld, x, y, k, POS=None, names=None):
    if POS is None:
        A = json.load(open(G.E / "affordance_vocab.json", encoding="utf-8"))["axes"]
        POS = {kk: np.array(v, float) for kk, v in A["axes"]["positions"].items()} \
            if "axes" in A else {kk: np.array(v, float) for kk, v in A["positions"].items()}
        names = A["order"]
    lf = G.lift(fld, x, y, k=k)
    sc, cl, per = clash_score(fld, lf, POS, names)
    return {"x": float(x), "y": float(y), "steps_out": round(lf.steps_out, 3),
            "claim": claim(fld, lf), "diversity": round(diversity(fld, lf), 4),
            "clash_score": round(sc, 4), "clash": cl,
            "periods": [[p, round(float(w), 4)] for p, w in per],
            "neighbours": [fld.ids[i] for i in lf.idx]}


def pick(rows, chosen, key, n, min_sep):
    """Greedy: best by `key`, refusing anything too near an already-taken point."""
    got = []
    for r in sorted(rows, key=lambda t: -t[key]):
        if len(got) >= n:
            break
        if any(math.hypot(r["x"] - c["x"], r["y"] - c["y"]) < min_sep
               for c in chosen + got):
            continue
        got.append(dict(r, mode=key.replace("_score", "")))
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=32, help="volumes in the tile")
    ap.add_argument("--samples", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-steps", type=float, default=3.0)
    ap.add_argument("-k", type=int, default=12)
    # the masses have to overlap or the tile cannot union into one solid, so the
    # separation floor is deliberately smaller than a footprint
    ap.add_argument("--min-sep-in", type=float, default=0.85)
    ap.add_argument("--size-min-in", type=float, default=1.0)
    ap.add_argument("--size-max-in", type=float, default=1.5)
    ap.add_argument("--mix", default="clash:5,mixed:14,empty:12",
                    help="how many of each mode, after Site A")
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--centre", default=None,
                    help="x,y in the protocol's own layout. Defaults to Site A, "
                         "which is only meaningful in CLIP")
    ap.add_argument("--site-label", default=None,
                    help="what the reserved point is, for the sidecar")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)

    centre = (tuple(float(v) for v in a.centre.split(",")) if a.centre else SITE_A)
    fld = G.load_field(a.protocol)
    win = window(centre)
    min_sep = a.min_sep_in / IN_PER_UNIT

    rows = survey(fld, win, a.samples, a.seed, a.min_steps, a.k)
    print(f"{len(rows)} admissible points in the window "
          f"(x {win[0]:.2f}..{win[1]:.2f}, y {win[2]:.2f}..{win[3]:.2f})")

    # Site A first and unconditionally: it is the anchor, and it is the void
    site_a = described(fld, *centre, k=a.k)
    site_a.update(mode="site_a", reserved=True,
                  note=a.site_label or ("Site A -- medieval x industrial, on "
                                        "labour. Ground held in reserve for the "
                                        "intervention model."))
    chosen = [site_a]

    want = {}
    for part in a.mix.split(","):
        kk, v = part.split(":")
        want[kk.strip()] = int(v)
    order = [("clash", "clash_score"), ("mixed", "diversity"), ("empty", "steps_out")]
    for mode, key in order:
        n = want.get(mode, 0)
        if n <= 0:
            continue
        got = pick(rows, chosen, key, n, min_sep)
        for g in got:
            g["mode"] = mode
            g["reserved"] = False
        chosen += got
        print(f"  {mode:6s} {len(got):2d}/{n}")

    # footprint: the composition's own size rule, one level up
    cl = np.array([c["claim"] for c in chosen], float)
    gain = (cl / cl.mean()) ** SF.INFLUENCE
    base = 0.5 * (a.size_min_in + a.size_max_in)
    size_in = np.clip(base * gain / gain.mean(), a.size_min_in, a.size_max_in)

    for c, s, g in zip(chosen, size_in, gain):
        c["plan_x_in"] = round((c["x"] - win[0]) * IN_PER_UNIT, 4)
        c["plan_y_in"] = round((c["y"] - win[2]) * IN_PER_UNIT, 4)
        c["size_in"] = round(float(s), 4)
        c["size_ft"] = round(float(s) * FT_PER_IN / TILE_IN, 1)
        c["claim_gain"] = round(float(g), 4)
        c["claim"] = round(c["claim"], 3)

    manifest = {
        "protocol": a.protocol, "protocol_display": fld.display_name,
        "scale": f"1/{SCALE_DEN:.0f}\" = 1'",
        "tile_in": TILE_IN, "tile_units": TILE_UNITS,
        "in_per_unit": round(IN_PER_UNIT, 5),
        "tile_represents_ft": FT_PER_IN,
        "centre": {"x": centre[0], "y": centre[1],
                   "why": a.site_label or "Site A, the anchor"},
        "window": {"x0": win[0], "x1": win[1], "y0": win[2], "y1": win[3]},
        "k": a.k, "min_steps": a.min_steps, "seed": a.seed,
        "min_sep_in": a.min_sep_in,
        "size_band_in": [a.size_min_in, a.size_max_in],
        "size_rule": "sparse_fit.INFLUENCE applied to lift mass, clamped",
        "edge_mode": "unavailable: 95% of this window is inside the hull",
        "n": len(chosen),
        "points": chosen,
    }
    (out / "points.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(chosen)} points, {sum(1 for c in chosen if c['reserved'])} reserved")
    print(f"  footprints {size_in.min():.2f}-{size_in.max():.2f} in "
          f"({size_in.min()*FT_PER_IN/TILE_IN:.0f}-{size_in.max()*FT_PER_IN/TILE_IN:.0f} ft)")
    cov = float((np.pi * (size_in / 2) ** 2).sum() / (TILE_IN * TILE_IN))
    print(f"  nominal plan coverage {100*cov:.0f}% of the tile")
    print(f"wrote {out / 'points.json'}")

    try:
        plan_preview(fld, manifest, out / "points.png")
        print(f"wrote {out / 'points.png'}")
    except Exception as ex:
        print(f"(preview skipped: {type(ex).__name__}: {ex})")
    return 0


def plan_preview(fld, man, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    w = man["window"]
    fig, ax = plt.subplots(figsize=(9, 9))
    m = ((fld.xy[:, 0] > w["x0"] - .4) & (fld.xy[:, 0] < w["x1"] + .4) &
         (fld.xy[:, 1] > w["y0"] - .4) & (fld.xy[:, 1] < w["y1"] + .4))
    ax.scatter(fld.xy[m, 0], fld.xy[m, 1], s=3, c="0.7", lw=0, zorder=1)

    col = {"site_a": "crimson", "clash": "darkorange",
           "mixed": "steelblue", "empty": "seagreen"}
    for p in man["points"]:
        r = p["size_in"] / man["in_per_unit"] / 2
        # alpha on a Patch fades the edge too, so a reserved plot has to be
        # drawn with facecolor "none" or the void becomes invisible
        ax.add_patch(Circle(
            (p["x"], p["y"]), r,
            facecolor="none" if p["reserved"] else col[p["mode"]],
            alpha=None if p["reserved"] else 0.35,
            edgecolor=col[p["mode"]],
            lw=2.2 if p["reserved"] else 0.8,
            ls="--" if p["reserved"] else "-", zorder=5 if p["reserved"] else 3))
    ax.plot([w["x0"], w["x1"], w["x1"], w["x0"], w["x0"]],
            [w["y0"], w["y0"], w["y1"], w["y1"], w["y0"]], "k-", lw=1.2, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(w["x0"] - .4, w["x1"] + .4)
    ax.set_ylim(w["y0"] - .4, w["y1"] + .4)
    ax.set_title(f"{man['protocol_display']} -- 12 x 12 in tile at {man['scale']}\n"
                 f"{man['n']} volumes, {man['tile_represents_ft']:.0f} ft across, "
                 f"Site A dashed")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=k)
                       for k, c in col.items()], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
