r"""
tile_stability_check.py -- does clash ground actually sit low in stability?

The site-model tile proposes to drive topography from cross-protocol stability,
inverted: ground every model agrees about becomes plateau, contested ground
becomes fissure. That is only worth building if the four sites -- which are
clash conditions -- really do fall in the fissures. If clash score and local
stability are uncorrelated, the idea is decorative and should be dropped rather
than defended.

This tests it three ways, on the protocol the tile is built in (CLIP) and, as a
control, on a second protocol:

  1. correlation, over thousands of sampled interior points, between the clash
     score used by sparse_fit.clash_points and the local mean stability of the
     k works those points lift from
  2. the top-n clash points' local stability against the distribution over all
     sampled points -- a percentile, which is the legible number
  3. Site A itself (CLIP x 8.30, y 8.01), reported alone

Writes data/gap_space/site_tile/stability_check.json plus a plot.

    python tile_stability_check.py
    python tile_stability_check.py --protocols clip_vitl14_laion2b,dinov2_base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gap_space_core as G

SITE_A = (8.30, 8.01)
OUT = G.OUT / "site_tile"


def clash_field(fld, samples=9000, seed=0, min_steps=4.0, k=12, cap=3000):
    """Every sampled interior point, scored the way clash_points scores them,
    and carrying its neighbourhood's mean stability. No thinning and no top-n
    cut -- this is the whole population, which is what a correlation needs."""
    A = json.load(open(G.E / "affordance_vocab.json", encoding="utf-8"))["axes"]
    POS = {kk: np.array(v, float) for kk, v in A["positions"].items()}

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
    pts = pts[:cap]

    rows = []
    for q in pts:
        lf = G.lift(fld, q[0], q[1], k=k)
        acc = {}
        for i, w in zip(lf.idx, lf.w):
            acc[fld.period[i]] = acc.get(fld.period[i], 0.0) + float(w)
        top = sorted(((pp, w) for pp, w in acc.items() if pp in POS),
                     key=lambda t: -t[1])[:2]
        if len(top) < 2:
            continue
        (pa, wa), (pb, wb) = top
        dist = float(np.linalg.norm(POS[pa] - POS[pb]))
        # weighted, not plain: the lift already says who contributes how much,
        # and an unweighted mean would let one distant work outvote the site
        s = fld.stability[lf.idx]
        w = np.asarray(lf.w, float)
        ok = np.isfinite(s)
        stab = float((s[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        rows.append(dict(x=float(q[0]), y=float(q[1]),
                         score=dist * (wa * wb), axis_distance=dist,
                         wprod=float(wa * wb), stability=stab,
                         a=pa, b=pb))
    return rows


def point_stability(fld, x, y, k=12):
    lf = G.lift(fld, x, y, k=k)
    s = fld.stability[lf.idx]
    w = np.asarray(lf.w, float)
    ok = np.isfinite(s)
    return float((s[ok] * w[ok]).sum() / w[ok].sum())


def report(fld, rows, top_n=12):
    sc = np.array([r["score"] for r in rows])
    st = np.array([r["stability"] for r in rows])
    m = np.isfinite(sc) & np.isfinite(st)
    sc, st = sc[m], st[m]

    from scipy.stats import pearsonr, spearmanr, mannwhitneyu
    pr, pp = pearsonr(sc, st)
    sr, sp = spearmanr(sc, st)

    order = np.argsort(-sc)
    top = order[:top_n]
    rest = order[top_n:]
    # percentile of each top clash point's stability within the whole population
    pcts = [float((st < st[i]).mean() * 100) for i in top]
    U, up = mannwhitneyu(st[top], st[rest], alternative="less")

    return {
        "protocol": fld.protocol,
        "n_points": int(m.sum()),
        "stability_all": {"mean": float(st.mean()), "median": float(np.median(st)),
                          "sd": float(st.std())},
        "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
        "spearman_r": round(float(sr), 4), "spearman_p": float(sp),
        "top_n": top_n,
        "top_stability_mean": float(st[top].mean()),
        "rest_stability_mean": float(st[rest].mean()),
        "top_stability_percentiles": [round(p, 1) for p in pcts],
        "top_median_percentile": round(float(np.median(pcts)), 1),
        "mannwhitney_p_top_lower": float(up),
        "_sc": sc, "_st": st, "_top": top,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="clip_vitl14_laion2b,dinov2_base")
    ap.add_argument("--samples", type=int, default=9000)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("-k", type=int, default=12)
    ap.add_argument("--top-n", type=int, default=12)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out, plots = {}, []
    for pid in [p.strip() for p in a.protocols.split(",") if p.strip()]:
        fld = G.load_field(pid)
        rows = clash_field(fld, samples=a.samples, cap=a.cap, k=a.k)
        r = report(fld, rows, top_n=a.top_n)
        plots.append((fld, r))
        print(f"\n{pid}  ({r['n_points']} points)")
        print(f"  stability over all sampled ground : "
              f"mean {r['stability_all']['mean']:.4f}  "
              f"median {r['stability_all']['median']:.4f}")
        print(f"  clash score vs stability          : "
              f"pearson r={r['pearson_r']:+.3f} (p={r['pearson_p']:.2g})  "
              f"spearman r={r['spearman_r']:+.3f} (p={r['spearman_p']:.2g})")
        print(f"  top {a.top_n} clash points          : "
              f"mean stability {r['top_stability_mean']:.4f} vs "
              f"{r['rest_stability_mean']:.4f} elsewhere")
        print(f"  their stability percentiles       : "
              f"median {r['top_median_percentile']:.1f}  "
              f"{r['top_stability_percentiles']}")
        print(f"  one-sided Mann-Whitney (top lower): p={r['mannwhitney_p_top_lower']:.2g}")
        if pid.startswith("clip"):
            sa = point_stability(fld, *SITE_A, k=a.k)
            pct = float((r["_st"] < sa).mean() * 100)
            r["site_a"] = {"x": SITE_A[0], "y": SITE_A[1],
                           "stability": round(sa, 4), "percentile": round(pct, 1)}
            print(f"  Site A (x{SITE_A[0]}, y{SITE_A[1]})        : "
                  f"stability {sa:.4f} = {pct:.1f}th percentile of this ground")
        out[pid] = {kk: v for kk, v in r.items() if not kk.startswith("_")}

    (OUT / "stability_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 5))
        if len(plots) == 1:
            axes = [axes]
        for ax, (fld, r) in zip(axes, plots):
            ax.scatter(r["_sc"], r["_st"], s=4, alpha=0.25, lw=0)
            ax.scatter(r["_sc"][r["_top"]], r["_st"][r["_top"]], s=40,
                       facecolor="none", edgecolor="crimson", lw=1.4,
                       label=f"top {r['top_n']} clash")
            ax.set_xlabel("clash score  (axis distance x weight product)")
            ax.set_ylabel("local stability (weighted mean over the lift)")
            ax.set_title(f"{fld.display_name}\nspearman r={r['spearman_r']:+.3f}")
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "stability_check.png", dpi=140)
        print(f"\nwrote {OUT / 'stability_check.png'}")
    except Exception as ex:
        print(f"(plot skipped: {type(ex).__name__}: {ex})")

    print(f"wrote {OUT / 'stability_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
