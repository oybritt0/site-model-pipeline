r"""
tile_protocol_mix.py -- which model space mixes the categories, and where.

The first tile was cut from CLIP at x 8.3 / y 8.0 and contained no drawings at
all: 724 works inside the window, 92.5% architecture, 0% drawing_print. That was
not a filter -- every work in the corpus is eligible -- it is that CLIP sorts by
medium before anything else, and the tile landed in architecture territory.

So the question is whether another protocol mixes them, and if so where to cut.
Two measurements, no GPU and no credits:

  purity    for every work, how often its 12 nearest neighbours share its
            category, against the chance rate implied by the corpus mix. A
            protocol at chance has no medium preference at all.

  windows   a tile is a 4.35-unit square, not a point, so candidate windows are
            slid across each layout and scored on the entropy of the category
            mix inside. A window can only build what it contains.

Reports the best-mixing window per protocol, and checks it still carries the
things a site needs -- enough works to lift from, and a real clash condition.

    python tile_protocol_mix.py
    python tile_protocol_mix.py --span 4.35 --min-works 300
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import gap_space_core as G

OUT = G.OUT / "site_tile"
TILE_UNITS = 4.35            # a 12 in tile at 2.7586 in per layout unit


def purity(fld, k=12):
    """How medium-segregated this layout is, against its own chance rate."""
    from scipy.spatial import cKDTree
    cat = np.array(fld.category, dtype=object)
    _, nn = cKDTree(fld.xy).query(fld.xy, k=k + 1)
    same = float(np.mean([(cat[nn[i, 1:]] == cat[i]).mean() for i in range(len(cat))]))
    chance = float(sum((np.mean(cat == c)) ** 2 for c in set(cat)))
    return same, chance


def entropy(counts):
    n = sum(counts.values())
    if n == 0:
        return 0.0
    p = np.array([v / n for v in counts.values() if v])
    return float(-(p * np.log(p)).sum() / math.log(3))      # 3 categories


def best_windows(fld, span, step, min_works, top=3):
    """Slide a tile-sized window over the layout and score the category mix."""
    cat = np.array(fld.category, dtype=object)
    lo, hi = fld.xy.min(0), fld.xy.max(0)
    xs = np.arange(lo[0], hi[0] - span, step)
    ys = np.arange(lo[1], hi[1] - span, step)
    rows = []
    for x0 in xs:
        for y0 in ys:
            m = ((fld.xy[:, 0] >= x0) & (fld.xy[:, 0] < x0 + span) &
                 (fld.xy[:, 1] >= y0) & (fld.xy[:, 1] < y0 + span))
            n = int(m.sum())
            if n < min_works:
                continue
            c = {k: int((cat[m] == k).sum()) for k in set(fld.category)}
            c = {k: v for k, v in c.items() if v}
            rows.append({"x0": float(x0), "y0": float(y0), "n": n,
                         "entropy": round(entropy(c), 4),
                         "n_categories": len(c),
                         "mix": {k: round(100 * v / n, 1) for k, v in
                                 sorted(c.items(), key=lambda t: -t[1])}})
    rows.sort(key=lambda r: (-r["n_categories"], -r["entropy"]))
    # thin so the top few are not the same window nudged one step
    out = []
    for r in rows:
        if any(abs(r["x0"] - o["x0"]) < span * 0.5 and
               abs(r["y0"] - o["y0"]) < span * 0.5 for o in out):
            continue
        out.append(r)
        if len(out) >= top:
            break
    return out, len(rows)


def site_check(fld, win, span, k=12, samples=2500, seed=0):
    """Does the window carry a real collision, or only a nice mix?"""
    A = json.load(open(G.E / "affordance_vocab.json", encoding="utf-8"))["axes"]
    POS = {kk: np.array(v, float) for kk, v in A["positions"].items()}
    names = A["order"]
    rng = np.random.default_rng(seed)
    pts = rng.uniform([win["x0"], win["y0"]],
                      [win["x0"] + span, win["y0"] + span], size=(samples, 2))
    try:
        from scipy.spatial import Delaunay
        pts = pts[Delaunay(fld.xy).find_simplex(pts) >= 0]
    except Exception:
        pass
    u = G.step_unit(fld)
    d, _ = fld.tree().query(pts, k=1)
    pts = pts[d >= 3.0 * u]
    best = None
    for q in pts[:900]:
        lf = G.lift(fld, q[0], q[1], k=k)
        acc = {}
        for i, w in zip(lf.idx, lf.w):
            acc[fld.period[i]] = acc.get(fld.period[i], 0.0) + float(w)
        top = sorted(((p, w) for p, w in acc.items() if p in POS),
                     key=lambda t: -t[1])[:2]
        if len(top) < 2:
            continue
        (pa, wa), (pb, wb) = top
        dist = float(np.linalg.norm(POS[pa] - POS[pb]))
        sc = dist * wa * wb
        cats = {}
        for i in lf.idx:
            cats[fld.category[i]] = cats.get(fld.category[i], 0) + 1
        if best is None or sc > best["score"]:
            best = {"x": float(q[0]), "y": float(q[1]), "score": round(sc, 4),
                    "a": pa, "b": pb, "axis": names[int(np.argmax(np.abs(POS[pa]-POS[pb])))],
                    "axis_distance": round(dist, 3),
                    "lift_categories": cats}
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=float, default=TILE_UNITS)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--min-works", type=int, default=250)
    ap.add_argument("--out", default=str(OUT / "protocol_mix.json"))
    a = ap.parse_args()

    res = {}
    for p in G.list_protocols():
        pid = p["id"]
        try:
            fld = G.load_field(pid)
        except Exception as ex:
            print(f"{pid}: skipped ({type(ex).__name__})")
            continue
        same, chance = purity(fld, 12)
        wins, n = best_windows(fld, a.span, a.step, a.min_works)
        print(f"\n{pid}  ({fld.display_name})")
        print(f"  medium segregation: neighbours share category {100*same:.1f}% "
              f"vs {100*chance:.1f}% chance  ->  {same/chance:.2f}x")
        if not wins:
            print(f"  no window of {a.span} units holds {a.min_works}+ works")
            res[pid] = {"purity": same, "chance": chance, "windows": []}
            continue
        for w in wins[:3]:
            mix = ", ".join(f"{k.replace('_','/')} {v}%" for k, v in w["mix"].items())
            print(f"  window x {w['x0']:.1f} y {w['y0']:.1f}  n={w['n']:4d}  "
                  f"cats={w['n_categories']}  H={w['entropy']:.3f}   {mix}")
        res[pid] = {"purity": round(same, 4), "chance": round(chance, 4),
                    "ratio": round(same / chance, 3),
                    "windows_scored": n, "windows": wins}

    order = sorted((r for r in res.items() if r[1]["windows"]),
                   key=lambda t: -t[1]["windows"][0]["entropy"])
    print("\n" + "=" * 74)
    print("best mixed window per protocol, ranked:")
    for pid, r in order:
        w = r["windows"][0]
        mix = ", ".join(f"{k.split('_')[0]} {v}%" for k, v in w["mix"].items())
        print(f"  {pid:24s} H={w['entropy']:.3f}  cats={w['n_categories']}  "
              f"segregation {r['ratio']:.2f}x   {mix}")

    if order:
        pid, r = order[0]
        fld = G.load_field(pid)
        w = r["windows"][0]
        s = site_check(fld, w, a.span)
        r["windows"][0]["site_check"] = s
        print(f"\nbest window is {pid} at x {w['x0']:.2f} y {w['y0']:.2f}")
        if s:
            print(f"  strongest collision inside it: {s['a']} x {s['b']} on "
                  f"{s['axis']}, distance {s['axis_distance']}, score {s['score']}")
            print(f"  that point's own lift draws from: {s['lift_categories']}")
        else:
            print("  no clash condition found inside it")

    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
