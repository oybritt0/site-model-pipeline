r"""
tile_map.py -- where the sites and the tiles sit in every model space.

One panel per protocol. Each shows the whole corpus in that protocol's own
layout, coloured by category, with the four exhibition conditions marked at the
coordinate that protocol puts them at, and the tile windows drawn where they
were cut.

The panels cannot be aligned to each other and that is the point. Every layout is
an independent UMAP, so `(8.30, 8.01)` in CLIP has no counterpart anywhere else --
EXHIBITION_SITES.md 1. A condition is portable because periods and axes are
properties of the corpus; a coordinate is not. So a site appears in as many
places as there are protocols that find it, and the scatter of those positions
across the panels is a picture of how little the models agree about where
anything is.

    python tile_map.py
    python tile_map.py --protocols clip_vitl14_laion2b,flux_vae --cols 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import gap_space_core as G

OUT = G.OUT / "site_tile"

# the four from EXHIBITION_SITES.md 3, by condition rather than by coordinate
SITES = [
    ("A", {"03_medieval", "16_industrial"},        "medieval x industrial · labour"),
    ("B", {"03_medieval", "11_High Tech"},         "medieval x High Tech · labour"),
    ("C", {"06_neoclassical", "10_deconstructivism"},
                                                   "neoclassical x deconstructivism · legibility"),
    ("D", {"01_prehistoric", "11_High Tech"},      "prehistoric x High Tech · labour"),
]

CATCOL = {"architecture": "#4878a8", "painting_sculpture": "#c0703c",
          "drawing_print": "#4f9e6a"}

# the tiles actually built: protocol -> (centre x, centre y, label)
TILES = {
    "clip_vitl14_laion2b": (8.30, 8.01, "tile 1 · Site A"),
    "flux_vae": (8.67, -0.75, "tile 2 · mixed media"),
}
SPAN = 4.35


def conditions():
    s = json.load(open(G.OUT / "site_survey" / "survey.json", encoding="utf-8"))
    out = {}
    for tag, pair, label in SITES:
        c = next((c for c in s["conditions"] if set(c["pair"]) == pair), None)
        out[tag] = (c, label)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default=None)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "latent_map.png"))
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    reg = [p["id"] for p in G.list_protocols()]
    pids = ([p.strip() for p in a.protocols.split(",")] if a.protocols else reg)
    pids = [p for p in pids if p in reg]
    cond = conditions()

    cols = min(a.cols, len(pids))
    rows = (len(pids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 5.9 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, pid in zip(axes, pids):
        fld = G.load_field(pid)
        cat = np.array(fld.category, dtype=object)
        for c, col in CATCOL.items():
            m = cat == c
            ax.scatter(fld.xy[m, 0], fld.xy[m, 1], s=3.0, c=col, lw=0,
                       alpha=0.45, zorder=1)

        if pid in TILES:
            cx, cy, lab = TILES[pid]
            ax.add_patch(Rectangle((cx - SPAN / 2, cy - SPAN / 2), SPAN, SPAN,
                                   fill=False, ec="k", lw=1.8, zorder=4))
            ax.text(cx, cy + SPAN / 2 + 0.22, lab, ha="center", va="bottom",
                    fontsize=8.5, weight="bold", zorder=5)

        for tag, (c, _lab) in cond.items():
            if not c:
                continue
            b = c["best_per_protocol"].get(pid)
            if not b:
                continue
            ax.scatter([b["x"]], [b["y"]], s=150, marker="o", facecolor="none",
                       edgecolor="crimson", lw=1.8, zorder=6)
            ax.text(b["x"], b["y"], tag, ha="center", va="center", fontsize=8.5,
                    color="crimson", weight="bold", zorder=7)

        found = sum(1 for t, (c, _) in cond.items()
                    if c and pid in (c["best_per_protocol"] or {}))
        from scipy.spatial import cKDTree
        _, nn = cKDTree(fld.xy).query(fld.xy, k=13)
        same = float(np.mean([(cat[nn[i, 1:]] == cat[i]).mean()
                              for i in range(len(cat))]))
        chance = float(sum((np.mean(cat == c)) ** 2 for c in set(cat)))
        ax.set_title(f"{fld.display_name}\n{found} of 4 sites found  ·  "
                     f"medium segregation {same/chance:.2f}x", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.8")

    for ax in axes[len(pids):]:
        ax.axis("off")

    handles = [Line2D([], [], marker="o", ls="", color=v, ms=7,
                      label=k.replace("_", " / ")) for k, v in CATCOL.items()]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec="crimson",
                       ms=10, label="exhibition site, this protocol's coordinate"),
                Line2D([], [], marker="s", ls="", mfc="none", mec="k", ms=10,
                       label="12 x 12 in tile window")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("The four sites and the two tiles, in every model space — "
                 "no two layouts are aligned, so a condition moves and a "
                 "coordinate means nothing across panels", fontsize=11)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    fig.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}")

    print("\nwhere each condition lands, per protocol:")
    for tag, (c, lab) in cond.items():
        if not c:
            print(f"  {tag}: not in the survey")
            continue
        bs = c["best_per_protocol"] or {}
        pos = "  ".join(f"{p.split('_')[0]}({b['x']:.1f},{b['y']:.1f})"
                        for p, b in bs.items())
        print(f"  {tag}  {lab}\n       {len(bs)} protocols: {pos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
