r"""
gap_prompt.py — build a diffusion prompt from where a gap sits in the space.

The argument: style is a codification of the social and political values of the
period that produced it, so a coordinate in this space is a weighted position
among codifications. The prompt states what those codifications afford, and
asks for an architecture that articulates it, drawn as plan and section fused.

Two texts come out, because they have different consumers:

    prompt      compact, CLIP-safe. SDXL truncates at 77 tokens per encoder, so
                a paragraph would be silently cut. Ordered so the operative
                terms survive even if something is dropped.
    rationale   the full articulation, for the SVG sheet and the sidecar. This
                is where the reasoning is legible to a person.

Everything is weighted by the SAME lift that painted the image, so the prompt
and the assemblage describe one coordinate, not two.

Sources, both authored and editable, neither invented here:
    data/embeddings/verbal_vocab.json      45 terms, 4 facets, medium-neutral
    data/embeddings/verbal_profile.npy     (4449, 45) per-work affinity
    data/embeddings/affordance_vocab.json  style -> codification -> affordance

    python gap_prompt.py --protocol dinov2_base --at 9.71 3.35
    python gap_prompt.py --protocol dinov2_base --sort mixed -n 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import gap_space_core as G

VOCAB = G.E / "verbal_vocab.json"
PROFILE = G.E / "verbal_profile.npy"
AFFORD = G.E / "affordance_vocab.json"

_CACHE: dict = {}


def _load():
    if _CACHE:
        return _CACHE
    v = json.load(open(VOCAB, encoding="utf-8"))
    # Columns are the facet lists concatenated in dict order; build_verbal_layout.py
    # line 143 iterates facets.items() to build them. Verified 12+14+12+7 == 45.
    facets, cols, i = {}, [], 0
    for name, terms in v["facets"].items():
        facets[name] = (i, i + len(terms), terms)
        cols.extend(terms)
        i += len(terms)
    prof = np.load(PROFILE).astype(np.float32)
    if prof.shape[1] != len(cols):
        raise SystemExit(f"verbal_profile has {prof.shape[1]} columns, vocab has "
                         f"{len(cols)} terms; they must match")
    idx = json.load(open(G.E / "index.json", encoding="utf-8"))["ids"] \
        if (G.E / "index.json").exists() else None
    aff = json.load(open(AFFORD, encoding="utf-8"))
    _CACHE.update(facets=facets, terms=cols, prof=prof, ids=idx, aff=aff)
    return _CACHE


def gap_terms(fld: G.Field, lf: G.Lift, per_facet=1) -> dict:
    """Top authored term(s) per facet at this coordinate, from the lift weights."""
    c = _load()
    row = {i: r for r, i in enumerate(c["ids"] or fld.ids)}
    rows, ws = [], []
    for i, w in zip(lf.idx, lf.w):
        r = row.get(fld.ids[i])
        if r is not None and r < len(c["prof"]):
            rows.append(r)
            ws.append(w)
    if not rows:
        return {}
    ws = np.array(ws, dtype=np.float32)
    ws = ws / ws.sum()
    v = (c["prof"][rows] * ws[:, None]).sum(0)
    out = {}
    for name, (a, b, terms) in c["facets"].items():
        seg = v[a:b]
        order = np.argsort(-seg)[:per_facet]
        out[name] = [terms[int(j)] for j in order]
    return out


def period_mix(fld: G.Field, lf: G.Lift, top=3) -> list:
    acc: dict = {}
    for i, w in zip(lf.idx, lf.w):
        acc[fld.period[i]] = acc.get(fld.period[i], 0.0) + float(w)
    return sorted(acc.items(), key=lambda kv: -kv[1])[:top]


def _fusion_word(stability: float, rng) -> tuple:
    c = _load()["aff"]["fusion"]
    if not np.isfinite(stability):
        band = "mixed"
    elif stability < 0.12:
        band = "contested"          # the protocols disagree here
    elif stability > 0.20:
        band = "settled"
    else:
        band = "mixed"
    return band, str(rng.choice(c[band]))


def build(fld: G.Field, px: float, py: float, k=10, power=2.0, seed=0,
          category=None) -> dict:
    """Prompt, negative, rationale and the parts they were built from."""
    c = _load()
    rng = np.random.default_rng(seed)
    mask = None
    if category:
        mask = np.array([x == category for x in fld.category], dtype=bool)
    lf = G.lift(fld, px, py, k=k, power=power, mask=mask)

    terms = gap_terms(fld, lf)
    mix = period_mix(fld, lf)
    with np.errstate(invalid="ignore"):
        stab = float(np.nanmean(fld.stability[lf.idx]))
    band, fuse = _fusion_word(stab, rng)

    P = c["aff"]["periods"]
    known = [(p, w) for p, w in mix if p in P]
    if not known:
        raise ValueError("no period at this point has an affordance entry")
    a = known[0]
    b = known[1] if len(known) > 1 else None

    def nice(p):
        return p.split("_", 1)[-1].replace("_", " ")

    # ---- the prompt describes a SPACE, not a set of style attributes.
    #
    # The first version listed what each codification looks like -- "strict axis,
    # portico, rational grid, stripped column, austere cornice" -- which is a bag
    # of tags. It never says what the architecture IS, so the model gets a style
    # label rather than a room to draw.
    #
    # This composes one sentence instead: the room the dominant codification
    # makes, how a body moves through it, what the second codification does to
    # it spatially, and how it is lit. The measured terms then qualify that room
    # rather than standing on their own. Materially the same evidence, but
    # manifested as architecture.
    ph = [f"{c['aff']['drawing']['base']} of {P[a[0]]['space']}"]
    ph.append(P[a[0]]["move"])
    if b:
        ph.append(f"{fuse} {P[b[0]]['space']}")
    ph.append(P[a[0]]["light"])
    if terms.get("material"):
        ph.append(terms["material"][0])
    # gestalt reads as an adjective on the whole, surface as a finish; both
    # qualify the room. organization is dropped: the space phrase already
    # carries the plan logic, and it was the term most often contradicting it.
    for facet in ("gestalt", "surface"):
        if terms.get(facet):
            ph.append(terms[facet][0])
    prompt = ", ".join(ph)

    negative = ("photograph, photorealistic, 3d render, perspective view, "
                "figures, people, sky, landscape, text, watermark, signature, "
                "frame, border, colour photograph, low quality")

    # ---- rationale: the argument, for the sheet and the sidecar
    lines = [
        f"At this coordinate the space is weighted "
        + ", ".join(f"{nice(p)} {w:.0%}" for p, w in mix) + ".",
        f"{nice(a[0]).capitalize()} codifies {P[a[0]]['codifies']}. It affords "
        f"{P[a[0]]['affords']}, and withholds {P[a[0]]['withholds']}.",
    ]
    if b:
        lines.append(
            f"{nice(b[0]).capitalize()} codifies {P[b[0]]['codifies']}, affording "
            f"{P[b[0]]['affords']}. Here the first is {fuse} the second.")
    lines.append(
        "Drawn as plan and section fused: "
        f"{P[a[0]]['plan']}; {P[a[0]]['section']}"
        + (f"; {P[b[0]]['section']}" if b else "") + ".")
    lines.append(
        "Measured at this point as "
        + ", ".join(f"{f} {t[0]}" for f, t in terms.items() if t) + ".")
    lines.append(
        f"The point sits {lf.steps_out:.0f} steps outside the nearest existing "
        f"work; cross-protocol agreement here is "
        + ("unavailable" if not np.isfinite(stab) else f"{stab:.3f} ({band})") + ".")

    return {
        "prompt": prompt, "negative": negative,
        "rationale": " ".join(lines),
        "periods": [{"period": p, "weight": round(w, 4)} for p, w in mix],
        "terms": terms, "fusion_band": band, "fusion_word": fuse,
        "stability": None if not np.isfinite(stab) else stab,
        "steps_out": lf.steps_out, "category": category or "all",
        "x": float(px), "y": float(py), "protocol": fld.protocol,
        "approx_tokens": len(prompt.split()) + prompt.count(","),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="dinov2_base")
    ap.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--sort", default="mixed")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--category", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    fld = G.load_field(args.protocol)
    pts = ([tuple(args.at)] if args.at else
           [(c["x"], c["y"]) for c in
            G.gap_candidates(fld, n=args.n, samples=8000, min_steps=8.0,
                             sort=args.sort)])
    for x, y in pts:
        r = build(fld, x, y, k=args.k, category=args.category)
        if args.json:
            print(json.dumps(r, indent=2))
            continue
        print("=" * 78)
        print(f"{args.protocol}  ({x:.2f}, {y:.2f})  "
              f"{r['steps_out']:.0f} steps out  [{r['fusion_band']}]")
        print(f"\nPROMPT ({r['approx_tokens']} approx tokens)\n  {r['prompt']}")
        print(f"\nRATIONALE\n  {r['rationale']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
