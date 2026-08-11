# The 12 × 12 in site model tile — what the test answered

Built 2026-08-10 on branch `gap-space`, against `HANDOFF_site_model_tile.md`.
Everything below is measured. Where the plan and the evidence disagreed, the
evidence is recorded and the plan was changed, not defended.

Data under `data/gap_space/site_tile/`. Scripts are `tile_*.py` in the project
root.

---

## The short version

| | question | answer |
|---|---|---|
| **1** | do the meshes hold the fragment silhouettes at printed size? | **Partly.** Outline yes (IoU 0.80), interior no. And raising marching-cubes resolution does not help — the limit is the model, not the grid. |
| **2** | do they boolean-union into a continuous mass? | **Yes.** 94 volumes + ground → one watertight solid in 100 s, no collapse. But every mesh arrives with 9–70 disconnected specks that have to be dropped. |
| **3** | what does one tile cost? | **536 g and 71.5 h**, in 4 pieces, from a real slice. 11.1 sq ft is then **5.9 kg and 33 days** on one machine — and **89% of it is the ground, not the city**. |
| **4** | does it look like the hallucinatory cityscape? | **No — it reads as an eroded settlement, a ruin field with a plaza cleared in it.** Not blobs: the masses are properly articulated. But geological rather than built. |
| **5** | do clash points sit in low-stability ground? | **No, not in CLIP.** The correlation is positive. Stability-driven topography dropped. |
| **6** | the real `run.py` flags | `--mc-resolution`, `--no-remove-bg`, `--foreground-ratio`, `--chunk-size`, `--device`, `--model-save-format`, `--render` — none of them in the README. |

---

## 0. Two things that were wrong in the brief, found early

**A 12 in tile does not fit the bed.** 12 in is 304.8 mm and
`EXHIBITION_SITES.md` §5b assumes a 256 mm bed. The tile has to be quartered:
4 pieces at 152.4 mm. `tile_print.py` does the split on the tile grid, so the
seams coincide with the tiling of the full 40 × 40 in model rather than being an
artefact of this test. On a 350 mm machine it fits whole.

**`1/64" = 1'` means one model inch is 64 feet, not 768.** The first build log
reported Site A's plot as 2,381 ft because of a stray × 12. Fixed, and pinned to
a named constant in `tile_model.py` so it cannot recur. The brief's own figures
were right; the code was not.

---

## 5. Does clash ground sit low in stability? No.

Answered first because it was cheap and because it governed the topography.
`tile_stability_check.py`, 2,518 admissible interior points in CLIP scored the
way `sparse_fit.clash_points` scores them, each carrying the lift-weighted mean
stability of the works it recruits.

| protocol | clash score vs stability | top 12 clash points | Mann-Whitney (top lower) |
|---|---|---|---|
| **CLIP** | spearman **+0.212** (p 5e-27) | 0.1534 vs 0.1580, median **57th** percentile | p = 0.76 |
| DINOv2 | spearman −0.096 (p 3e-07) | 0.1381 vs 0.1550, median 28th percentile | p = 0.019 |

**Site A sits at the 63rd percentile of stability — above the median.** In the
protocol this tile is built in, contested ground is very slightly *more* agreed
about, not less. The effect even changes sign between protocols, which is its
own small finding about instrument disagreement.

So the proposal — plateau where models agree, fissure where they do not, the four
sites in the fissures — is not supported and was dropped. `--relief stability`
still runs, and its map is in the record, but it is not the tile.

**What replaced it.** Ground height falls with distance to the nearest work:
occupied territory is plateau, gaps are basins. A site in this project is
*defined* as a gap, so the sites now sit in low ground by construction rather
than by hope, and the relief is a measurement of the layout rather than an
assertion about it.

Relief is **0.35 in (22 ft)**, not the 0.75–2 in budgeted. 2 in is 128 ft against
masses only 43–97 ft tall; the landscape would have swallowed the city.

---

## 1. Silhouettes, and the marching-cubes resolution

TripoSR is fed `sparse_fit`'s own matte, not rembg's guess. Each mesh's
silhouette from the input camera (TripoSR's own convention: x back, y right,
z up, input at azimuth 0, so the picture plane is `(y, −z)`) is compared to the
input alpha, both normalised to their own bounding box so this is shape
agreement and not a scale artefact.

`thin_recall` is the number that answers the question actually asked: of the
input pixels lying in parts thinner than one 0.4 mm nozzle at this object's real
printed size, how many survive into the mesh.

**Sweep, 3 points, one forward pass each** (`mc_sweep/summary.json`):

| mc | IoU | thin_recall | faces | watertight | extract |
|---|---|---|---|---|---|
| 128 | 0.836 | **0.768** | 32,000 | 3/3 | 1.1 s |
| 192 | 0.838 | 0.762 | 81,000 | 3/3 | 2.0 s |
| 256 | 0.834 | 0.741 | 143,000 | 3/3 | 5.2 s |
| 384 | 0.835 | 0.737 | 333,000 | 2/3 | 19 s |
| 512 | 0.835 | 0.732 | 603,000 | **0/3** | 54 s |

**Resolution buys nothing.** IoU is flat to the third decimal across a 4× range,
`thin_recall` gets slightly *worse*, and past 256 the meshes stop closing. Peak
allocation hits 9.2 GB at 512 on a 6.1 GB card — it spills to shared memory,
which is where the 54 s comes from.

At 128 the grid spacing on a 1.25 in object is 0.0098 in = **0.25 mm, already
finer than the nozzle**. There is no printed detail above 128 to recover.

The limit is the model. Looking at `silhouette_mc*.png`: the mesh gets the outer
contour and the major fragment boundaries and **fills every interstice** — the
slots and notches between fragments close up. So at 1.0–1.5 in printed, what
survives is the massing, not the tracery. That is the honest answer to question 1
and it is not a resolution problem.

**Full runs.** 32 points at mc 128 and 192 (`meshes_n32/`): IoU 0.795 both,
thin_recall 0.777 / 0.775, watertight 29/32 and 31/32, 6.4 s per object for both
extractions. Then all 96 points at mc 128, twice over (`meshes/`, `meshes_swarm/`):

| compositions | IoU | thin_recall | thin_frac | faces | watertight | per object |
|---|---|---|---|---|---|---|
| `drift` | 0.805 | 0.805 | 0.24 | 31,900 | 95/96 | **1.3 s** |
| `swarm` | 0.784 | 0.822 | 0.28 | 40,600 | 95/96 | **1.2 s** |

96 meshes in 2.0 minutes, peak VRAM 1.96 GB. The 3D step is not the bottleneck
anywhere — composing the assemblages takes 40 s each and the meshes take 1.3 s.

### The go/no-go on paid Tripo

The handoff set two triggers and asked which one fired, since they are different
problems. **Neither fired outright, and the one that half-fired is the silhouette
trigger, not the topology one.**

- *Topology that will not boolean* — **did not fire.** The meshes are closed and
  union cleanly (§2). The only topological cost is debris, and dropping unattached
  bodies handles it.
- *Cannot hold the silhouettes at 1.5 in printed* — **half fired.** The outline
  holds; the interstices do not. That is a property of the density field, proven
  by the resolution sweep being flat, so no local setting recovers it.

So if the paid service is tried, it should be tried **on one image, judged on
whether the gaps between fragments survive**, and nothing else. That is a
single-call test and it is the whole question.

---

## 2. The union holds

`manifold3d` through trimesh, balanced pairwise rather than sequential
accumulation — pairing keeps the operands small for longer, where accumulating
into one growing mesh makes every step pay for the whole accumulated mass.
94 volumes → 1 in seven rounds, 6–9 s per round.

```
mass:    2,953,030 faces, watertight True, 1,367 bodies
ground:    148,992 faces, watertight True
solid:   2,398,810 faces, watertight True, after the void and the tile clip
tile:    2,196,492 faces, watertight True, ONE body, after dropping debris
```

No collapse, no degenerate output, **100 seconds** for the whole tile. The
documented `ShapeUpgrade_UnifySameDomain` and quadric-decimation traps were
simply not entered: nothing is decimated anywhere in this pipeline, and no
OpenCASCADE step is involved. `pymeshlab` (`generate_boolean_union`, libigl) is
wired as a fallback and was never needed.

**The one real problem is debris.** Every TripoSR mesh is one mass plus 9–70
disconnected specks — 1,087 bodies survived into the unioned solid. They are
unprintable islands, so anything left unattached to the ground is dropped after
the union rather than before it, since a speck that lands on a neighbour is
legitimately part of the field and `sparse_fit`'s stranded outliers are meant to
survive. **1,318 bodies dropped, 0.13% of volume.**

One mesh in 32 stayed open after hole-filling and was excluded rather than fed to
`manifold3d`, which rejects non-volumes outright and can lose a part silently.

---

## 3. What one tile costs

Really sliced, not estimated: Bambu Studio CLI, P1S 0.4 nozzle, 0.20 mm standard,
Bambu PLA Basic, default 15% infill. Filament weight from length at 1.24 g/cm³
because the CLI leaves `filament_density` at 0 when it resolves a system profile.

| piece | size | solid | filament | time |
|---|---|---|---|---|
| 00 | 152 × 152 × 34 mm | 269 cm³ | 118.5 g | 15 h 13 m |
| 10 | 152 × 152 × 44 mm | 332 cm³ | 142.9 g | 18 h 46 m |
| 01 | 152 × 152 × 54 mm | 336 cm³ | 154.8 g | 21 h 24 m |
| 11 | 152 × 152 × 46 mm | 249 cm³ | 119.7 g | 16 h 09 m |
| **tile** | 12 × 12 in | 1,186 cm³ | **536 g** | **71.5 h** |

**11.1 sq ft = 11.1 tiles → 5.9 kg of filament and 794 hours: 33.1 days of
continuous printing on one machine, in 44 printed parts.**

That is the number the brief asked for and it is large enough to be a decision.

**Almost none of it is the city.** The ground is 64.6 in³ of the tile's 72.4 in³
— **89%** — and the ninety-four masses add only about 8 in³ net, because they are
lacy, partly buried, and the void removes some. The first build made the point
even more starkly: at 0.75 in of relief the ground was 105.5 in³ of 107.5 in³,
**98%**, and printed *more* filament from 32 volumes than the final tile does
from 94. The architecture is a rounding error on the landscape.

Which means the levers are all in the ground, and none of them touch the design:

| | ground solid | one tile |
|---|---|---|
| 0.20 in base, 0.75 in relief *(first build, 32 volumes)* | 105.5 in³ | 641 g, 79.2 h |
| 0.20 in base, 0.35 in relief *(the tile, 94 volumes)* | 64.6 in³ | **536 g, 71.5 h** |
| 0.12 in base, 0.35 in relief | 53.1 in³ | not sliced |

- **0.6 mm nozzle at 0.3 mm layers** roughly halves the time again, and the
  geometry is already below nozzle resolution at mc 128, so nothing is lost.
- **the ground does not have to be printed at all.** If it is 89% of the cost and
  it is a smooth height field, it is a candidate for CNC foam, casting, or a
  ribbed shell, with only the masses printed. That is a bigger saving than any
  slicer setting and it is worth deciding before 11 sq ft is committed.

The physical print is the one step of this handoff that could not be executed
here. `print/piece_*.stl` are in millimetres, on the bed, ready to slice.

---

## 6. The real `run.py` flags

The README documents `--bake-texture` and `--texture-resolution` only. The rest,
from `run.py --help`:

| flag | default | note |
|---|---|---|
| `--mc-resolution` | 256 | marching cubes grid. **Use 128–192.** |
| `--no-remove-bg` | off | skips rembg; expects RGB on **0.5 grey** with the foreground already sized |
| `--foreground-ratio` | 0.85 | only used when rembg *is* running |
| `--chunk-size` | 8192 | lower trades VRAM for time |
| `--device` | cuda:0 | |
| `--model-save-format` | obj | or glb |
| `--render` | off | NeRF turntable, not needed here |

`--no-remove-bg` is the one that matters and its help text is misleading: it says
"gray background", and the model really was trained on a **0.5** grey composite
at `--foreground-ratio` framing. `tile_triposr.py` reproduces that exactly
through TripoSR's own `resize_foreground`, so the matte goes in without rembg
ever seeing an assemblage.

---

## Install notes

- **`torchmcubes` cannot be built on this machine.** No Visual Studio, no CUDA
  toolkit, so CMake fails at `nmake` before it gets anywhere near CUDA.
  Substituted `skimage.measure.marching_cubes` via
  `tsr/models/_mc_skimage.py`, imported by a `try/except ImportError` in
  `tsr/models/isosurface.py`. 0.49 s at 256³, which is nothing.

  Two conventions had to be matched and both were verified against an offset
  anisotropic ellipsoid — per-axis bounds exact, volume within 0.03% of
  analytic, watertight:

  - torchmcubes returns vertices reversed and the caller reverses them back, so
    the shim pre-reverses and the two cancel.
  - **interior sense is the trap.** `system.extract_mesh` passes
    `-(density - 25)` and `isosurface.forward` negates *again*, so the solid is
    the side **above** the level. skimage's default `gradient_direction="descent"`
    assumes the opposite and returns inward-facing triangles that look watertight
    and boolean into nonsense. Hence `"ascent"`. The first smoke test produced a
    chair with volume −0.0645 before this was caught.

- **Installed into its own venv, and the project's environment is untouched.**
  `../tools/triposr/.venv` created with `--system-site-packages`, so it shares
  torch by reference and no 2.5 GB download was needed. pip correctly refused to
  uninstall the project's trimesh 4.12.2 and einops 0.8.2 ("outside
  environment") and shadowed them locally instead. Verified after:
  **torch 2.5.1+cu121, diffusers 0.32.2, transformers 4.44.2, peft 0.14.0.**
  `rembg` needs `onnxruntime`, which it does not declare.

- **`manifold3d` was added to the project environment** with `--no-deps`, for the
  boolean. It has no shared dependencies with the diffusion stack and the pins
  above were re-verified after installing it.

- **VRAM was never the constraint at working resolutions.** Photoshop, Rhino and
  Illustrator were open throughout and held 389 MB between them; the full 32-mesh
  run peaked at 2.1 GB of a 6.1 GB card. The handoff's instruction to close them
  is only needed above mc 256.

---

## The other route, on the same points

`batch_assemblage_geometry.py` gained `--coords`, which takes a `points.json` and
builds those exact coordinates instead of searching for its own, advancing the
seed per point so the placement really is the same one `sparse_fit` composed.
Six points, both routes, `tile_compare.py`:

| | TripoSR | assemblage_geometry |
|---|---|---|
| faces | 79,800 | 72,600 |
| bodies | 51 | 20 (= 10 fragments × 2 readings) |
| watertight | 6/6 | 6/6 |
| depth / width | **0.62** | **0.11** |
| solid / bbox | 0.14 | 0.59 |
| largest body | 99.3% of volume | 16.9% |
| time per point | **6.4 s** | 46 s |

They are not competing versions of one thing. TripoSR **infers** a volume: soft,
eroded, striated, single-bodied, and it does not know what a fragment is — the
interstices close. `assemblage_geometry` **derives** one: every fragment survives
as its own closed solid carrying its period and four terms, with real arcs, but
its depth is authored (`--depth 0.04`), not found, so it stands as a set of thin
fins.

See `compare/compare.png` — top row inferred, bottom row derived, same six
coordinates, both stood upright so the only difference on show is depth.

For a *site model*, TripoSR's masses are the more building-like object by a long
way. For a *drawing*, the geometry route keeps everything that matters and
TripoSR loses it. That is a cleaner division of labour than "which one wins".

---

## 4. Does it look like the city?

**Not a pile of blobs, and not a cityscape either. It reads as an eroded
settlement — an archaeological plateau with a plaza cleared in it.**

That is the honest answer and it took two builds to get to, because the first one
was wrong for a reason worth recording.

### The first build was too sparse, and the arithmetic in the brief overstated it

32 volumes at 1.0–1.5 in, exactly as specified, gave 26% nominal footprint
coverage — and **5.1% actual built coverage**, measured off the solid rather than
off the circles. The masses read as crumbs scattered on a landscape.

The cause is measurable and it is not the count. Each TripoSR volume fills only
**11% of its own bounding box** (range 4–24%). The compositions are sparse
scatters with large gaps, and the model reproduces that sparsity as lacy,
branching volume. So a footprint circle overstates built ground about fivefold,
and "30 of them in 768 ft square is dense-urban" was optimistic by roughly that
factor: 30 buildings on 13.5 acres is suburban.

The brief said to raise the count before changing the scale, so the count went to
**96** at 0.9–1.6 in.

| | volumes | nominal | **built** |
|---|---|---|---|
| first build | 32 | 26% | **5.1%** |
| `drift`, 96 | 94 | 71% | **17.4%** |
| `swarm`, 96 | 94 | 71% | **20.9%** |

17–21% is a town, not a metropolis — real dense-urban plan coverage is 30–50% —
but it is a field, and the masses now union into continuous ridges and blocks
instead of standing apart.

### Composition protocol barely moves density

`swarm` (24 forms at 0.90) against `drift` (10 at 1.30), both run on all 96
points: matte coverage **30.3% vs 29.1%**. Almost nothing. Because the
composition is shrunk rigidly to clear the frame, more forms means *smaller*
forms, not more ground covered. `swarm` does buy 3.5 points of built coverage and
a visibly finer grain — more, smaller incidents — and it is the better-looking
tile, but the lever for density is the number of coordinates, not the protocol.

### What it actually looks like

At the scale a viewer meets it — the 5 in cut block in
`views_drift_vs_swarm.png` — the masses are genuinely architectural and genuinely
strange. The rolled sections and colonnades in the compositions come through as
fluting, arcading, stepped buttresses and towers. Nothing reads as a blob. The
skyline in the cut elevation is a real skyline.

But the character is **geological**. TripoSR erodes; its surfaces are weathered
and undercut rather than built, and because it closes every interstice the
fragments stop being fragments and become one continuous rock. What you get is a
ruined citadel or an excavated site, not a proposal.

For this project that may be the right object rather than the wrong one — the
thesis is unbuilt architecture for ground that exists only inside a model, and a
field of ruins of things that were never built is a legible form of that. But it
is not what the brief asked for, and the difference should be a decision rather
than a discovery.

**Site A is the strongest thing in the tile.** The 3.1 in plot reads with total
clarity in plan: a clean circle of levelled ground with the field crowding its
rim. Nine volumes fell inside and were cleared; **eleven are cut by the site
boundary**, and those cut faces are exactly the adjacent condition an
intervention would attach to. Both options the brief offered for the void — a
clean reserved plot, or an adaptive-reuse edge — turn out to be present at once,
without having to choose.

### If it should read more urban

In order of leverage, none of them touching the method:

1. **More coordinates.** Coverage is close to linear in count so far (32 → 96
   took 5.1% → 17.4%). Roughly 200 per tile would reach dense-urban.
2. **`swarm` over `drift`**, for grain rather than for density.
3. **Denser source compositions.** The lever is `sparse_fit`'s selection, not the
   protocol — and the brief is explicit that the compositions are not to be
   "improved", so this one is closed unless that changes.

`--stand flat` was also built and tested. It lays the picture plane down so the
composition becomes the plan, and it reads as **excavated foundations** — lower,
flatter, more archaeological still. It is in the record as
`tile_n32/tile_density_flat_mc192.stl`; upright is the better object.

---

## What is on disk

```
data/gap_space/site_tile/
  points.json points.png            96 coordinates, plan positions, footprints
  stability_check.json .png         question 5
  assemblages/                      96 x drift + swarm, each with its matte
  meshes/  meshes_swarm/            96 x TripoSR at mc 128, measured
  mc_sweep/                         3 points x 5 resolutions, question 1
  geometry/                         6 points through assemblage_geometry
  compare/compare.png .json         the two routes, same six points
  tile/       tile_density_upright_mc128.stl / .ply / .json / _views.png
  tile_swarm/ the same, from swarm compositions
  tile_n32/   the first build, kept — and the flat variant
  print/      piece_[00,01,10,11].stl in mm, print_estimate.json
  views_drift_vs_swarm.png          the sheet to look at
```

Scripts, all new except the last two:

| | |
|---|---|
| `tile_stability_check.py` | question 5 |
| `tile_points.py` | coordinates → plan positions and footprints |
| `tile_assemblages.py` | compositions + mattes at those coordinates |
| `tile_triposr.py` | TripoSR batch, **runs in the TripoSR venv**, measures every mesh |
| `tile_model.py` | ground, placement, union, void, clip → STL/PLY |
| `tile_compare.py` | the two 3D routes on the same points |
| `tile_print.py` | bed split + real Bambu Studio slice |
| `tile_views.py`, `tile_render.py` | looking at it |
| `sparse_fit.py` | `--matte`; `compose_fitted` now returns the matte as a 4th value |
| `batch_assemblage_geometry.py` | `--coords`, `--max-per-work` |

---

## Addendum — the paid Tripo API, wired in

Reviewed 2026-08-10 against three hosted-Tripo trials of the same kind of
assemblage. The verdict on the local model stands and is stronger than §1 put
it: what TripoSR loses is not detail, it is **separation**. A new measure makes
that exact.

`interstice_fill` is the fraction of *enclosed* background — the gaps between
fragments, found by flooding the outside and inverting — that the mesh fills in.
Scored over all 96 meshes:

| backend | iou | thin_recall | **gaps kept** | fill | faces | bodies |
|---|---|---|---|---|---|---|
| TripoSR, drift, mc 128 | 0.805 | 0.805 | **0.222** | 0.129 | 31,920 | 15.6 |

**TripoSR fills 78% of the gaps.** An assemblage reads as an assemblage because
of the space between its fragments, and that is the number that says the local
model destroys it. A fine iou is compatible with having fused the whole
composition into one rock, which is what it does.

### What was built

| | |
|---|---|
| `tile_tripo.py` | the hosted API as a drop-in for `tile_triposr.py` |
| `tile_measure.py` | one ruler, shared by both backends, so they compare |
| `tile_score.py` | re-score any mesh directory, any backend, and tabulate |

`tile_model.py` gained `--mesh-file` and does not otherwise care which backend
produced the meshes, so the tile, the void, the union, the split and the slice
all work unchanged.

Notes on the integration:

- **The matte is the advantage.** Tripo takes a transparent PNG, so `sparse_fit`'s
  own alpha goes straight in — trimmed to its bounds with 6% air and squared —
  and no background removal ever has to guess at an assemblage.
- **Texture and PBR are off by default.** The tile prints in white and the
  geometry is booleaned; a texture is paid for per task and thrown away.
- **`--generate-parts` is worth a try.** The API can return the model split into
  parts, and fusing the fragments is precisely the failure being fixed.
- **Nothing runs without `--spend`.** Default is a dry run that prints the
  estimate. Completed points are skipped on re-run, so a resumed run costs
  nothing for work already paid for, and the account balance is read before and
  after so the recorded cost is the real one, not the estimate.
- **The key is read from `TRIPO_API_KEY` only** — never a command-line argument,
  where it would land in shell history and the process list, and never written
  to a sidecar.
- Installed into the same isolated 3D-tools venv as TripoSR
  (`pip install tripo3d`); the project's pinned environment is untouched.

### What the API actually gave, and what it cost

Run 2026-08-10. 93 of 95 points regenerated on **v3.1-20260211**, tile rebuilt,
sliced. The hosted model is better on the measure that matters and the tile is
now architecture rather than geology.

| | TripoSR | Tripo P1 | **Tripo v3.1** |
|---|---|---|---|
| **gaps kept** | 0.222 | 0.06 | **0.42** |
| iou | 0.805 | 0.53 | 0.60 |
| faces out | 32 k | 4 k | 1.48 M -> 40 k |
| credits each | 0 | 40 | **20** |
| reads as | one fused rock | an invented object | arches, openings, cornices |

**P1 was the wrong model and the mistake was mine.** It is marketed as clean
low-poly and it behaves like it: 4 k faces, and its own preview abstracts the
collage into a single simplified object. 40 credits a task, double v3.1's.

**IoU falls while the object improves**, which is worth understanding rather than
explaining away. v3.1 reconstructs fragments *in depth* — they move apart in
three dimensions — so the frontal silhouette matches less tightly than a flat
fused slab does. `interstice_fill` is the measure that tracks the thing wanted;
iou alone would have picked the worse model.

### Three things the API run cost that the local one did not

- **The raw output cannot be booleaned.** v3.1 returns ~1.5 M faces in 600-800
  **open** components, none watertight, and hole filling does not close them.
  `solidify()` voxelises at 256 divisions — a 0.12 mm voxel on a 1.25 in object,
  finer than the nozzle, so nothing printable is lost and any gap wider than a
  voxel survives — then decimates to 40 k with `preservetopology=True`. Result:
  watertight, 10 bodies, **98.5% of volume retained**. About 1.6 minutes each,
  which is 25× the generation time.
- **glTF is y-up, this project is z-up.** Meshes arrive lying on their side, and
  an orientation search covering only 6 of the 24 possible framings scored a
  perfectly good mesh as broken. Both fixed; `tile_measure.AXES` now holds all 24
  and searches cheaply on a thinned draw before rendering the winner in full.
- **Moderation.** Two points, `clash_02` and `empty_89`, were refused with
  `code 2008`, content policy, on the input image. The corpus carries classical
  and baroque figure fragments. 93 of 95 is not a problem for the tile; it is a
  problem for any run that needs a specific coordinate.

### The tile, both ways

| | TripoSR | **Tripo v3.1** |
|---|---|---|
| volumes | 94 | 93 |
| built coverage | 17.4% | **11.5%** |
| faces | 2.20 M | 2.80 M |
| watertight, one body | yes | yes |
| debris dropped | 0.44% | 0.82% |
| solid volume | 72.4 in³ | 66.2 in³ |
| **one tile** | 536 g, 71.5 h | **465 g, 59.7 h** |
| **11.1 sq ft** | 5.9 kg, 33.1 days | **5.2 kg, 27.6 days** |

**Coverage drops and that is the honest price of the fix.** TripoSR inflated
coverage by fusing fragments into solid mass; v3.1 keeps the voids, so the same
fragments cover less ground while reading far better. The print gets cheaper for
the same reason.

### Billing, which cost more time than the code

`platform.tripo3d.ai` and `tripo3d.ai` are **separate business lines with
separate balances**. A Tripo Studio Pro subscription — 3,000 credits a month, and
the plan that carries full commercial rights — buys nothing on the API, which is
pay-as-you-go. A Studio account still reads `balance: 0` here and every task
returns 403 / `code 2010`.

Two key types exist and only one works: the SDK rejects anything not prefixed
`tsk_`, and a Studio `tcli_` token is not it.

**Commercial rights on API output are still unverified.** Studio Pro states them
explicitly; the API terms were not checked. This project already ruled out
FLUX.1 Fill on exactly this ground, so it should be settled before anything here
goes into a saleable edition.

---

## What to decide next

1. **Is a ruin field the object you want?** Everything else follows from this. If
   yes, the tile is finished apart from printing it. If no, the route is the
   count and the footprint, not a different model.
2. **Do not print the ground.** It is 98% of the material. CNC foam, casting or a
   ribbed shell for the landscape, with only the masses printed, changes 5.9 kg
   and 33 days into something much smaller.
3. **Paid Tripo, if at all, on one image only**, judged solely on whether the
   gaps between fragments survive. Nothing else about TripoSR failed.
4. **The 40 × 40 in model is 11 tiles of 4 pieces = 44 printed parts.** Worth
   knowing before it is designed.
