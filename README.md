# Site model pipeline

Turns a 2D layout of a machine-learning model's latent space into a printable
architectural site model. A window of the layout becomes a 12 × 12 in tile; the
works near each chosen coordinate are cut into fragments and composed into an
assemblage; each assemblage is inferred into a volume; the volumes are placed at
their layout coordinates on terrain derived from the layout's own occupancy, and
booleaned into one watertight solid.

Scale is **1/64" = 1'**, so a 12 in tile represents 768 × 768 ft.

Two tiles have been built with this. Both are documented in
[`docs/SITE_MODEL_TILE_TEST.md`](docs/SITE_MODEL_TILE_TEST.md), which is the
record of what was measured and where the plan turned out to be wrong.

---

## The pipeline, in order

| step | script | what it does |
|---|---|---|
| 1 | `tile_points.py` | pick 80–96 coordinates inside a window of one protocol's layout, and convert them to inches on the tile |
| 2 | `tile_assemblages.py` | compose one `sparse_fit` assemblage per coordinate, saving an RGB image **and a straight-alpha matte** |
| 3a | `tile_tripo.py` | infer a volume per assemblage using the **paid Tripo API** (v3.1). Costs money. |
| 3b | `tile_triposr.py` | the same, using **local TripoSR**. Free, and much worse — it fuses the fragments. |
| 3c | `batch_assemblage_geometry.py` | the third route: trace the fragment outlines and extrude. Exact, prismatic, closed, tagged. |
| 4 | `tile_model.py` | terrain, placement, boolean union, the site marking, clip to the tile → STL + PLY |
| 5 | `tile_print.py` | split to the bed, slice each piece with the Bambu Studio CLI, report real filament and time |

Looking at it, and checking it:

| script | what it answers |
|---|---|
| `tile_views.py`, `tile_render.py` | plate, plan, cut block and elevation, without an OpenGL context |
| `tile_contact.py` | a labelled colour contact sheet of every assemblage in a tile |
| `tile_map.py` | where the sites and tiles sit in all seven model spaces |
| `tile_score.py`, `tile_measure.py` | score any directory of meshes against the assemblages they came from |
| `tile_compare.py` | the inferred route against the traced route, same coordinates |
| `tile_protocol_mix.py` | which protocol mixes the media, and where to cut for it |
| `tile_stability_check.py` | whether clash ground sits low in cross-protocol stability (it does not) |

---

## Install

Two environments, deliberately separate.

**1. The project environment** runs everything except the 3D inference.

```bash
pip install -r requirements.txt
```

`OCP` (OpenCASCADE) is only needed for STEP output from
`assemblage_geometry.py`; everything else works without it.

**2. A separate venv for the 3D step.** Do not merge it into the project
environment — the diffusion side of the original project is version-pinned and
TripoSR pins conflicting versions.

```bash
git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git tools/triposr
cd tools/triposr
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install omegaconf einops trimesh rembg huggingface-hub imageio xatlas moderngl onnxruntime
./.venv/Scripts/python.exe -m pip install tripo3d          # only for the paid API
```

`--system-site-packages` shares torch by reference so you do not download 2.5 GB
again. pip correctly refuses to uninstall anything from the parent environment
and shadows it locally instead.

**`torchmcubes` will probably not build** — it needs MSVC and a CUDA toolkit. It
is not required. Copy `triposr_patch/_mc_skimage.py` into
`tools/triposr/tsr/models/` and change the import at the top of
`tsr/models/isosurface.py` to:

```python
try:
    from torchmcubes import marching_cubes
except ImportError:
    from ._mc_skimage import marching_cubes
```

Read the docstring in that file before touching it. Two conventions have to be
matched exactly and both are easy to get wrong in ways that look like success.

---

## Data

**The repository contains no data.** See [`DATA.md`](DATA.md) for exactly what is
needed, how big it is, and what you can run without it.

Short version: about **118 MB** of derived data (2D layouts, feature matrices,
the catalogue manifest, per-work co-segmentation labels), plus about **4 GB of
source images** which are third-party artwork and are not redistributable.

`bootstrap_data.py` copies the derived data across from an existing checkout of
the parent project.

**Without the source images you can still run** steps 1, 4 and 5, every analysis
script, and every renderer — so the whole geometry and fabrication half of the
pipeline works. Step 2 needs the images, because it cuts fragments out of them.

---

## Running it

```bash
# 1. coordinates
python tile_points.py --protocol flux_vae --centre 8.67,-0.75 -n 96 \
    --min-sep-in 0.5 --size-min-in 0.9 --size-max-in 1.6 \
    --mix "clash:16,mixed:58,empty:21" --out out/mytile

# 2. compositions, ~25 s each
python tile_assemblages.py --points out/mytile/points.json \
    --out out/mytile/assemblages --protocols drift

# 3. volumes. THIS SPENDS MONEY -- nothing is sent without --spend
"tools/triposr/.venv/Scripts/python.exe" tile_tripo.py \
    --points out/mytile/points.json --assemblages out/mytile/assemblages \
    --out out/mytile/meshes --model-version v3.1-20260211 \
    --download-only --concurrency 4 --spend

# then solidify the raw GLBs (no key needed, project environment)
python tile_tripo.py --convert-only --out out/mytile/meshes \
    --assemblages out/mytile/assemblages --points out/mytile/points.json

# 4. the tile
python tile_model.py --mesh-file mesh_tripo.obj \
    --points out/mytile/points.json --meshes out/mytile/meshes \
    --out out/mytile/tile --base-in 0.08 --relief-in 0.45 \
    --relief-gamma 1.9 --smooth-in 0.20 --no-void-cut

# 5. printable pieces and a real slicer estimate
python tile_print.py --stl out/mytile/tile/tile_density_upright_tripo.stl \
    --out out/mytile/print
```

### The Tripo API key

Never passed as an argument. `tile_tripo.py` reads, in order:

1. `TRIPO_API_KEY` in the environment
2. the path in `TRIPO_API_KEY_FILE`
3. `~/.tripo_key`, first line

It must be a **`tsk_`** key from `platform.tripo3d.ai` → API Keys. A `tcli_`
token from the Tripo Studio web app is a different product and will be rejected.

**Studio credits are not API credits.** They are separate business lines with
separate balances; a Studio Pro subscription reads `balance: 0` on the API and
every task returns 403 / `code 2010`.

Cost at the time of writing: **20 credits per untextured image-to-model**, so
about $19 for a 96-point tile. `--probe N` runs a handful for choosing a model
version, and completed points are skipped on re-run so a resumed job never pays
twice.

---

## Things that cost a day each, recorded so they do not again

- **Model choice.** Tripo `P1` is a *clean low-poly game asset* model: 4 k faces,
  and it abstracts the collage into an invented object. Use **`v3.1`**. It is
  also half the price.
- **glTF is y-up, this pipeline is z-up.** Meshes arrive on their side. If you
  write your own orientation search, cover all **24** framings — six ordered axis
  pairs at four sign combinations. Six is not enough, and a mirrored match scores
  like a total failure, so a perfectly good mesh reads as broken.
- **Tripo's raw output cannot be booleaned.** ~1.5 M faces in 600–800 *open*
  components. `tile_tripo.solidify()` voxelises then decimates with
  `preservetopology=True`, giving a watertight 40 k-face solid that keeps 98.5%
  of the volume. Hole-filling alone does not work.
- **The Bambu profile omits `printable_height`**, so the CLI defaults to about
  100 mm — a tenth of the machine — and taller pieces fail with "no object is
  fully inside the print volume", which reads like a broken mesh. A plain
  40 × 40 × 105 mm box fails identically. `tile_print.py` passes
  `--printable-height` for this reason.
- **STL loses the solid.** Re-merging a 2.4 M-face STL on load drops degenerate
  faces and leaves it a few faces short of closed, which `manifold3d` refuses.
  `tile_model.py` writes a **PLY** alongside, and the downstream steps read that.
- **Splitting the tile severs masses that span a seam**, leaving unprintable
  off-cuts floating inside the neighbouring piece. Dropped per piece.
- **Never `decimate` without `preservetopology=True`.** trimesh's quadric
  decimation introduces non-manifold edges with no open edges, so `is_watertight`
  returns False and hole-filling silently does nothing.

---

## Licence

Not set. The code is the author's; the corpus it operates on is third-party
artwork and is **not** included here. Decide the licence before making the
repository public.
