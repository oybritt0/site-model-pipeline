# What data this needs

The repository ships **no data**. This is what it expects, where it goes, and how
much of the pipeline runs without each part.

Everything resolves relative to a project root, found by walking up from the
scripts until a directory containing `data/catalog/manifest.json` is found
(`gap_space_core.find_root`). So the layout is:

```
<root>/
  *.py                      the pipeline
  data/
    catalog/manifest.json           4,475 works: id, period, category, path
    embeddings/
      protocols.json                the protocol registry
      index.json                    work ids and which embedded successfully
      affordance_vocab.json         the five value axes and period positions
      misclosure_stability.json     per-work cross-protocol agreement
      projection_stress.json        per-work projection tear
      *_latent_2d.json              one 2D layout per protocol
      *_features.npy                one feature matrix per protocol
      verbal_profile.npy            per-work term scores
    dino_dense/coseg/labels/*.npy   per-work co-segmentation, one file per work
    gap_space/                      all output lands here
```

## Sizes

| | size | needed for |
|---|---|---|
| `catalog/manifest.json` | 1.9 MB | everything |
| `embeddings/` json + npy | ~92 MB | everything |
| `dino_dense/coseg/labels/` | 25 MB | cutting fragments (step 2) |
| **derived total** | **~118 MB** | |
| source images | **~4.0 GB** | cutting fragments (step 2) |

`bootstrap_data.py` copies the derived 118 MB across from an existing checkout of
the parent project. It does not copy images.

## The source images are not redistributable

`manifest.json` records a `path` per work pointing at a local image file. Those
4,475 images are third-party artwork gathered from museum and archive sources.
They are **not** in this repository and should not be added to it.

Without them, `rift_atlas.forms_at` cannot cut fragments, so **step 2
(`tile_assemblages.py`) will not run**. Everything else does.

## What runs with only the derived data

| | runs? |
|---|---|
| `tile_points.py` — coordinate selection | **yes** |
| `tile_stability_check.py` | **yes** |
| `tile_protocol_mix.py` | **yes** |
| `tile_map.py` | **yes** |
| `tile_assemblages.py` — compositions | **no**, needs the images |
| `tile_tripo.py` / `tile_triposr.py` | only if assemblages exist |
| `tile_model.py` — terrain, union, STL | **yes**, given meshes |
| `tile_print.py`, `tile_views.py`, `tile_render.py` | **yes**, given an STL |
| `tile_score.py`, `tile_compare.py`, `tile_contact.py` | only with assemblages |

So the geometry and fabrication half is fully workable from the derived data
alone. To work on composition, you need the corpus.

## Handing over a working set instead

If the point is to let someone iterate on the *model building* rather than on the
corpus, the smallest useful handover is not the images. It is:

- the derived 118 MB, and
- one tile's `assemblages/` directory — the RGB images and their mattes, roughly
  120 MB for 96 points — which is enough to re-run steps 3, 4 and 5 end to end
  and to try different 3D backends, terrain rules and packing.

That is ~240 MB total and needs no corpus at all.

## Rebuilding the derived data from scratch

Not part of this repository. The layouts, feature matrices and co-segmentation
come from the parent project's embedding and segmentation scripts, which are a
separate body of work.
