# The four sites, and what the fabrication actually costs

Produced by `site_survey.py`, 2026-08-10. Raw data in
`data/gap_space/site_survey/survey.json`, readable table in `SITES.md` beside it.
272 candidates across 7 protocols, 40 conditions appearing in 2 or more.

---

## 1. How a site is defined across models, and why it cannot be a coordinate

Each protocol's 2D layout is an independent UMAP with **no alignment to any
other**. `(8.30, 8.01)` in CLIP has no counterpart in DINOv2. The project already
measured that layout rank barely tracks embedding rank even inside one protocol.
So "the same site seen in six models" cannot mean the same x,y.

Two definitions do hold, and the survey computes both:

- **Condition** — the same two periods colliding on the same value axis. Portable
  by construction: periods and axes are properties of the corpus and of
  `affordance_vocab.json`, not of any layout.
- **Constituency** — the same actual works on the ground, measured as Jaccard
  overlap between the neighbour sets each protocol's best point recruits.

**The gap between them is the exhibit.** A condition every model finds, recruiting
the same works, is a site the paradigm agrees about. A condition every model finds
while recruiting *different* works is a site where the models agree a collision
exists and cannot agree who is party to it — which is the subject/object question
directly: not what the building is, but which works are even eligible to be in the
room.

**Read overlap against its chance floor, not in absolute terms.** With k=12 out of
a few hundred works per period pair, chance Jaccard is about 0.010. An observed
0.079 is therefore not "almost no agreement" — it is **eight times** chance. Every
figure below is given as a multiple.

## 2. The ranked conditions

Ranked by **intensity** = axis distance × mean score: genuinely opposed *and* both
parties really present. Ranking by breadth instead surfaces medieval × renaissance
and Classical × medieval, which appear in every protocol precisely because they
are the normal sequence of history, and are therefore the least compelling ground.

| # | condition | axis | dist | protocols | intensity | overlap |
|---|---|---|---|---|---|---|
| 1 | medieval × industrial | labour | 3.06 | 5 | 2.20 | **7.95×** |
| 2 | neoclassical × deconstructivism | legibility | 3.26 | 5 | 1.60 | 4.08× |
| 3 | prehistoric × High Tech | labour | 3.04 | 4 | 1.43 | 6.22× |
| 4 | medieval × High Tech | labour | **3.28** | 4 | 1.36 | **0.71×** |
| 5 | prehistoric × parametric | labour | 2.88 | 5 | 1.33 | **0.49×** |
| 6 | prehistoric × postmodernism | address | 2.51 | 3 | 1.28 | **0.00×** |
| 7 | baroque × industrial | legibility | 2.78 | 3 | 1.27 | 1.34× |
| 8 | Brutalism × deconstructivism | legibility | 2.65 | 7 | 1.26 | 1.43× |
| 10 | medieval × postmodernism | authority | 2.66 | 5 | 1.09 | 3.39× |

## 3. The recommended four

Chosen so the set spans two axes, both kinds of constituency behaviour, and both
model families — and so that three of the four share a spine.

### Site A — medieval × industrial, on **labour**

`dist 3.06 · intensity 2.20 · 7.95× chance · CLIP, DINOv2, MetaCLIP 2, SigLIP 2, verbal`

*The hand and the guild visible* against *the machine and the standard visible*.
The strongest site in the corpus on both measures at once: highest intensity, and
the highest constituency agreement of any condition found. Every model that finds
it recruits substantially the same works.

Already rendered at `clash_00` in CLIP space (x 8.30, y 8.01): the survey returns
**one** industrial work at weight 0.445 against fifteen medieval works at 0.05–0.08
each, and the assemblage renders that asymmetry directly — a few enormous
standardised rolled sections crossing a crowd of small singular hand-cut fragments.
The axis is the drawing's structure, not its caption.

**This is the anchor.** It is the site where the argument is least contestable.

### Site B — medieval × High Tech, on **labour**

`dist 3.28 · intensity 1.36 · 0.71× chance · FLUX VAE, SDXL VAE, MetaCLIP 2, SigLIP 2`

**The widest opposition anywhere in the corpus**, and constituency agreement
*below chance* — the models that find it do not agree on a single constituent.
Note who finds it: the two compression protocols plus the two later
image–text models. **Neither DINOv2 nor CLIP sees this collision at all.**

Set directly against Site A this is the whole comparison in one pair. Same axis,
same period, two different machine-ages — the factory and the exhibited machine —
and the model families split on which one they can even perceive. Site A is what
the semantic encoders agree about; Site B is what only the compression protocols
find, and they cannot agree who is on the ground.

### Site C — neoclassical × deconstructivism, on **legibility**

`dist 3.26 · intensity 1.60 · 4.08× chance · CLIP, DINOv2, MetaCLIP 2, SDXL VAE, SigLIP 2`

*The order withheld from its occupant* against *the order legible at a glance*.
The second-widest opposition in the corpus, high agreement, and the only top site
found by both a semantic encoder and a VAE in strength.

This is the site most directly about subject formation: whether the occupant is
permitted to read the order they are standing inside. It carries a second axis, so
the exhibition is not four variations on hand-versus-machine.

### Site D — one of two, and this is a curatorial choice

| | `prehistoric × High Tech` | `prehistoric × parametric` |
|---|---|---|
| dist | 3.04 | 2.88 |
| intensity | 1.43 | 1.33 |
| overlap | **6.22×** | **0.49×** |
| protocols | CLIP, DINOv2, MetaCLIP 2, verbal | CLIP, DINOv2, FLUX VAE, MetaCLIP 2, verbal |

**prehistoric × High Tech** is the purest statement of the labour axis available —
the hand at its most primal against the machine at its most theatrically displayed
— and the models strongly agree who is in it.

**prehistoric × parametric** is contested below chance and spans both model
families, and would give the set three high-agreement sites against two contested
ones rather than three against one.

Recommendation: **prehistoric × High Tech**, for conceptual clarity, since Site B
already carries the contested case and carries it more sharply (0.71× at the
corpus's widest opposition beats 0.49× at its ninth).

## 4. Scale, worked through

Taking a **36 × 48 in** site model, 12 sq ft, at the top of your range.

| site model scale | 36 in represents | 48 in represents | territory |
|---|---|---|---|
| 1/32" = 1' | 1,152 ft | 1,536 ft | 0.22 × 0.29 mi |
| **1/64" = 1'** | **2,304 ft** | **3,072 ft** | **0.44 × 0.58 mi** |
| 1/128" = 1' | 4,608 ft | 6,144 ft | 0.87 × 1.16 mi |

**1/64" = 1' is the recommendation.** At 1/128" a 200 ft site is a 1.6 in void —
too small to read as ground held in reserve. At 1/32" the whole field is only a
quarter mile and stops reading as territory.

The step to the section models then works out cleanly:

- a 200 ft site is a **3.1 in** void in the site model at 1/64"
- the same 200 ft as a section model at 1/8" = 1' is **25 in**
- an **8× step**, which is legible as a deliberate zoom rather than an arbitrary one

At 1/2" = 1' a 24 in section model covers only 48 ft — one room sequence, not a
building. Use 1/2" only if the intervention is a single room; 1/8" if it is a
building. Mixing them across the four sites is defensible if each is labelled, but
holding all four at 1/8" makes them comparable, which is the point of having four.

## 5. Two things that gate fabrication and should be decided now

### 5a. The large prints cannot be made from the existing masters

32 × 40 in at 150 ppi is **4,800 × 6,000 px**. Two problems:

- **Resolution.** The current render ceiling is `--out-size 4096`, so the long
  dimension is short by about 1.5×.
- **Aspect.** Every assemblage, atlas and rift output in the project is **square**.
  A 32 × 40 print is 4:5. There is no non-square render path.

And this cannot be fixed by upscaling, for a reason already established in
`print_export.py`: **re-rendering at another size changes the composition**,
because the form set depends on resolution. The exporter deliberately uses the
existing file rather than re-rendering.

**Consequence: the large prints have to be composed at their final size and aspect
from the start.** That is a change to the render path — non-square canvas plus a
higher ceiling — and it must happen before any large print is designed, not after.
The 12 × 12 in prints are fine at existing resolution (3072 px = 256 ppi at 12 in).

`PRINT_SPEC_FOR_QUOTE.md` is also written for 16 × 16 on a 20 × 20 sheet and needs
a second and third size added before it goes to a supplier.

### 5b. Image-to-3D is the one genuinely unproven step in the plan

Everything else in the site-model pipeline is built and tested. This is not:

- **modly** is installed at `../tools/modly` and has **never been run on actual
  output**. 6 GB card, ~5 GB free; most image-to-3D models want 8–16 GB.
- **Tripo** is cloud and paid, cost unknown, and the Gemini credit being exhausted
  suggests budget for cloud inference is not currently allocated.
- The **boolean union** step has documented traps already recorded in this
  project: trimesh's quadric decimation introduces non-manifold edges with no open
  edges, so `is_watertight` reports False and hole-filling does nothing — use
  pymeshlab with `preservetopology=True`. And OpenCASCADE's
  `ShapeUpgrade_UnifySameDomain` with any angular tolerance merges non-coplanar
  faces into invalid shapes with nearly doubled volume; only exact coplanar
  merging is safe.
- **Print volume.** 12 sq ft is ~1.11 m² of plan area. On a 256 mm bed that is
  17 tiles at perfect packing and realistically 25–35 parts once height and
  supports are accounted for. That is a multi-week machine commitment and a real
  filament cost, and it should be estimated before the site model is designed
  rather than discovered during it.

**Recommendation: prove this on one site before committing the field.** Take Site A,
which already has rendered assemblages at `clash_00`, run it through modly and
through a Tripo trial, boolean the result, and print one tile. That answers the
hardware question, the cost question and the mesh-quality question for the price
of one tile, and it is unblocked today.

## 6. What each site needs, as a deliverable list

Per site, from your brief:

- a reserved void in the shared site model, at 1/64" = 1'
- one section model at 1/8" = 1' (or 1/2" if the intervention is a single room)
- several 12 × 12 in prints — assemblage or sparsefit, existing resolution is fine
- one or two 32 × 40 in prints — **blocked on §5a**
- plan and/or section drawings in the Gemini or SDXL manner — **the Gemini path
  is the one that passes the period-separation test and its credit is exhausted**

The cross-model comparison the brief asks for is already computable for all four:
each site exists in 4–5 protocols with recorded coordinates, so the same condition
can be composed once per protocol and hung as a row. That needs no new machinery —
`sparse_fit` takes a coordinate and a protocol today.
