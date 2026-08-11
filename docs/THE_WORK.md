# The work

Read this before `EXHIBITION_BRIEF.md`. The brief leads with measurement, and
that is the wrong order. The measurement is apparatus. This is the work.

---

## What the project makes

**Architecture for sites that exist only in the latent space of models trained on
architectural history.** Plans, sections, and models of buildings that resolve a
conflict which has no location in the world.

The gap is not a hole in a dataset. **It is a site** — in the architect's sense: a
piece of ground with adjacent conditions, inherited material, and a conflict that
the intervention has to answer. What makes it a site rather than an absence is
that it has *politics*, and the politics are declared.

## The site, worked through one example

`clash_00` in CLIP space, coordinate x 8.30 / y 8.01.

**Site conditions.** Two codifications hold this ground and pull opposite ways:
`03_medieval` at weight 0.55 and `16_industrial` at 0.45. The axis they are widest
on is **labour** — *the hand and the guild visible* against *the machine and the
standard visible* — at distance 3.06, the fourth-widest opposition anywhere in the
corpus. Both are genuinely present; the selector multiplies the two weights, so a
site does not qualify unless both parties really have a claim.

**Survey.** The lift returns who claims the ground and how hard. Here: **one**
industrial work at 0.445, then fifteen medieval works at 0.05–0.08 each. That
distribution is the site's asymmetry, and it is a fact about the ground, not a
compositional choice.

**Parti.** `sparse_fit` builds from material found on site. Each fragment keeps the
position it held in its own source, is sized by how strongly its work claims the
ground, and is turned on its own principal axis. Then the whole thing is shrunk
**rigidly** until it clears the frame — so the parti is never adjusted to fit the
sheet. Overlaps, gaps, and stranded outliers survive exactly.

The result at this site: a few enormous raking rolled sections — machine-made,
standardised, repeated — crossing a crowd of small singular hand-cut fragments,
tracery and a crocketed canopy and a carved relief. **The labour axis is not
described in the drawing. It is the drawing's structure.** One standard element at
scale against many singular ones, because that is what the survey found.

**Drawing.** `assemblage_geometry` replays that placement into closed curves with
real arcs — SVG, DXF, OBJ, STEP — each curve tagged with its period, material,
organization, gestalt and surface. It goes to Rhino. At this point it stops being
an image and becomes a document you could build from.

**Rendering.** `regional_diffusion` treats each closed curve as a room and asks it
to be its own period's architecture, in plan or in section, with a child room
inheriting its parent's material as context. The Gemini `wide1` sheet is what this
produces: an archipelago plan, rotundas and radial stairs and an arcaded spine
linking two centralised masses, a timber-framed pitched insert, poché, paving — and
every outer contour still the traced silhouette of a fragment cut from a real work.

That is the whole method: **site → survey → parti → drawing → rendering.** It is
the ordinary architectural sequence, performed on a site in model space.

## Why the conflict is the point

An architect intervenes where conditions conflict. The five axes are the conflict,
stated in advance and open to challenge:

| axis | one pole | the other |
|---|---|---|
| authority | centralised and staged | refused or distributed |
| legibility | order withheld from its occupant | order legible at a glance |
| permanence | built as monument, to outlast | built as flow, obsolescence accepted |
| labour | the hand and the guild visible | the machine and the standard visible |
| address | addressed to a public assembled | addressed to a subject alone |

A `clash` site is where two periods take opposed positions on one of these. The
four `rift_atlas` methods are four different architectural responses to such a
site, and they already have architectural names:

- **rift** — the building is split by the axis that divides the site, along a seam
  whose angle *is* the axis. Sited on the fault.
- **stratum** — one element only, cut from every work on the site and stacked
  oldest to newest. A core sample through the ground.
- **palimpsest** — what recurs across the neighbourhood is convention and is drawn
  heavy; what appears once is mutation and is drawn faint. A contextual reading.
- **negative** — the parts become apertures cut from a field rather than figures on
  one, so what is drawn is what the site *withholds*. Poché, and the unbuilt.

**A spatial solution to a latent spatial problem.** The conflict is spatial in the
model's terms — two codifications occupying one region — and the answer is spatial
in ours.

## What this changes about the measurements

I had the hierarchy inverted. Corrected:

**`period_divergence` is the central measurement**, because it asks the
architectural question: *can the instrument hold two opposed value systems in two
different rooms of the same building?*

| | can a room be told from another period's room? | p |
|---|---|---|
| real works | **0.561** | 0.000 |
| Gemini | 0.117 | 0.06 |
| SDXL + local LoRA | 0.039 | 0.31 |

**SDXL cannot.** Its separation is statistically indistinguishable from zero — ask
it for medieval and for High Tech and you get the same room. That is not a critique
of generative AI in the abstract; it is a **fitness test for a rendering
instrument, against a specific brief, which the local stack fails and Gemini
partly passes.**

And it inverts what leads. The *drawings* hold the difference, because every
fragment is cut from a real work and carries that work's period with it. The
diffusion pass flattens it. So **the assemblages and the vector drawings are the
work, and diffusion is a rendering pass currently on probation.**

**The return trip is demoted to site verification.** "Is this ground actually
unoccupied?" is a real question a survey asks, and the answer across six protocols
is consistent: **median gap closed 0.0%, without exception.** Read architecturally
that is not a failure — the proposal occupies the site without becoming part of
the history it was drawn from. It stays a proposal. That is what unbuilt work is.
The percentile spread (55 to 87 across protocols) is a footnote about instrument
disagreement, not a thesis. Details in `RETURN_TRIP_RESULTS.md`.

## The thesis question, answered

None of the three options in `EXHIBITION_BRIEF.md` §7. Not *(a)* history has
holes, not *(b)* the machines are biased, not *(c)* watch the generator drag you
home. Those are all readings in which **the models are the subject and the drawings
are evidence.**

> **(d) The models are the site. The drawings are the work.**
>
> Unbuilt architecture for grounds that exist only inside a model's account of
> architectural history, sited where two value systems make opposed claims, drawn
> from material found on site.

(b) survives inside (d) as the reason the sites are real and legible rather than
arbitrary — the apparatus is what makes the ground surveyable. It is site notice,
not thesis.

## What follows immediately

**Restore the Gemini credit.** It is $25 and it is the highest-leverage spend in
the project. Gemini is the only renderer that partially passes the fitness test —
three times SDXL's period separation and the only one approaching significance —
and by eye it is not close. The `wide1` sheet took 100 calls and 36 minutes.
Everything downstream, including which images get printed, is gated on having a
renderer that can hold a difference.

**The style references are an authorship position and should be stated.** The
Gemini runs used `lai_1`, `lai_2` and `RB_1` — Jimenez Lai, and your own drawing.
The drawing convention is sourced and named rather than defaulted to whatever the
model does. That belongs in the disclosure alongside the authored axes, and it is
a stronger position than hiding it.

**Models, not only drawings.** `assemblage_geometry` already emits closed solids —
OBJ and STEP — and `glb_to_step_solids.py` exists. Physical models of a building
sited in latent space are available and have barely been discussed. For an
architecture audience a model on a plinth does more than any wall text about
embeddings.

**Titles are already written.** Every clash sidecar carries the two periods, the
axis, and the distance. *Medieval / Industrial — labour, 3.06.* That is the title
system, and it names the argument rather than a serial number.

**Selection follows the sites, not the measurements.** Which drawings get printed
should follow from which sites carry the widest and most legible conflicts, and
whether the drawing answers them. The return-trip numbers do not select images.
That was my error and it is corrected here.
