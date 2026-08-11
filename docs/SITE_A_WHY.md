# Why Site A is where it is

The tile locates Site A but does not excavate it. This is the argument for the
location, so that moving it is a decision made against evidence rather than a
free hand.

Coordinate: **CLIP x 8.30 / y 8.01**, which is the centre of the tile,
plan **6.00 / 6.00 in**. Diameter 3.1 in = **200 ft** at 1/64" = 1'.

---

## 1. What the condition is

**Medieval × industrial, on the labour axis.** *The hand and the guild visible*
against *the machine and the standard visible* — the two poles of
`affordance_vocab.json`'s labour axis, 3.06 apart, the fourth-widest opposition
anywhere in the corpus.

At this coordinate the lift returns **medieval 0.588 against industrial 0.412**.
Both parties really hold the ground; neither owns it.

That balance is the point, and it is enforced rather than hoped for.
`sparse_fit.clash_points` scores a candidate as **axis distance × the product of
the two period weights**, so a point that is 80% one period and 5% another
cannot score, however far apart those two sit. Site A scores **0.741**.

## 2. Why this condition rather than another

From `site_survey.py`, 272 candidates across 7 protocols:

| | medieval × industrial | rank of 40 |
|---|---|---|
| axis distance | 3.06 | |
| **intensity** (distance × mean score) | **2.20** | **1st** |
| constituency agreement | 7.95× chance | **2nd** |
| protocols that find it | 5 of 7 | |

**Correction to `EXHIBITION_SITES.md` §3, which this document originally
repeated.** That file calls it "the highest constituency agreement of any
condition found". Checked against all 40 conditions in `survey.json`, it is
**second**. `neoclassical × industrial` on permanence reaches **10.19×** — but on
3 protocols and at intensity 0.68, against this one's 5 protocols and 2.20. The
brief's table only ranked the intensity leaders, so the claim was true of that
table and not of the corpus.

So the accurate statement is: **first on intensity, second on agreement, and the
only condition in the top three of both.** That is still the best combined
position available, and it is the reason to choose it — but it is not a clean
sweep and should not be described as one.

**A second nuance the survey records and the summary loses.** `n_shared_works`
for this condition is **0**. The 7.95× is a *pairwise* mean Jaccard between
protocols; no single work is recruited by all five. The models agree
substantially in pairs about who is on this ground, and unanimously about
nobody.

Ranking by breadth instead would surface medieval × renaissance and Classical ×
medieval, which appear everywhere precisely because they are the ordinary
sequence of history — the least interesting ground available.

## 3. The asymmetry is a fact about the ground

The survey at this coordinate returns **one industrial work at weight 0.445,
then fifteen medieval works at 0.05–0.08 each**.

One standardised, repeated thing at scale against a crowd of singular hand-cut
ones. That distribution is what the assemblage draws, and it is why the labour
axis is the drawing's *structure* rather than its caption.

### Does the assemblage actually show it?

Tested rather than assumed. The six points traced by `assemblage_geometry` carry
a period on every fragment, so the plan area each period holds can be measured
against the lift weights that produced it.

**At Site A, yes, and precisely.** Ten fragments: **two industrial holding 60% of
the plan area, eight medieval holding 40%** — and the single largest fragment is
industrial at **57% on its own**. One standardised element at scale against a
crowd of small singular ones, which is exactly the 0.445-against-fifteen
distribution §3 describes. The drawing carries the survey.

**Across the other points it does not hold reliably.** Same condition, same
method:

| point | lift weights | fragment area |
|---|---|---|
| **site_a** | med 0.59 / ind 0.41 | ind 60% (n=2), med 40% (n=8) |
| clash_01 | med 0.55 / ind 0.46 | ind 58% (n=5), med 42% (n=5) |
| clash_03 | med 0.67 / ind 0.33 | med 86% (n=6), ind 14% (n=4) |
| clash_04 | med 0.56 / ind 0.31 | med 71% (n=5), ind 19% (n=4) |
| **clash_02** | med 0.55 / ind 0.45 | **ind 85% (n=8), med 15% (n=2)** |

`clash_02` inverts: the lift makes medieval the majority claim and the drawing
comes out 85% industrial. Nothing is broken — `compose_fitted` samples forms with
probability proportional to weight and then sizes them by weight, but how many
cuttable parts a work happens to have is not controlled, so area share is
stochastic and only loosely tracks the lift.

**The honest reading: the assemblage is evidence at Site A, not a guarantee
everywhere.** If the correspondence needs to hold across a set of drawings, it
has to be measured per drawing and selected on, which this now makes possible.

## 4. Why *here* and not fifty feet away

This is the part the model answers and a hand could not.

Of the **14 clash points** the tile selected independently, **12 carry the same
medieval × industrial condition**, and they sit **35 to 151 ft from Site A**,
median 107 ft. Five of them fall inside the 200 ft plot.

They are the orange dots in `site_a_location.png`. **The condition is not a
point, it is a region**, and Site A is its centre of mass. Anywhere inside that
cluster is defensible ground; the coordinate chosen is simply the strongest
reading of it, and it ranks 3rd of 15 on clash score among the tile's own
points, which is a useful check that it is not a freak maximum.

The wider tile is dominated by medieval (17), renaissance (15), Classical (13)
and baroque (13) weight, with industrial at 10. So the industrial claim is a
minority everywhere in this window *except* around this coordinate — which is
what makes it a collision rather than a background.

## 5. Why 200 ft

Not chosen for the tile. It is set by the step to the intervention model, from
`EXHIBITION_SITES.md` §4:

- 200 ft at 1/64" = 1' is a **3.1 in** plot here
- the same 200 ft at 1/8" = 1' is a **25 in** section model
- an **8× step**, which reads as a deliberate zoom rather than an arbitrary one

At 1/128" the plot would be 1.6 in and stop reading as ground held in reserve.

## 6. What is standing on it now

Nothing has been removed. In this variant the site is **located, not excavated**:

- **9 volumes** stand inside the 3.1 in circle
- **11 more** cross its edge

In the earlier build those 9 were deleted by a boolean cylinder and the 11 were
sliced through. That was not wrong — a cleared plot is what "held in reserve"
means, and the cut faces on the 11 read as the party walls an intervention would
attach to. But it decided the question in geometry, permanently, before the
question had been asked. Locating it instead leaves both readings open:

- **a cleared plot** — subtract the cylinder, the 9 go, the 11 become the edge
  condition
- **adaptive reuse** — keep some of the 9 as existing fabric the intervention
  attaches to, which is the second option the brief offered and the one the
  geometry now actually supports

`tile_model.py --no-void-cut` produces this version. Dropping the flag restores
the cut, at any centre and diameter, in about two minutes.

---

## The one thing this argument does not establish

Site A is the *strongest* condition. It is not the *most contested* one. Site B —
medieval × High Tech — is the widest opposition in the corpus at 3.28 and has
constituency agreement **below chance** at 0.71×, meaning the models that find it
cannot agree on a single constituent. Read against each other, A is what the
semantic encoders agree about and B is what only the compression protocols find.

If the intervention is meant to be about disagreement rather than about
intensity, the site should be B, and it is not in this tile's window at all.
