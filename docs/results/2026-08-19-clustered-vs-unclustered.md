---
title: Clustered vs Unclustered Seeds, 17 DRAC22 Images
description: The first cross-validated results, the level-set contribution on the 17-image domain, and the design deviations from the original study.
kind: measurement-record
status: historical — 17-image domain, level-set pipeline
dataset: DRAC22 (17)
updated: 2026-08-19
---

# Clustered vs Unclustered Seeds — 17 DRAC22 Images

> **Record.** Moved verbatim from `CNN+GVF/README.md` on 2026-09-25. Every number here is
> the **17-image DRAC22 domain** under the level-set pipeline; not comparable with the
> 41- or 68-image figures. The code that produced them is at tag `archive/pre-cleanup`
> (runner names and paths below are as they were then). Current state:
> [../SPEC.md](../SPEC.md).

## Two point sets, two units of evaluation

`DRAC22_RNV_Localization_Points_Sorted.csv` holds **clustered centres** — one DBSCAN
representative per lesion, 35 over the 17 ground-truth images.
`DRAC_RNV_Points_Unclustered.csv` holds the **raw detections** behind them: 184 on the same
images, 27 of which land outside every lesion.

They demand different scoring, and `cfg.EvalMode` selects it:

- **`"segment"`** — one row per seed, against `SegmentFilledGT/<stem>-<x>-<y>.png`. Valid only
  for the clustered centres, since those files are keyed by their coordinates.
- **`"image"`** — one row per image, union mask against the whole `FilledGT/<stem>.png`.
  Required for the unclustered points: they have no per-point ground truth, and several land
  on one lesion (62 on `1115.png` alone), so per-segment scoring would count one lesion once
  per detection and weight lesions by how many detections they attracted.

The two units are **not comparable**. Whole-image scoring also charges for lesions missed
entirely, which per-segment scoring structurally cannot see — on the same predictions, recall
falls from 0.681 to 0.527 purely because 13 ground-truth components have no clustered seed.
To compare two networks, run both under the same mode.

## Measured results

### Headline: 5-fold cross-validation, both arms, paired

Every image scored out-of-sample by the fold that held it out. Both arms use the **same
deterministic area-stratified folds** (`cnngvf.cvFolds`), so the comparison is paired
image-by-image. Integrity verified for both: 17 rows, 17 unique, no fold scored an image it
trained on.

| 5-fold CV (17 images) | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| clustered centres | 0.3299 | 0.3215 | 0.4576 | **0.7965** | 0.4079 |
| **unclustered detections** | **0.4163** | **0.4632** | **0.5640** | 0.7210 | **0.5698** |

**Unclustered training seeds improve segmentation by +0.086 mean IoU (+0.123 median).**

| | |
|---|---|
| improved on | **13 of 17** images |
| bootstrap 95% CI on mean delta | **[+0.019, +0.154]** — excludes zero |
| exact two-sided sign test | p = 0.049 |
| per-fold delta | +0.072 / +0.114 / +0.194 / +0.017 / +0.030 — **positive in all 5** |

The mechanism is coverage: the detector fires many times on a large lesion, so seed-centred
patches tile it, where a single cluster centre can only see one 512 px window of it. The cost
is precision (0.797 → 0.721) bought with recall (0.408 → 0.570).

**Honest limits.** n = 17, so p = 0.049 is marginal and the CI is wide — this establishes the
direction confidently and the magnitude only loosely. Four images regress, worst `573.png`
(−0.200) and `641.png` (−0.123). The single-split estimate of the same effect was +0.109
against the cross-validated +0.086, so that protocol overstated the gain by roughly a quarter
while getting the sign and rough size right.

Reproduce with `run_cv('unclustered')` and `run_cv('clustered')` (~8 h and ~10 h).

**The single-split numbers below are optimistic and should not be quoted.** They were
inflated twice over:

| estimate | value | what is wrong with it |
|---|---|---|
| single-split, all 17 images | 0.5430 | 13 of 17 images trained the model |
| single-split, 4 "validation" images | 0.5591 | those same 4 images selected the checkpoint |
| **5-fold CV** | **0.4163** | — |

The second inflation is the subtle one. `OutputNetwork="best-validation"` picks the checkpoint
that scores best on `cfg.Split.Val`, and those same images were then reported as held-out —
selection on the test set. Scored honestly under CV, those four images give 0.4697 rather than
0.5591, so checkpoint selection alone was worth **+0.089**. `461.png` illustrates it: 0.731 as
a single-split validation image, 0.307 under cross-validation.

**Consequence:** the "+0.068 out-of-sample gain" attributed to balanced sampling was measured
on that contaminated 4-image set, so its magnitude is not trustworthy. The direction is
supported by the mechanism (39% of the epoch going to one image), but the size remains
unverified — the cross-validated runs both used balanced sampling, so they establish the
unclustered-vs-clustered effect, not the balancing effect. Isolating that needs its own CV
pair.

Note the CV figures cannot be compared with any single-split figure — different protocols. The
clustered-vs-unclustered comparison above is valid because *both* arms were cross-validated on
identical folds.

### Clustered vs unclustered training seeds (whole-image scoring, single split)

Identical evaluation, identical GT; only the localization points differ. **Both rows are
single-split and so ~13/17 in-sample — read the direction, not the values.** Neither is
comparable with the cross-validated 0.4163 above.

| | mean IoU | median IoU | mean F1 | precision | recall |
|---|---|---|---|---|---|
| clustered centres | 0.4226 | 0.5030 | 0.5587 | 0.7770 | 0.5270 |
| **unclustered detections** | **0.5315** | **0.5896** | **0.6698** | 0.6319 | **0.8293** |

**+0.109 mean IoU, +0.087 median, improved on 12 of 17 images** — under a protocol that
flatters both arms equally, so the direction is informative even though the values are not.

The gain is concentrated exactly where the clustered pipeline was structurally broken — on
lesions far larger than one 512 px patch, which a single centre could never cover:

| image | GT px | clustered | unclustered | recall |
|---|---|---|---|---|
| 1115 | 239 227 | 0.044 | **0.811** | 0.044 → 0.872 |
| 461 | 77 384 | 0.206 | **0.791** | 0.207 → 0.928 |
| 1236 | 64 902 | 0.149 | 0.604 | 0.155 → 0.666 |
| 500 | 13 544 | 0.263 | 0.487 | 0.405 → 0.774 |
| 795 | 100 656 | 0.061 | 0.259 | 0.061 → 0.271 |

`1115.png` carries 62 unclustered detections across its lesion, so the network finally sees
it whole, as a union of overlapping patches. This is direct evidence that **coverage of the
probability map was the bottleneck** — the same conclusion the GVF diagnostics below reach
from the other direction.

The cost is precision, 0.777 → 0.632, and five images regress. The two worst are pure
over-segmentation: `576.png` predicts 101 731 px against 14 216 GT at recall 0.937, and
`573.png` 9 757 px against 2 016 at recall 0.996 — the lesion is found and then massively
overshot. Alongside that, 27 unclustered detections fall outside every lesion; they are
dropped during training but still generate patches at inference, where they can only add
false positives. Filtering those at inference is untried and is the cheapest remaining win.

### Per-segment scores (clustered centres only)

| configuration | mean IoU | precision | recall |
|---|---|---|---|
| patch 256, whole-mask labels, 60 epochs | **0.534** | 0.816 | 0.646 |
| patch 512, per-seed labels, 30 epochs | 0.493 | 0.719 | 0.681 |
| threshold `P > 0.6` alone, no GVF, no level set | 0.496 | 0.728 | 0.668 |

### Does the active contour stage earn its place? Yes — corrected

An earlier version of this README concluded it did not. **That conclusion was wrong**, for a
methodological reason worth recording: the two arms of that comparison were not built the same
way — the pipeline arm applied `cnngvf.seedMasks`, the threshold arm only `bwareaopen` — so
seed-driven component selection, not the level set, carried the difference. It was also
measured on the weakest of the three models.

Re-run on the current model with **identical post-processing in both arms**
(`imclose → imfill → seedMasks`), so the only variable is what produced the binary mask:

| arm | mean IoU | precision | recall |
|---|---|---|---|
| threshold `P > 0.60` | 0.5239 | 0.7393 | 0.6808 |
| **GVF + DRLSE** | **0.5403** | 0.7413 | 0.7054 |

**+0.017 overall, better on 13 of 17 images**, and on the four out-of-sample images
0.5595 → 0.5945, **+0.035**. The four regressions are small (worst −0.034). The gain is real
but modest, and it is mostly recall — the level set grows the contour off the thresholded core
into boundary the threshold clips.

Caveat: 13 of those 17 images are training images, so the out-of-sample figure rests on four
images. The cross-validation protocol below is what will settle the size of the effect.

The mechanism described below still holds and explains why the gain is modest rather than
large.

### Why the gain is modest: the level set converges to an iso-contour of P

The level set moves 583–1171 px per lesion and converges by iteration 100. It also genuinely
*forms* the curve: without it there is no contour, only a pixel mask.

The reason it improves the mask by 0.017 rather than transforming it is *where* it converges.
With `f = |∇P|`, the GVF field points at the ridge of
`|∇P|`, and that ridge is the steepest part of `P` — the `P ≈ 0.5` iso-contour. So the
evolution drags the boundary from wherever `tau` put it back towards the `P ≈ 0.5` level.
**The level set is largely an iso-contour finder for `P`**, which bounds how far it can
depart from a threshold of `P`. Its contribution is regularization and a modest recall gain,
not independent localization.

Three attempts to enlarge the effect all failed, measured on the earlier patch-512 model:

| hypothesis | test | result |
|---|---|---|
| the force field is `P`-derived, so add image information | `Force.source="blend"`, w=0.7/0.5/0.3 | 0.4946 / 0.4958 / 0.4974 — no change |
| IoU is blind to boundary quality, `bfscore` will show more | `bfscore` at matched `tau` | 0.641 vs 0.657 for raw threshold |
| the curvature term over-smooths the fronds | `lambda` 5 → 0.2 | raggedness 4.75 → 4.81, IoU flat |

So the ~0.017 gain appears to be what this force field can deliver; enlarging it needs a force
field carrying information `P` does not already contain, and the raw intensity gradient has
been tested and is too speckled to supply it.

The raggedness index (perimeter / √area) shows the level set produces 4.75 against a ground
truth of 6.31 — it *is* over-smoothing RNV's fern-like boundary, which
[../docs/CLAUDE.md](../THEORY.md) warns about for the snake's β. But that smoothing
survives a 25× cut in `lambda`, so it comes from the distance-regularized evolution itself,
not the curvature weight.

The dominant errors are also the wrong shape for contour refinement to fix. On the clustered
pipeline the worst cases were detection failures, not boundary-placement errors — `1115` at
recall 0.044, `795` at 0.061, `461` at 0.207 — and no active contour recovers a lesion the
CNN never responded to. **The bottleneck is the probability map, not the refinement**, which
is why the unclustered points bought +0.11 IoU where every post-processing lever bought
~0.01–0.02.

### Caveats on these numbers

- The GVF/DRLSE comparison was measured on the patch-512 model, which is the weaker of the
  two. The *mechanism* above is model-independent — it holds for any `P` — but the exact
  margins would move on a better network.
- The 256 → 512 comparison changed **four** variables at once (patch size, label source,
  epochs 60→30, miniBatch 8→2), so the regression cannot be attributed to any one of them.
  Per-seed labels dropped the foreground fraction from ~0.147 to ~0.004, which with
  generalized Dice and half the epochs is the leading suspect, but that is untested. A proper
  ablation varies one at a time.
- The clustered → unclustered comparison also moves more than one thing: 30 training seeds
  become 137, `perSeed` drops 16 → 4 to hold the epoch size, and the foreground fraction
  jumps from 0.004 to 0.436. The *direction* is unambiguous and the mechanism (patch coverage
  of large lesions) is confirmed per-image, but the magnitude is not attributable to seed
  count alone.
- The unclustered training seeds are drawn area-weighted in effect — `1115.png` supplies 62 of
  185 points, `517.png` supplies 2 — while whole-image scoring weights every image equally.
- The 256-patch model that scored best per-segment (0.534) was overwritten before this was
  understood. `cnngvf.trainNet` now archives any existing network with its timestamp before
  saving, but recovering that one needs a retrain.
- The localization CSVs have been revised repeatedly during development (`StartRow`, the
  clustered row count, and the unclustered points for `573`/`664` have all changed). Any
  number quoted here is tied to the CSV in place when it was measured; `run_eval` exists
  partly so a stored run can be re-scored cheaply after such a change.

## Deviations from the spec, and why

- **No `f = P` force source.** The spec lists it. The GVF of `P` points at lesion *cores*,
  not *boundaries*, so under the advection sign in `cnngvf.drlse` it collapses the front to a
  point. `buildForce` offers only boundary maps (`"probGrad"`, `"blend"`).
- **Balloon term `alfa = 0`.** The spec suggests ±1.5, which assumes a contour initialized far
  from the object. `phi0` here comes from the probability map and already sits on the boundary,
  and being binary rather than a distance function it gives the Dirac no narrow band — so a
  balloon pushes on the whole image and floods the background. Measured on the synthetic scene
  in `test_cnngvf`: `alfa` 0 → IoU 0.965 stable to 600 iterations, −0.2 → 0.735, −0.5 → 0.130.
- **Edge stopper is gain-corrected.** The classic `g = 1/(1+|∇(G*I)|²)` assumes a 0–255 image.
  On a probability map it yields `g ≈ 0.999` everywhere, i.e. no stopper at all.
  `cnngvf.drlse` normalises the gradient by its own peak and applies `cfg.Drlse.edgeGain`.
- **Patch is 512, not the spec's 256** (still inside its 128–512 range). Measured on this
  dataset, 13 of 35 ground-truth segments need a seed-centred window wider than 256 px, and
  the CNN cannot predict what the patch does not contain. At 256 every clipped segment scored
  badly (`1115` needs 391×1501 → IoU 0.019; `795` needs 501×1815 → 0.074) while every segment
  that fitted scored 0.50–0.88 — precision 0.82 against recall 0.65. At 512, 32 of 35 are
  fully covered and mean coverage is 97.5%. `461`, `535`, `795` and `1115` need 525–1815 px
  and stay clipped; only full-image inference would reach them.
- **Training labels are the seed's own segment**, not the whole `FilledGT` cropped to the
  patch. This makes the training task the one the evaluation scores — "segment the lesion at
  the heat-map" — and gives the seed heat-map channel a job. With whole-image labels a patch
  holding two lesions marks both as foreground and the neighbour leaks into this seed's
  component.
- **No cached patch file.** The spec's `extractPatches` writes patches to disk. Cropping on
  demand from the 17 in-memory images is less code, avoids a multi-GB `.mat`, and
  re-randomises jitter and flips every epoch — augmentation a fixed cache cannot give.
- **No `parfor`.** Parallel Computing Toolbox is used through `gpuArray` (level set),
  `canUseGPU`, and GPU training. Several workers contending for one laptop GPU is slower than
  running serially, so there is no worker-pool branch.

## Known limits and next steps

- 17 ground-truth images. That is thin for a U-Net; `../docs/spec.md` suggests pretraining on
  ROSE or OCTA-500.
- **Clustered centres under-cover the ground truth.** `458.png` has 7 centres against 12
  components; 13 components have no centre at all. Regions no seed reaches stay at `P = 0` by
  design. The unclustered detections largely fix this and are the recommended training input.
- **A seedless P was tried and not adopted** (branch `seedless-prob`, `docs/SEEDLESS.md`).
  The `DRACTUH_seedless` arm drops the heat-map, tiles the whole frame (3×3 × 512 px, stride
  256) and leaves seeds only for claiming, through the same reduced + relax 0.20 pipeline. The
  fold-1 pilot (14 images) failed both pre-registered gates: IoU 0.4124 → **0.3273** (gate
  ≥ 0.39) and ground-truth coverage 0.462 → **0.329** (gate: above seeded). Recall collapsed
  (0.71 → 0.38) while precision rose, and validation loss sat at 0.65–0.73 for 19 epochs
  against the seeded network's 0.437. The likely cause is `1 − generalizedDice` on
  all-background crops, where any foreground scores Dice ≈ 0; the loss was held fixed by
  design, so fix it before revisiting. Folds 2–5 were not run.
- **Routing by device does not help.** The pilot showed TUH better seedless (0.443 → 0.491,
  n = 5) and DRAC22 worse, so "TUH → seedless, DRAC22 → seeded" was checked with both fold-1
  networks over all 68 images (batch-style, mostly in-sample): routed **0.4578** vs all-seeded
  0.4575. On the images the networks trained on the split reverses (DRAC22 seedless 0.476 vs
  0.437, TUH 0.507 vs 0.516) and seedless wins 13 of 27 TUH images — the pilot's TUH edge was
  noise. Seedless is also the most overfit: 0.488 on trained-on images, 0.327 held out.
- **27 unclustered detections lie outside every lesion.** They are dropped in training but
  still generate patches at inference, where they can only add false positives. Filtering them
  at inference is the cheapest available precision win and is untried.
- **The GVF/DRLSE stage is currently inert.** It forms the contour but does not improve on
  thresholding `P`. Either accept that as a finding, or give the force field information `P`
  does not already carry — a vesselness- or Frangi-driven field would be the next candidate,
  since the raw intensity gradient has been tested and is too speckled.
- The dominant remaining errors are detection failures and over-segmentation, both properties
  of `P`. Effort spent on the probability map will pay more than effort spent on the contour.
- `573.png`'s ground truth is a data-collection error and should be regenerated.
