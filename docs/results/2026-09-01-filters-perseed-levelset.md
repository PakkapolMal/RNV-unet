---
title: Filters, perSeed and the Level Set — Records from the Specification
description: Measurements that used to live in the specification — the filter ablation, the 41-image DRAC22 record, the perSeed underfit, and the removed GVF + DRLSE stages.
kind: measurement-record
status: historical — level-set pipeline
dataset: DRAC22 (17 / 41) and DRAC22 + TUH (68, single split)
updated: 2026-09-17
---

# Filters, perSeed and the Level Set

> **Record.** Moved verbatim from `docs/UNCLUSTERED.md` (now [../SPEC.md](../SPEC.md)) on
> 2026-09-25, when the GVF + DRLSE stages were deleted. State the domain with every number:
> 17-, 41- and 68-image figures are different questions. The code is at tag
> `archive/pre-cleanup`.

## Result

> **Every number in this section was measured on DRAC22 alone, before TUH
> existed.** They are the frozen record of the `unclustered` arm. No
> cross-validated combined-domain figure exists yet —
> `run_cv("DRACTUH_unclustered")` produces the first one.

### The perSeed=1 underfit

The first combined-domain training run used `Patch.perSeed = 1`, chosen purely to hold
runtime down and explicitly unmeasured. **It underfit, and the evidence is worth keeping
because it is the kind that is easy to misread as a real result.**

| | |
|---|---|
| validation loss, epochs 1 → 30 | 0.421 → 0.389, **wobble ±0.03** |
| single-split IoU, 68 images | 0.3852 (F1 0.533, precision 0.580, recall 0.644) |
| ... DRAC22 subset | **0.379** |
| ... TUH subset | 0.395 |
| lesions > 100k px (n=8) | IoU 0.231, precision **0.800**, recall **0.247** |

Three signals, all pointing the same way:

1. **Validation loss never descended.** Total improvement over 29 epochs was 0.032, the
   same size as the epoch-to-epoch noise. With only 6 validation images the selection
   signal is smaller than the wobble, so `best-validation` picking epoch 15 may be luck.
2. **An in-sample score below another model's out-of-sample score.** 62 of the 68 images
   trained this model, so 0.385 is ~91% in-sample and should be *flattered*. Its DRAC22
   subset scored 0.379 against the DRAC22-only arm's cross-validated **0.4603**. That
   direction is not achievable by noise.
3. **High precision, collapsed recall on large lesions.** `nv (21).png` has 113 seeds
   tiling a 432k px lesion and recalled 12% at precision 0.90 — the network under-responded
   rather than running out of patch coverage, which is what distinguishes underfitting from
   the known patch-size limit.

`perSeed` is now 4, restoring the per-seed exposure the other arms have.

### Confirmed: perSeed 1 → 4 is worth +0.141 IoU

Retrained identically apart from `perSeed`. **Both arms single-split on the same 68 images,
so the protocol flatters both equally** — the direction and the mechanism are trustworthy,
the magnitude is not.

| | n | IoU (1 → 4) | Δ | precision | recall |
|---|---|---|---|---|---|
| **all** | 68 | 0.3852 → **0.5262** | **+0.141** | 0.580 → 0.710 | 0.644 → 0.691 |
| DRAC22 | 41 | 0.3786 → 0.5218 | +0.143 | 0.524 → 0.699 | 0.656 → 0.664 |
| TUH | 27 | 0.3952 → 0.5329 | +0.138 | 0.665 → 0.726 | 0.625 → 0.733 |
| **GT > 100k px** | 8 | 0.2313 → **0.4403** | **+0.209** | 0.800 → 0.875 | **0.247 → 0.482** |
| GT ≤ 100k px | 60 | 0.4057 → 0.5377 | +0.132 | 0.550 → 0.688 | 0.697 → 0.719 |

Improved on **60 of 68 images**; the 8 regressions are small (worst −0.06). Precision *and*
recall both rose, which is unusual and rules out a simple threshold shift.

Validation loss confirms it independently: best **0.324** against `perSeed = 1`'s **0.389**,
with a visible downward trend instead of a flat line.

**This revises a standing conclusion.** Large lesions were documented as failing because a
512 px patch cannot cover them. Recall on lesions over 100k px nearly doubled (0.247 → 0.482)
with *no change to patch size* — so a large part of that failure was undertraining, not
geometry. The patch limit is real but was masking a bigger, cheaper problem.

**None of 0.385 / 0.526 is quotable** — single-split, ~91% in-sample. The paired delta is
the finding; `run_cv` is what produces a reportable number. The `perSeed = 1` record is in
`result/diagnostic_perSeed1/`.

### Frozen DRAC22-only record

### DRAC22-only configuration, full dataset *(superseded scope)*

Unclustered detections, balanced sampling, **no median filter, no vesselness** (2 input
channels), 20 epochs. 5-fold cross-validated, every image scored out-of-sample:

| 5-fold CV, 39 seeded images of 41 | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| **unclustered, filters off** | **0.4603** | 0.4511 | 0.6040 | 0.6434 | 0.6349 |

Per-fold 0.4215 / 0.3468 / 0.4947 / 0.4569 / 0.5673 — sd 0.082, a wider spread than the earlier
17-image run, so read fold-level differences as noise. Integrity verified: 39 rows, 39 unique,
every fold scored exactly its held-out images. Runtime 3.9 h.

The 22 newly added images are no harder than the original 17 (0.4514 vs 0.4717), so the
enlarged dataset did not shift the distribution.

**Against the previous cross-validated figure, on the 17 images common to both:** 0.4163 →
0.4717, **+0.055**. Three things changed together — filters removed, 2.4× more training data,
30 → 20 epochs — so this is not attributable to any one. The extra data is the likely driver:
the filter ablation measured ≈0, and fewer epochs should if anything hurt.

**There is no paired baseline for this dataset.** `run_cv('clustered')` (~12 h) would restore
it.

### The paired comparison (17 images, superseded)

Both arms on identical folds, 3 channels, 30 epochs. Preserved per-image in
`result/cv_17images/`. **This is still the only paired evidence that unclustered seeds help.**

| 5-fold CV (17 images) | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| clustered centres (baseline) | 0.3299 | 0.3215 | 0.4576 | 0.7965 | 0.4079 |
| **unclustered detections** | **0.4163** | **0.4632** | **0.5640** | 0.7210 | **0.5698** |

**+0.086 mean IoU (+0.123 median)**, improved on 13 of 17 images, bootstrap 95% CI
**[+0.019, +0.154]**, exact sign test p = 0.049, and positive in **all five folds**
(+0.072 / +0.114 / +0.194 / +0.017 / +0.030).

`n = 17`, so the direction is established confidently and the magnitude only loosely.

## Stages 1–2: the input filters

### 1. Denoising — `02_denoised.png`  *(currently OFF)*

**Does:** median filter, `[3 3]`, on the grayscale image (`rnv.loadImage`). Disabled by
`cfg.Denoise = []`, which `run_train_unclustered` and `run_batch_unclustered` now set.

**The argument for it** is that OCTA speckle is multiplicative and pixel-local, exactly what a
median filter removes while leaving edges intact, and everything downstream differentiates this
image. **The measurement did not support it** — see the ablation below.

### 2. Vesselness channel — `03_vesselness.png`  *(currently OFF)*

**Does:** multiscale Frangi response, `fibermetric(I, [2 4 6 8], ObjectPolarity="bright")`, as a
second input channel. Disabled by `cfg.Vessel.enabled = false`; the network then takes 2 inputs
(intensity + heat-map) rather than 3.

**The argument for it** is that RNV *is* a vessel structure and a vesselness response suppresses
the background texture a raw gradient latches onto, with deliberately small scales because
[CLAUDE.md](../THEORY.md) notes large sigmas blur thin vessels away. **The measurement did not
support it either.**

### Ablation: both filters removed

Same seeds, same scoring, compared on the **17 images both runs scored**:

| | mean IoU | median |
|---|---|---|
| 3-channel: median filter + vesselness | 0.5430 | 0.5829 |
| 2-channel: neither | 0.5341 | **0.5833** |
| delta | −0.0089 | +0.0004 |

Better on 8 of 17 images — indistinguishable. **Neither filter is earning its place**, so both
are off by default in the unclustered arm. The CNN evidently extracts what it needs from raw
intensity, which is consistent with everything else measured here: the probability map is the
bottleneck, and hand-designed front-ends have not moved it.

**Confounded, so do not over-read the −0.009.** The filter-free run also trained on 41
ground-truth images instead of 17. If the extra data helped, it may be masking a real cost from
dropping the filters. Separating them needs a 2-channel run on the original 17, or a 3-channel
run on all 41.

Both are still one flag away (`cfg.Denoise`, `cfg.Vessel.enabled`), and the channel count is
derived throughout, so re-enabling either needs only a retrain.

## The removed level-set stages (5–9)

Deleted on 2026-09-25: removing them measured −0.0017 IoU on 68 images, not significant
([2026-09-16-reduce.md](2026-09-16-reduce.md) §6). Kept here because the measured fixes
(edge-stopper gain, balloon off, advection sign) are the reusable knowledge.

### 5. Force field `f = |∇P|` — `06_force.png`

**Does:** gradient magnitude of a lightly blurred `P`, normalised to `[0,1]`.

**Matters because** it defines *what the contour is attracted to*. The GVF of a ridge points at
that ridge from both sides, and the ridge of `|∇P|` is the CNN's boundary — so this is the
choice that makes the level set converge to the boundary rather than anywhere else.

**`f = P` is deliberately not offered**, though [spec.md](../briefs/design-study.md) lists it: the GVF of `P`
points at lesion *cores*, so under the advection sign in `drlse` it collapses the front to a
point. Both available sources (`"probGrad"`, `"blend"`) are boundary maps.

### 6. GVF diffusion → `(u,v)` — `07_gvf_magnitude.png`, `08_gvf_quiver.png`

**Does:** Xu & Prince diffusion (`alpha 0.2, mu 0.2, 80` iterations) via the compiled mex in
`lib/Fast-Gradient-Vector-Flow`, then scaled so the peak magnitude is 1.

**Matters because** a raw edge force exists only in a thin band around the edge; diffusion
extends it across the image, which is what gives capture range and lets a contour reach a
boundary it did not start on. It is also what carries force into concavities — essential for
frond-like RNV, whose boundary is fern-like rather than convex.

Scaling is **global, not per-vector**: the field's relative weakness far from the lesion is
meaningful to the level set, and per-vector normalisation would erase it.

### 7. Edge stopper `g` — `09_edge_stopper.png`

**Does:** `g = 1/(1 + (edgeGain · |∇(G*P)| / max)²)` — small where `P` has a strong boundary,
so the front slows there.

**Matters because** it is the brake. Without a working brake the balloon and curvature terms
push the contour through boundaries.

**This one was a measured bug fix.** The classic `g = 1/(1+|∇(G*I)|²)` assumes a 0–255 image.
On a probability map in `[0,1]` it returns **g ≈ 0.999 everywhere** — no brake at all, and the
front floods the background. Normalising the gradient by its own peak and applying
`edgeGain = 8` restores the intended range and makes it independent of how sharp this
particular network's output happens to be.

### 8. Initialisation `phi0` — `10_phi0_init.png`

**Does:** `phi0 = c0·(1 − 2·BW0)` where `BW0 = P > tau`, restricted to seed-claimed components.
Negative inside, positive outside. A seed whose neighbourhood never crosses `tau` gets a small
disk (`seedRadius = 15`) instead of nothing.

**Matters because** the level set only finds boundaries it starts near, and this is what makes
the start good — it is already roughly correct, which is why the balloon term is off (below).
The disk fallback matters for a different reason: with 17 training images the network *will*
miss, and without it a missed lesion is silently dropped rather than given a chance to grow.

### 9. DRLSE level set — `11_levelset_raw.png`

**Does:** Li et al. (2010) distance-regularized evolution, with the GVF field substituted for
`∇g`:

```
φ ← φ + dt·[ muDR·distReg(φ) + λ·δ(φ)·(g·curv − (u·φx + v·φy)) + alfa·g·δ(φ) ]
```

**Matters** for two things a parametric snake cannot do. It **splits and merges natively** —
one `φ` holds every lesion in the image, and `458.png`'s ground truth has 12 components against
7 clustered seeds. And the distance regularization removes re-initialisation, which permits the
large time step.

Measured contribution: **+0.017 IoU** over thresholding the same `P` with identical
post-processing, better on 13 of 17 images. Modest, and the reason is structural — see
[the level-set section](#what-the-level-set-contributes).

Two parameters are counter-intuitive and both were measured:

- **`alfa = 0`** (no balloon), against [spec.md](../briefs/design-study.md)'s ±1.5. That suggestion assumes a
  contour starting far from the object. Here `phi0` is already on the boundary *and* binary
  rather than a distance function, so the Dirac has no narrow band and a balloon pushes on the
  whole image at once. Measured: `alfa` 0 → IoU 0.965, −0.2 → 0.735, −0.5 → 0.130.
- **The advection sign.** `∇g` points *away* from a boundary; a GVF field points *toward* it.
  The substituted term is therefore `−(u,v)·∇φ`. With the opposite sign the front runs away
  from every edge it should stop on.

### Parameters

| parameter | value | why |
|---|---|---|
| `Force.source` | `"probGrad"`, `f = |∇P|` | the GVF of a ridge points at that ridge from both sides. `f = P` is **not** offered: its field points at lesion *cores*, which collapses the front under the advection sign. |
| `Gvf.alpha / mu / iter` | 0.2 / 0.2 / 80 | Xu & Prince; via the compiled mex in `lib/Fast-Gradient-Vector-Flow` |
| `Drlse.tau` | 0.5 | threshold seeding `phi0` |
| `Drlse.lambda` | 5 | edge/length weight |
| `Drlse.edgeGain` | **8** | the classic `g = 1/(1+|∇(G*I)|²)` assumes a 0–255 image; on a probability map it gives `g ≈ 0.999` everywhere — no stopper at all. The gradient is normalised by its own peak first. |
| `Drlse.alfa` | **0** | spec suggests ±1.5, for a contour starting far from the object. Here `phi0` already sits on the boundary and is binary, so the Dirac has no narrow band and a balloon floods the background. Measured: alfa 0 → 0.965, −0.2 → 0.735, −0.5 → 0.130. |
| `Drlse.iter` | 300 | converges by ~100 |

**Sign convention:** `∇g` points *away* from a boundary, a GVF field points *toward* it, so the
substituted term is `-(u,v)·∇φ` — standard advection. With the opposite sign the front runs
away from every edge it should stop on.

## What the level set contributes

Measured with **identical post-processing in both arms**, so the only variable is what
produced the mask:

| arm | mean IoU | precision | recall |
|---|---|---|---|
| threshold `P > 0.60` | 0.5239 | 0.7393 | 0.6808 |
| **GVF + DRLSE** | **0.5403** | 0.7413 | 0.7054 |

**+0.017**, better on 13 of 17 images. Real but modest, and mostly recall — the level set grows
the contour off the thresholded core into boundary the threshold clips.

It is modest because the GVF field points at the `|∇P|` ridge, which *is* the `P ≈ 0.5`
iso-contour: the level set is largely an iso-contour finder for `P`, which bounds how far it
can depart from a threshold of `P`. Three attempts to enlarge the effect all failed — image
blending, `bfscore` (which agreed with IoU), and cutting the curvature weight 25×.

An earlier version of this analysis reported that the level set contributed *nothing*. That
was wrong: the two arms had not been built the same way (one applied `seedMasks`, the other did
not), and it was measured on the weakest model.

## Known limits as recorded on 2026-09-17

*Updated 2026-09-17 from [REFINE_RESULTS.md](2026-09-17-refine.md). The bullets after this block are
the earlier record and are kept as written; where they are superseded, this block says so.*

- **The combined-domain number now exists.** Reduced pipeline + relax, floor cross-fitted:
  **0.5061 IoU** over 68 images (0.5097 at floor 0.20). Level set on the same `P`: 0.4708.
  The claiming rule is now `"boundary"` for this arm, which resolves the centroid bullet below.
- **Inference-only post-processing is exhausted on this `P`, measured.** A claim-distance cap
  changed nothing, a peak-relative relax guard cost −0.005, and d4 test-time augmentation gave
  +0.003 with a CI crossing −0.009. None was adopted.
- **The large-lesion gap is not patch coverage.** 98% of the largest-quintile ground truth lies
  inside a 512 px patch, and 76% of its missed pixels have P < 0.20. The network sees these
  lesions and under-responds. This answers the "patch size" and "tiled inference" bullets below:
  the lever is training, not geometry.
- **Stray localization points are confident, not weak.** 29 of 32 fire with P ≈ 1. Only
  isolation separates them (AUC 0.93), and isolation also marks true lesions with a single
  detection, so no `P`-based seed filter fits the budget. The ~+0.03 oracle headroom belongs to
  the localization stage.
- **Tiny lesions are threshold-unstable.** A 0.8% shift in `P` can halve IoU on a ~1 000 px lesion
  by joining a neighbouring blob, so expect large single-image swings in the smallest quintile
  from any change to `P`.
- **Under relax, `tau` 0.55–0.60 scores higher on the surface** (up to 0.519) than the
  pre-registered 0.5. Untested as a result.


- **No combined-domain number exists yet.** `run_cv` on `DRACTUH_unclustered` produces the
  first one. Every figure in this document is DRAC22-only.
- **The TUH acquisition overlay is still in the images.** Straight-line false positives are the
  symptom to watch for; masking before `rgb2gray` is the fix.
- **Cross-validation costs ~18 h** at `perSeed = 4` / 20 epochs (5 h per training run at 30
  epochs). Cutting either `perSeed` or epochs to save time now has a measured cost, so this
  is the price of a reportable number.
- **30 epochs may not be enough.** The `perSeed = 4` curve was still descending at epoch 29
  (best 0.324, and 0.334 / 0.338 / 0.336 through the late 20s). Training longer is the
  cheapest untested lever left — and note it points the *opposite* way from the
  `perSeed = 1` curve, which flatlined by epoch 3 and made 30 look generous.
- **Is `perSeed = 8` better still?** 4 was a restoration of the other arms' setting, not an
  optimum. The 1 → 4 step was worth +0.141, so the curve has not obviously saturated.
- **Whether the remaining large-lesion gap is patch size** is now open again. Recall on
  lesions over 100k px went 0.247 → 0.482 on training alone; the rest may or may not be
  geometry. Test before investing in tiled inference.
- **Precision is the weak side** (0.721 vs the clustered baseline's 0.797); the failure mode is
  over-segmentation.
- **32 detections are genuine localization errors.** Rejected from training, but they still
  generate patches at inference, where they can only add false positives. A ground-truth
  filter there would be leakage; a `P`-based one is untried, and an *oracle* filter was worth
  only +0.015, so the ceiling is low.
- **`299.png` and `445.png` have ground truth but no usable seed** and will score ≈ 0. That is
  a localization-stage gap charged to the segmenter. If the headline is meant to measure
  segmentation rather than the pair, they need re-localizing, not excluding.
- **`cnngvf.seedMasks` still claims by nearest *centroid*, unlimited distance**, while training
  now labels by nearest *boundary* within 20 px. Two rules for "which lesion does this seed
  belong to". Centroid distance misassigns seeds around large concave lesions — the bug
  `RNV-Claude/main` was fixed for — so aligning them is worth doing.
- **Post-processing is exhausted.** The pipeline already beats an oracle that picks the best
  threshold per image (0.5315 vs 0.5068). Effort belongs in the probability map.
- **The balanced-sampling gain is unverified.** Both DRAC22 CV arms used it, so they measure the
  unclustered effect, not the balancing effect.
- **Lesions exceed even a 512 px window** — four in DRAC22, and TUH's `nv (21)` at 41% of the
  frame is far worse. Only full-image tiled inference reaches them.
- **68 ground-truth images** is still thin for a U-Net; [spec.md](../briefs/design-study.md) suggests pretraining
  on ROSE or OCTA-500.
