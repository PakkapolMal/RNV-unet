---
title: RNV Segmentation Pipeline — Specification
description: The working specification of the CNN+GVF pipeline — architecture, contracts, parameters, and the measurement behind every non-obvious choice.
kind: specification
status: current
dataset: DRAC22 + TUH (68 images)
updated: 2026-08-31
---

# RNV Segmentation from Unclustered Localization Points — Specification

The working specification: how the pipeline is wired, what it assumes, and the
measurement behind every non-obvious choice. Implementation:
[`../CNN+GVF/`](../CNN+GVF/). Vocabulary: [CONTEXT.md](CONTEXT.md). Theory:
[CLAUDE.md](CLAUDE.md). Original design: [spec.md](spec.md).

> **Filename note.** This file is named for the unclustered detections because
> that was the finding it was created to record. It is now the general
> specification, covering both datasets and every arm.

## The domain

**68 images across two datasets**, pooled. Every image with ground truth is in
scope; `cnngvf.manifest` is the single place that says so.

| | images | points (unclustered) | usable | rejected | GT area range |
|---|---|---|---|---|---|
| **DRAC22** | 41 | 283 | 259 | 24 | 2 016 – 239 227 px |
| **TUH** | 27 | 591 | 583 | 8 | 1 106 – 432 533 px |
| **combined** | **68** | **874** | **842** | **32 (3.7%)** | 1 106 – 432 533 px |

Usable counts are at `Patch.seedTolerance = 20` — see [Seed tolerance](#seed-tolerance)
below. At tolerance 0 (the frozen arms' setting) they are 234 / 558 / 793.

Both are OCTA en-face, 1024×1024, but from **different devices** — DRAC22 as
distributed by the DRAC 2022 challenge, TUH from a ZEISS PLEX Elite. TUH
localization is markedly cleaner (5.6% of detections off-lesion against 17.3%),
and its lesions run larger: `nv (21).png` is 41% of the frame and carries 113 of
its dataset's 591 detections.

Folds are **pooled across datasets**, not stratified by device. The research
question is RNV segmentation, not per-device performance, so a fold is not
restricted to one device. Area-stratified round-robin dealing interleaves them
in practice — every fold currently holds both — and `per_image.csv` carries a
`Dataset` column so a pooled run can be broken out afterwards without re-running.

### Seed tolerance

The localization points come from separate research and contain errors: 81 of 874 fall
outside every ground-truth lesion. **They are not one population.** Measured as true
boundary distance to the nearest lesion, the distribution is sharply bimodal, with
*nothing at all* between 50 and 100 px:

| distance | points | | distance | points |
|---|---|---|---|---|
| ≤ 5 px | 36 | | 50–100 px | **0** ← separator |
| 5–10 px | 8 | | 100–200 px | 7 |
| 10–20 px | 5 | | > 200 px | 21 (max 843) |
| 20–50 px | 4 | | | |

The near group are **boundary near-misses**, not localization errors — a detection 3 px
outside a lesion edge is a quibble about where the annotator drew the line. The far group
are **genuine detector false positives**, confirmed by eye:

- **`299.png`** — all 4 detections 307–726 px away on ordinary capillary background, while
  the real lesion (a clearly visible tuft) went undetected entirely.
- **`419.png`** — 2 detections correctly on the lesion, one at 116 px on normal
  vasculature, one at 602 px on plain background.

In both cases the ground truth is correct and the *detector* is wrong, so dropping the far
points is the right repair. Neither is an under-annotation.

`cfg.Patch.seedTolerance = 20` px labels a seed with the nearest lesion within that
distance, and drops the rest: **49 recovered, 32 rejected**, lifting usable training seeds
from 793 to 842. The exact value does not matter — the empty 50–100 px band means anything
in 20–100 behaves identically — and 20 sits inside the ±32 px crop jitter the network
already sees.

Distance is to the **boundary**, not the centroid. A seed just outside a large concave
lesion can be nearer another blob's centroid than its own; that is the bug
`RNV-Claude/main`'s `selectSeedComponent` was fixed for, and `cnngvf.seedMasks` still uses
the centroid rule at prediction time (see [Known limits](#known-limits-and-next-steps)).

**Rejection uses ground truth, so it applies to training only.** Filtering the points at
*inference* by whether they hit a lesion would be leakage — using the answer to choose the
question — and no such number is reportable. An *oracle* inference filter was measured at
only **+0.015** anyway, so the ceiling is low. A legitimate inference-time filter would
read only the model's own output (drop a seed whose patch never produces `P > tau`); that
is untried.

Two images end up with ground truth but no usable seed — **`299.png` and `445.png`**. They
are still scored, and will score ≈ 0. That is a localization-stage gap being charged to the
segmenter, which is the honest accounting: whole-image scoring is supposed to charge for
lesions missed entirely.

### The TUH acquisition overlay

Every TUH image carries burned-in device graphics: a **magenta vertical and cyan
horizontal crosshair spanning the whole frame**, plus a corner watermark. After
`rgb2gray` these become bright, perfectly straight, high-gradient lines — the
shape a vesselness filter and a `|∇P|` force field are most drawn to — and they
cross the middle of the frame where lesions are.

**They are deliberately left in.** They are removable (colored pixels are
trivially separable in RGB, before the grayscale conversion discards the
information) and that option stays open, but nothing here removes them today.
If a combined run shows straight-line false positives, this is the first
suspect, and the fix is a masking step in `rnv.loadImage` before `rgb2gray`.

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

## The core idea

The localization stage emits raw detections, then DBSCAN clusters them and keeps one
representative per lesion. **Train on the raw detections, not the representatives.**

| | clustered centres | unclustered detections |
|---|---|---|
| points over the 17 GT images | 35 | 185 |
| usable for training (inside a lesion) | 35 | 161 |
| points on `1115.png` | 1 | 62 |

A 512 px patch cannot cover a 239 227 px lesion. One cluster centre sees one window of it; 62
detections tile it. That is the whole mechanism, and it is why the gain is concentrated in
large lesions and shows up as recall (0.408 → 0.570) rather than precision (0.797 → 0.721).

**The detections are not a superset of the centres** — check per image rather than assuming.

## Pipeline

```
image ─► preprocess ─┬─► intensity ──┐
                     └─► vesselness ─┤
                                     ├─► 512² seed-centred patch ─► U-Net ─► P
seed (x,y) ─► Gaussian heat-map ─────┘                                       │
                                                                             ▼
mask ◄─ seedMasks ◄─ imfill ◄─ imclose ◄─ DRLSE level set ◄─ GVF field ◄─ |∇P|
```

Per-seed masks are unioned into one image-level mask. One level set evolves over the whole
image and the seeds decide afterwards which components survive — that is what lets a lesion
split into fronds, which a parametric contour cannot do.

## Stage by stage

Each stage below is written out by [`run_steps.m`](../CNN+GVF/run_steps.m) under the numbered
filename given, so every claim here can be inspected on a real image:
`run_steps("458.png")`.

Measurements are from this dataset. Where a stage's importance is reasoned rather than
measured, it says so.

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
[CLAUDE.md](CLAUDE.md) notes large sigmas blur thin vessels away. **The measurement did not
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

### 3. Seed heat-map — `04_seed_heatmap.png`, `patches/seed_NN.png`

**Does:** a Gaussian bump (`sigmaSeed = 32`) at the seed's position **within the patch**,
appended as the third input channel.

**Matters because** it is the only thing that makes the network seed-*conditioned* rather than
a plain segmenter. Without it the network answers "where is RNV in this patch"; with it,
"which lesion does this point belong to" — the question the evaluation actually asks.

The subtlety is that the training crop is jittered ±32 px while the heat-map stays on the true
seed. Had the seed always sat at the patch centre, the channel would be constant and therefore
ignorable, and the network would learn nothing from it. `test_cnngvf` asserts the heat-map peak
tracks the seed through the jitter, precisely because this is silent when wrong.

### 4. U-Net → probability map `P` — `05_prob.png`

**Does:** `unet([512 512 3], 2, EncoderDepth=4)`, four down/up stages with skip connections,
softmax over {background, RNV}; `P` is the foreground channel. Per-seed patches are predicted
in batches of `inferBatch`, then stitched with a raised-cosine window so overlapping seeds
leave no seams.

**Matters most.** Every large gain in this project came from improving `P`, and every
post-processing lever was worth an order of magnitude less:

| change | effect |
|---|---|
| clustered → unclustered seeds | **+0.086 IoU** (cross-validated) |
| patch 256 → 512 | fixed segments that scored 0.02–0.07 |
| GVF + level set vs thresholding `P` | +0.017 |
| best possible per-image threshold | *worse* than the pipeline (0.5068 vs 0.5315) |

The loss is generalized Dice, whose inverse-square-volume class weighting is what stops the
network collapsing to all-background on a small foreground fraction.

**Its failures cannot be repaired downstream.** A lesion the network never responds to is gone:
under the clustered baseline `1115.png` had recall 0.044 and no contour method recovers it.

### 5. Force field `f = |∇P|` — `06_force.png`

**Does:** gradient magnitude of a lightly blurred `P`, normalised to `[0,1]`.

**Matters because** it defines *what the contour is attracted to*. The GVF of a ridge points at
that ridge from both sides, and the ridge of `|∇P|` is the CNN's boundary — so this is the
choice that makes the level set converge to the boundary rather than anywhere else.

**`f = P` is deliberately not offered**, though [spec.md](spec.md) lists it: the GVF of `P`
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

- **`alfa = 0`** (no balloon), against [spec.md](spec.md)'s ±1.5. That suggestion assumes a
  contour starting far from the object. Here `phi0` is already on the boundary *and* binary
  rather than a distance function, so the Dirac has no narrow band and a balloon pushes on the
  whole image at once. Measured: `alfa` 0 → IoU 0.965, −0.2 → 0.735, −0.5 → 0.130.
- **The advection sign.** `∇g` points *away* from a boundary; a GVF field points *toward* it.
  The substituted term is therefore `−(u,v)·∇φ`. With the opposite sign the front runs away
  from every edge it should stop on.

### 10. Morphology — `12_after_morph.png`

**Does:** `imclose(disk 2)` then `imfill('holes')`.

**Matters because** the target is `FilledGT`, whose lesions are filled — leaving interior holes
would be scored as false negatives. Closing bridges the 1–2 px gaps the level set leaves along
ragged frond boundaries.

Modest but not free: it moves the boundary raggedness index from 8.14 (raw threshold) to 6.04,
against a ground truth of 6.31 — closer to the truth than either the raw threshold or the level
set alone (4.75, which over-smooths).

### 11. Seed component claiming — `13_final_mask.png`

**Does:** `cnngvf.seedMasks` — each seed claims the component it falls inside, or the nearest
qualifying one by centroid; components no seed claims are dropped; anything below
`minArea = 500` cannot be claimed.

**Matters far more than it looks.** This is the stage that converts a probability map into a
*localization-driven* segmentation, and it removes false-positive blobs no threshold can:

| | mean IoU |
|---|---|
| best possible **per-image** threshold (oracle) | 0.5068 |
| the pipeline | **0.5315** |

The pipeline beats an oracle threshold-picker, and this stage is why. `641.png` is the extreme
case: 0.873 in the pipeline against 0.189 at its best possible threshold.

It also mirrors the ground truth's own convention — `separate_segments.py` writes one file per
seed, not per component — so two seeds inside one lesion both return that lesion, exactly as
two ground-truth files would.

## Specification

All values from [`+cnngvf/defaults.m`](../CNN+GVF/+cnngvf/defaults.m); per-arm overrides live
in [`+cnngvf/arm.m`](../CNN+GVF/+cnngvf/arm.m).

### Data

| item | value |
|---|---|
| images | 1024×1024 OCTA en-face — **68 with ground truth**, all seeded (41 DRAC22 + 27 TUH) |
| training seeds | the unclustered CSVs of both datasets — 874 rows, **842 usable** (32 rejected as detector errors; see [Seed tolerance](#seed-tolerance)) |
| inference seeds | **all 874** — rejection uses ground truth, so it cannot apply at inference |
| labels | the GT component containing each seed, or the nearest within `Patch.seedTolerance` |
| scoring | union mask vs the whole GT mask (`EvalMode = "image"`) |

`cnngvf.manifest` defines the domain by scanning each dataset's image directory and keeping
what has a mask, so adding images needs no code change. Adding a *dataset* needs one entry in
`cfg.Datasets`.

**`EvalMode = "segment"` is fenced to DRAC22 + clustered points.** TUH ships no per-seed
masks, and `SegmentFilledGT/` covers only the original 17 DRAC22 images, so
`cnngvf.evaluate` now errors with the reason rather than silently scoring against missing
files. No current arm uses it; it is kept so the original baseline stays reproducible.

Any figure measured before the domain changed is not comparable with one measured after. The
cross-validated 0.4163 was measured on 17 images, the 0.4603 on 41, and neither on 68.

Labels are computed by component lookup, not read from `SegmentFilledGT/`. Verified to
reproduce all 35 DRAC22 segment files exactly, and it works for point sets those files are not
keyed to — which is what lets both the unclustered detections and TUH be trained on at all.
Seeds further than `Patch.seedTolerance` from every lesion are dropped from training:
labelling them against pure background would teach the network to ignore its own heat-map.

**Scoring must be whole-image, not per-segment.** Several detections land on one lesion, so
per-segment scoring would count that lesion once per detection and weight lesions by how many
detections they attracted. Whole-image scoring also charges for lesions missed entirely.

### Input channels

Derived from `cnngvf.loadChannels`, never assumed. The current arms build **2**:

| channel | source | enabled by |
|---|---|---|
| 1 | intensity, optionally median-filtered `[3 3]` | always; filter by `cfg.Denoise` |
| 2 | Frangi vesselness, `fibermetric` scales `[2 4 6 8]` | `cfg.Vessel.enabled` — **off** in current arms |
| last | Gaussian seed heat-map, `sigmaSeed = 32` | always, appended per patch |

Small vesselness scales because RNV vessels are thin; large scales blur the structure away.
Both filters measured ≈0 (see the ablation above) and are off.

**A network is tied to the channel count it trained with.** `cnngvf.inferProbMap` compares
`net.Layers(1).InputSize(3)` and errors readably. The pairing is set once per arm, which is
what stops a runner disagreeing with the network it loads.

### Patches and sampling

| parameter | value | why |
|---|---|---|
| `Patch.size` | **512** | 13 of 35 GT segments need a window > 256 px, and each scored badly at 256. At 512, 32 of 35 are fully covered. |
| `Patch.jitter` | ±32 px | crop offset; the heat-map marks the seed, not the patch centre |
| `Patch.perSeed` | **4** (combined arm) | raised from 1 after measuring what 1 cost — see [The perSeed=1 underfit](#the-perseed1-underfit). ~3 180 patches/epoch, ~5 h to train. |
| `Patch.balanceByImage` | **true** | equal patches per *image*, not per seed |
| `Patch.seedTolerance` | **20 px** (combined arm), 0 on the frozen arms | how far outside a lesion a seed may fall and still be labelled with it. Recovers 49 near-misses, rejects 32 detector errors. See [Seed tolerance](#seed-tolerance). |

**`balanceByImage` does two jobs on the combined domain.** Within a dataset it stops
detection-rich images dominating: sampling uniformly per seed once gave `1115.png` 39% of the
epoch and the top two images 54%. Across datasets it decides the mix — TUH holds 70% of the
usable detections but only 40% of the images, so per-image balancing gives it **40% of the
epoch, not 70%**. That is intended (an image is one observation regardless of how many times
the detector fired), but it is a choice, and the opposite one is a flag away.

The budget is drawn **with replacement from each image's own seeds** — taking the first *N*
would pin a seed-rich image to the same subset in CSV order for the whole run.

### Network

`unet([512 512 3], 2, EncoderDepth=4)`, loss `1 - generalizedDice`, Adam, lr 1e-3,
30 epochs, `OutputNetwork="best-validation"`.

| parameter | value | why |
|---|---|---|
| `Net.miniBatch` | **2** | 512² is 4× the activations of 256; miniBatch 4 died out of memory at iteration 0 on an 8 GB GPU. Validation follows `MiniBatchSize` — `trainingOptions` has no `ValidationMiniBatchSize` for `adam`. |
| `Net.inferBatch` | **4** | an image can carry 62 seeds; they cannot all go through `predict` at once |

Generalized Dice, not cross-entropy: RNV is a small foreground fraction, and its
inverse-square-volume class weighting is what stops the network collapsing to all-background.

`cnngvf.trainNet` calls `gpuDevice(1)` first so a stale allocation cannot starve the run, and
archives any existing network with a timestamp before saving.

### GVF and level set

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

### Post-processing

`imclose(disk 2)` → `imfill('holes')` → `seedMasks(minArea = 500)`. Each seed claims the
component it falls in, or the nearest qualifying one; unclaimed components are dropped.
Holes are filled because the ground truth is `FilledGT`.

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

## Running it

```matlab
cd RNV-Claude/CNN+GVF
test_cnngvf                            % 8 checks; the last two need the datasets

run_train                              % combined arm -> models/rnvNet_DRACTUH_unclustered.mat
run_batch("DRACTUH_unclustered")       % -> result/DRACTUH_unclustered/
run_cv                                 % the defensible number, 5 folds over 68 images

run_steps("458.png")                   % every intermediate, one DRAC22 image
run_steps("nv (21).png")               % ... or one TUH image

run_train("unclustered")               % the frozen DRAC22-only arm
run_eval(dir)                          % re-score cached masks, ~1 s, no GPU
```

`run_train`, `run_batch`, `run_cv`, `run_eval` and `run_steps` all take an **arm name**.
`run_train_unclustered` and `run_batch_unclustered` survive as thin wrappers on the
DRAC22-only arm.

**MATLAB R2026a.** Needs Image Processing, Deep Learning, Computer Vision and Parallel
Computing. R2025b on this machine has Image Processing only and fails immediately with a
message from `cnngvf.defaults`.

## Architecture

### Data contracts

Five, none enforced by a type.

**Seeds are `[cx cy]` = (column, row).** `rnv.loadSeeds` reads columns *positionally*, so a
header may be renamed but not reordered. Every consumer flips to `(row, col)` at the point of
use. Getting this backwards is silent: a transposed seed still lands inside the image and
still produces a mask.

**An image is addressed by name, resolved through the manifest.** `cnngvf.manifest` maps a
name to its image file, its mask and its dataset. Names must be unique across datasets — they
are the join key for the CSVs and for output filenames — and a collision is an error, not a
mask silently scored against the wrong image. The image → mask rule is per-dataset data
(`GtFrom`/`GtTo`, a regexprep on the stem): identity for DRAC22, `nv (N)` → `gt (N)` for TUH.

**Ground-truth labels are computed, not read.** A seed is labelled with the GT connected
component it falls in. That is what lets point sets with no per-seed files be trained on.

**`cfg.EvalMode` picks the unit, and the units are not comparable.** On identical predictions,
recall falls 0.681 → 0.527 purely from the mode change. `"segment"` is fenced to DRAC22 +
clustered points.

**`run_batch` clears its output directories before writing.** Per-seed masks are named by
coordinates, so a changed CSV would otherwise leave two generations side by side — and
`run_eval` recovers seeds from those filenames.

### Control flow

One image, one call — `cnngvf.segmentImage`:

```
imgPath ──► loadChannels ──► chans (h x w x C)
                               │
seeds (S x 2) ─────────────────┼──► inferProbMap ──► P        [seedPatch per seed]
                               │                      │
                               │        buildForce ◄──┤        f = |∇P|
                               │             │        │
                               │         gvfField     └──► initPhi ──► phi0
                               │          (u,v)                 │
                               └──────────────► drlse ◄─────────┘
                                                  │
                     imclose ──► imfill ──► seedMasks ──► mask + perSeed{}
```

Nine stages, one file each, all taking `cfg`. Above them sit three domain files:
`manifest` (which images), `loadPoints` (which seeds), `arm` (which configuration).

`inferProbMap` is where seed-conditioning lives: seeds are the only patch centres, so anywhere
no seed reaches stays `P = 0` by design. `drlse` is why there is no per-lesion bookkeeping —
one `φ` holds every lesion and splits natively. `seedMasks` is called twice with different
`minArea`, which is why it is a parameter: at initialization a small blob is a lesion that has
not grown, after evolution it is noise.

### Arms

An arm is one comparable configuration, defined once in `+cnngvf/arm.m`:

| arm | datasets | points | channels | perSeed | output |
|---|---|---|---|---|---|
| `clustered` | DRAC22 | clustered | 3 | 16 | `result/` |
| `unclustered` | DRAC22 | unclustered | 2 | 4 | `result/unclustered/` |
| **`DRACTUH_unclustered`** | DRAC22 + TUH | unclustered | 2 | 1 | `result/DRACTUH_unclustered/` |

Everything that must agree between training and inference is set here. That is the point:
`run_train` and `run_train_unclustered` were once near-copies that drifted apart on the
channel flags — a mismatch a network cannot survive — and `run_steps` was left behind by the
same change for weeks. A runner now asks for an arm by name and cannot disagree with another
runner about what it means. Adding a configuration means one table entry, not a new file.

Arms over the **same** domain share folds and compare paired image-by-image. Arms over
different domains do not: `unclustered` is 41 images, `DRACTUH_unclustered` is 68.

### Invariants that fail silently

- **The seed heat-map must track the seed through the training jitter.** If the seed always sat
  at the patch centre the channel would be constant, therefore ignorable, and the network would
  learn a plain segmenter instead of a seed-conditioned one. `test_cnngvf` asserts the peak
  position for exactly this reason.
- **`OutputNetwork="best-validation"` selects on `cfg.Split.Val`.** Those images are *not* held
  out. Quoting them as validation inflated one estimate by +0.089.
- **`Split.Val` is derived, never written down.** The old literal list of four DRAC22 files
  survived the dataset growing to 41 and TUH arriving, still validating on the same four and on
  none of TUH. `cnngvf.valSplit` now spreads the picks across the area range *and* across
  datasets, and `test_cnngvf` asserts every dataset is represented.
- **Inner-validation picks must be interior, not the head of the area-sorted list.** Taking the
  head hands checkpoint selection the largest lesions; in a one-epoch dry run that made a fold
  keep its untrained network and predict 0 px on every held-out image.
- **`balanceByImage` draws with replacement.** Taking the first *N* would pin a seed-rich image
  to the same CSV-order subset for the whole run.
- **Every score is tied to the CSVs in place when it was measured.** They live outside version
  control and have been revised repeatedly.

### Environment

`gpuDevice` enumerates CUDA devices only, so an onboard Intel GPU never appears; `cfg.GpuName`
is printed by every runner so the device is on the record. At patch 512 an 8 GB card has little
headroom — `miniBatch` 4 died out of memory at iteration 0, so it is 2 (validation follows
`MiniBatchSize`; `trainingOptions` exposes no `ValidationMiniBatchSize` for `adam`), and
`trainNet` calls `gpuDevice(1)` first so a stale allocation cannot starve the run. No `parfor`:
several workers contending for one laptop GPU is slower than serial. The GVF mex in
`lib/Fast-Gradient-Vector-Flow` is the fast path; `cnngvf.gvfField` falls back to a vectorised
MATLAB loop with a warning, and `mex GVF.cpp` restores it.

## Reporting rules

1. **Quote cross-validated numbers only.** `run_batch` scores every domain image, but most of
   them trained the model.
2. **Never quote a single-split "validation" score as held-out.** `OutputNetwork="best-validation"`
   selects the checkpoint on `cfg.Split.Val`, so those images are not held out. Doing this
   inflated one estimate by +0.089 (`461.png`: 0.731 single-split vs 0.307 cross-validated).
3. **Do not compare across protocols.** A CV number and a single-split number are not
   comparable in either direction.
4. **Do not compare across domains.** 17-image, 41-image and 68-image figures are three
   different questions. State the domain with every number.
5. **Every score is tied to the CSVs in place when it was measured.** The localization files
   have been revised repeatedly and live outside version control.

## Known limits and next steps

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
- **68 ground-truth images** is still thin for a U-Net; [spec.md](spec.md) suggests pretraining
  on ROSE or OCTA-500.
