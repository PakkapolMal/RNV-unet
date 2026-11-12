---
title: RNV Segmentation Pipeline — Specification
description: The working specification of the current pipeline — architecture, contracts, parameters, and where the measurement behind every non-obvious choice lives.
kind: specification
status: current
dataset: DRAC22 + TUH (69 images, standardized datasets/)
updated: 2026-11-12
---

# RNV Segmentation from Localization Points — Specification

How the current pipeline is wired, what it assumes, and why each non-obvious choice was
made. It describes the code on `master` and nothing else; how the pipeline got here is in
[results/](results/). Implementation: [`../cnn/`](../cnn/). Vocabulary:
[CONTEXT.md](CONTEXT.md). Background theory: [THEORY.md](THEORY.md). The current
cross-validated number is in the [README](../README.md).

## The domain

**69 images across two datasets**, pooled, read from the standardized
`datasets/` (not in the repository) shared with the team. **Every image is in scope,
with or without points**, and every image must have a mask; `cnngvf.manifest` is the single
place that says so, and errors on an image without a mask. The datasets are read-only
(CLAUDE.md, "Datasets").

| | images | points (unclustered) | usable | rejected | no points | GT area range |
|---|---|---|---|---|---|---|
| **DRAC22** | 42 | 280 | 253 | 27 | 3 | 601 – 239 227 px |
| **TUH** | 27 | 591 | 584 | 7 | 0 | 1 126 – 432 602 px |
| **combined** | **69** | **871** | **837** | **34 (3.9%)** | **3** | 601 – 432 602 px |

Usable counts are at `Patch.seedTolerance = 20` — see [Seed tolerance](#seed-tolerance)
below. At tolerance 0 (the frozen arms' setting) they are 228 / 562 / 790.
[results/2026-10-22-standardized-datasets.md](results/2026-10-22-standardized-datasets.md)
lists what changed from the previous 68-image domain. Figures elsewhere in this file that
name 68 images or 874 points were measured on that domain.

Both are OCTA en-face, 1024×1024, but from **different devices** — DRAC22 as
distributed by the DRAC 2022 challenge, TUH from a ZEISS PLEX Elite. TUH
localization is markedly cleaner (5.6% of detections off-lesion against 17.3%),
and its lesions run larger: `nv (21).png` is 41% of the frame and carries 113 of
its dataset's 591 detections.

Folds are **pooled across datasets**, not stratified by device. The research
question is RNV segmentation, not per-device performance. Area-stratified
round-robin dealing interleaves the datasets in practice — every fold holds both —
and `per_image.csv` carries a `Dataset` column so a pooled run can be broken out
afterwards without re-running.

The split is published as [folds.csv](folds.csv) (`image, dataset, fold`), for use
outside this repo, for example to match a peer's folds. `cnngvf.cvFolds` stays the
truth: `test_cnngvf` fails if the file stops matching it.

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
points is the right repair.

`cfg.Patch.seedTolerance = 20` px labels a seed with the nearest lesion within that
distance, and drops the rest: **49 recovered, 32 rejected**. The exact value does not
matter — the empty 50–100 px band means anything in 20–100 behaves identically — and 20
sits inside the ±32 px crop jitter the network already sees. Distance is to the
**boundary**, not the centroid, the same rule `cnngvf.seedMasks` claims with at inference.

**Rejection uses ground truth, so it applies to training only.** Filtering the points at
*inference* by whether they hit a lesion would be leakage, and no such number is
reportable. The oracle version of that filter bounds the headroom at about +0.03 IoU, and
no filter that reads only `P` came close: stray seeds fire with `P ≈ 1`, so the headroom
belongs to the localization stage ([results/2026-09-17-refine.md](results/2026-09-17-refine.md),
step 2).

Six DRAC22 images end up with ground truth but no usable seed — **`082.png`, `217.png` and
`445.png`** have no points at all, **`299.png`, `573.png` and `664.png`** only points beyond the
tolerance. They are still scored. The three with no points go to the seedless network
([stage 9](#9-no-points--the-seedless-mask)): `082` and `217` now score 0.79 and 0.86, and
`445` still 0. The three with stray points score ≈ 0, because recognising a point as wrong
needs GT. That is a localization-stage gap charged to the segmenter, which is the honest
accounting: whole-image scoring charges for lesions missed entirely.

**Data-quality note (2026-09-29, 68-image domain).** `299.png` and `445.png` were upstream
localization bugs, not segmentation failures. All four of `299.png`'s detections are errors, and **`445.png`'s single detection,
(878, 653), is a copy of `443.png`'s** — a copy error in the detector output, so other copies
may exist unchecked. Points placed from the ground truth were tried as training-only seeds
and reverted: two extra seeds could not move `P` measurably, and a second hand-edited CSV was
a maintenance risk. They are kept, unused, in `datasets/DRAC22_raw/` as
`DRAC_RNV_Points_Unclustered_gt_corrected.csv`. The fix belongs in the localization stage;
the pipeline reads only the detector's CSV. A mean over the images with working detections
may be reported as a labelled secondary figure, never in place of the all-image headline.

### The TUH acquisition overlay

Every TUH image carries burned-in device graphics: a **magenta vertical and cyan
horizontal crosshair spanning the whole frame**, plus a corner watermark. After
`rgb2gray` these become bright, perfectly straight, high-gradient lines, and they cross
the middle of the frame where lesions are.

**They are deliberately left in, and measured as harmless.** The crosshair sits at rows and
columns 510–513 in all 27 images. False positives within ±3 px of it occur at 0.80× the rate
elsewhere in the covered region (CI [0.34, 1.28]). Deleting every in-band false positive,
an oracle that uses GT, would gain only +0.0019 TUH IoU, so no inpaint-and-retrain fix is
worth a run (measured 2026-11-05; the record is not public).

## The core idea

The localization stage emits raw detections, then DBSCAN clusters them and keeps one
representative per lesion. **Train and infer on the raw detections, not the
representatives.** A 512 px patch cannot cover a 239 227 px lesion: one cluster centre sees
one window of it, while 62 detections tile it. That coverage is the whole mechanism, and it
is why the gain over cluster centres was concentrated in large lesions and showed up as
recall ([results/2026-08-19-clustered-vs-unclustered.md](results/2026-08-19-clustered-vs-unclustered.md)).

The detections are not a superset of the centres — check per image rather than assuming.

## Pipeline

```
image ─► intensity ────────────────┐
                                   ├─► 512² seed-centred patch ─► U-Net ─► P
seed (x,y) ─► Gaussian heat-map ───┘                                       │
                                                                           ▼
       whole frame ─► seedless U-Net ─► P_s ─► fusion: P = max(P, P_s > 0.9)
                                                                           │
                                                              BW = P > tau (0.5)
                                                                           │
                                          seed fallback: relax at 0.20 ◄───┘
                                                                           │
                        seedMasks ◄─ imfill ◄─ imclose ◄───────────────────┘
                            │
                            ▼
              seed hull: fill triangles of seeds within 128 px ─► mask
```

A second, **seedless network** sees the whole frame with no seed input. Its `P` is used
twice: fused into every image's seeded `P` before the threshold
([stage 4a](#4a-fusion-with-the-seedless-p)), and, for an image with **no points at all**, as
the mask itself: `P > 0.40`, closed, filled, `MinArea` ([stage 9](#9-no-points--the-seedless-mask)).

One mask is built for the whole image and the seeds decide afterwards which components
survive — that is what lets a lesion split into fronds, which a per-seed contour cannot do.
Per-seed masks are unioned into the image-level mask.

The pipeline used to evolve a GVF-driven DRLSE level set between `P` and the morphology.
Removing it cost 0.0017 IoU on 68 images, not significant, because the level set converged
to the `P ≈ 0.5` iso-contour anyway ([results/2026-09-16-reduce.md](results/2026-09-16-reduce.md)).
It was deleted on 2026-09-25.

## Stage by stage

Each stage below is written out by [`run_steps.m`](../cnn/run_steps.m) under the numbered
filename given, so every claim here can be inspected on a real image:
`run_steps("458.png")`.

### 1–2. Denoising and vesselness — `02_denoised.png`, `03_vesselness.png` *(both OFF)*

A `[3 3]` median filter (`cfg.Denoise`) and a Frangi vesselness channel
(`cfg.Vessel.enabled`, `fibermetric` scales `[2 4 6 8]`). Both measured ≈ 0 and are off; the
CNN extracts what it needs from raw intensity
([results/2026-09-01-filters-perseed-levelset.md](results/2026-09-01-filters-perseed-levelset.md)).

### 3. Seed heat-map — `04_seed_heatmap.png`, `patches/seed_NN.png`

**Does:** a Gaussian bump (`sigmaSeed = 32`) at the seed's position **within the patch**,
appended as the last input channel.

**Matters because** it is the only thing that makes the network seed-*conditioned* rather than
a plain segmenter. Without it the network answers "where is RNV in this patch"; with it,
"which lesion does this point belong to" — the question the evaluation actually asks.

The subtlety is that the training crop is jittered ±32 px while the heat-map stays on the true
seed. Had the seed always sat at the patch centre, the channel would be constant and therefore
ignorable. `test_cnngvf` asserts the heat-map peak tracks the seed through the jitter,
precisely because this is silent when wrong.

### 4. U-Net → probability map `P` — `05_prob.png`

**Does:** an ImageNet-pretrained EfficientNet-B3 U-Net with `C+1` input channels (see
[Network](#network)), softmax over {background, RNV}; `P` is the foreground channel. Per-seed patches are predicted in batches of `inferBatch`, then
stitched with a raised-cosine window so overlapping seeds leave no seams. Regions no seed
reaches stay `P = 0` by design.

**Matters most.** Every large gain in this project came from improving `P` — unclustered
seeds, the 512 px patch, `perSeed` 1 → 4, the pretrained encoder recipe — and every
post-processing lever was worth an order of magnitude less. The remaining large-lesion misses are low `P` *inside* the patch, not
lesions outside it: the lever is training, not geometry
([results/2026-09-17-refine.md](results/2026-09-17-refine.md), step 5).

**Its failures cannot be repaired downstream.** A lesion the network never responds to is gone,
unless the seedless network sees it (stage 4a).

### 4a. Fusion with the seedless P

**Does:** for every image with points, `cnngvf.fuseSeedless` replaces the seeded `P` with
max(seeded `P`, seedless `P` where it exceeds `Post.fuseTauS` = 0.9, else 0), over the whole
image. Stages 5–8 then run unchanged on that map.

**Matters because** the seeded `P` is defined only inside the 512 px patches around seeds, and
stops short of the extent of the largest lesions. The seedless network sees the whole frame.
The fused pixels are about three-quarters lesion. On the two largest gainers, `795` and `461`,
83 % of the added area lies outside every seeded patch.

**Seeds still decide what survives.** A seedless blob that no seed's component reaches is
dropped by claiming, like any other.

**Worth +0.0104 IoU on a second, independent seedless network** (CI [+0.004, +0.019]) and
+0.0120 on the adopted one, with the cell fixed before the second run; +0.0116 cross-fitted
where it was first chosen ([results/2026-11-12-fusion-confirm.md](results/2026-11-12-fusion-confirm.md)).
- The gain is a trade: recall +0.018, precision −0.005. It is largest on the largest lesions.
- About half of it is `795` and `461`, on both runs. The other 67 images gain +0.006.
- An uncertainty gate (U) and a sliver filter were both tried and lost to plain fusion
  ([results/2026-11-12-fusion-uncertainty.md](results/2026-11-12-fusion-uncertainty.md)).

### 5. Threshold and seed fallback — `06_binary_raw.png`

**Does:** `BW = P > Post.tau`, then gives every seed the threshold left empty a region of
its own (`cnngvf.seedFallback`).

`tau = 0.5` is **pre-registered**: the softmax decision boundary. It sets the output
directly, so sweeping it on the scored images and quoting the best is leakage. The measured
`tau` curve is flat around 0.5 on the combined domain.

The fallback is not a rare safety net: **258 of 874 seeds (29.5%, on 34 of 68 images) never
produce `P > tau`**, so for them the fallback *is* the output. `"relax"` thresholds `P` once
at `relaxFloor = 0.20` inside a ±128 px window around the seed and takes the component
under it — the lesion's own outline as the CNN saw it weakly. Only when that finds nothing
claimable does the seed get a 15 px disk, so relax never removes the recall floor. The
floor was pre-registered and confirmed cross-fitted
([results/2026-09-16-reduce.md](results/2026-09-16-reduce.md) §8–9).

### 6. Morphology — `07_after_morph.png`

**Does:** `imclose(disk 2)` then `imfill('holes')`.

The target is `FilledGT`, whose lesions are filled, so interior holes would be scored as
false negatives. The close is small but measured positive on the threshold pipeline.

### 7. Seed component claiming — `08_final_mask.png`

**Does:** `cnngvf.seedMasks` — each seed claims the component it falls inside, or the
nearest qualifying one by **boundary** distance; components no seed claims are dropped;
anything below `minArea = 500` cannot be claimed.

**Matters far more than it looks.** This is the stage that converts a probability map into a
*localization-driven* segmentation, and it removes false-positive blobs no threshold can. On
the 17-image domain the pipeline beat an oracle that picks the best threshold per image,
0.5315 against 0.5068, because of this stage.

It mirrors the ground truth's own convention — `separate_segments.py` writes one file per
seed, not per component — so two seeds inside one lesion both return that lesion.

Boundary, not centroid: a seed just outside a large concave lesion can be nearer another
blob's centroid than its own. It is also the rule training labels use, so the pipeline answers
"which lesion does this seed belong to" the same way at both ends. Fixing this was measured
alone before anything else ([results/2026-09-16-reduce.md](results/2026-09-16-reduce.md) §1).

### 8. Seed hull — also `08_final_mask.png`

**Does:** `cnngvf.seedHull` — after claiming, every triangle of the seeds' Delaunay
triangulation whose three seeds are pairwise within `hullLink = 128` px is added to the mask,
with holes enclosed by those triangles filled. Each seed's mask grows to its component of the
new mask, so the image mask stays the union of the per-seed masks.

**Matters because** of **inter-frond gaps**. `P` follows the vessels, while FilledGT marks the
space between them, and that space opens onto the lesion's edge, where `imfill` cannot reach
it. Detections tile large lesions, so seeds close together bracket those gaps. The fill adds
only inside the seed cloud and never touches a lesion's rim, which is why it costs far less
precision than a closing at a similar scale. Worth +0.0087 IoU cross-fitted on 68 images,
nearly all of it on the largest lesions
([results/2026-09-30-seed-hull-d.md](results/2026-09-30-seed-hull-d.md)).

**Adopted against its own gate.** The pre-registered precision guard (smallest three
GT-area quintiles, limit −0.01) measured −0.0106. It was overridden by decision, recorded in
the results file. The IoU gain cleared its CI and 4 of 5 folds chose 128.

At 64 px (2 · `sigmaSeed`, where two heat-map bumps merge) it was inert: 97 % of the
qualifying area was already in the mask
([results/2026-09-30-seed-hull.md](results/2026-09-30-seed-hull.md)).

### 9. No points — the seedless mask

**Does:** for an image with **zero** localization points, `run_cache` reads the seedless
network's whole-frame `P` from `Post.seedlessCache` (through `cnngvf.probCache`'s leakage
guard), and `cnngvf.noPointMask` returns `P > noPointTau`, closed, filled, with components
below `MinArea` dropped. Every image with points goes through stages 3–8, with stage 4a.

**Matters because** without it such an image gets `P = 0` by construction and scores 0. In
this domain that is `082`, `217` and `445`. With it they score 0.788, 0.863 and 0, which is
+0.024 on the 69-image headline. The other 66 images do not move
([results/2026-11-12-seedless-fallback.md](results/2026-11-12-seedless-fallback.md)).

**Why it is honest.** "This image has no points" is known at inference. `noPointTau` was
cross-fitted on the 18 images with 1–3 points, scored as if they had none: chosen by point
count, never by the three targets. Their cross-fitted IoU, 0.467, is the estimate of what the
rule does on images like the targets; the targets' 0.550 rests on 3 images.

**Why 0.40 and every component.** The threshold barely matters: 0.30–0.90 give the targets
0.55–0.56, because a seedless `P` is close to binary. Keeping only the component with the most
`P` mass lost about 0.03 on the proxy at every threshold, so the rule keeps all and needs no
assumption that every image contains RNV.

**The seedless network** is the adopted network with intensity only and no heat-map, trained
on whole 1024² frames (see [Network](#network)). Alone it scores 0.520 IoU on DRAC22 and 0.473
on TUH, below the seeded pipeline on both. That is why it is the mask only where there are no
seeds, and elsewhere only adds to the seeded `P` (stage 4a).

**`run_batch` and `run_steps` apply neither this stage nor 4a.** Like the seeded network, the seedless one
trains in PyTorch, and those runners load MATLAB networks; they predict nothing for an image
with no points.

## Specification

All values from [`+cnngvf/defaults.m`](../cnn/+cnngvf/defaults.m), which is the current
configuration; the frozen arms' overrides live in [`+cnngvf/arm.m`](../cnn/+cnngvf/arm.m).

### Data

| item | value |
|---|---|
| images | 1024×1024 OCTA en-face — **69, all scored** (42 DRAC22 + 27 TUH); 66 carry points |
| training seeds | the unclustered CSVs of both datasets — 871 rows, **837 usable** |
| inference seeds | **all 871** — rejection uses ground truth, so it cannot apply at inference |
| labels | the GT component containing each seed, or the nearest within `Patch.seedTolerance` |
| scoring | union mask vs the whole GT mask |

`cnngvf.manifest` defines the domain by scanning each dataset's image directory; every image
must have a mask, matched by the dataset's `GtFrom`/`GtTo` file-name rewrite. A point is
matched to its image by stem (the TUH CSV names `.jpg`, the files are `.png`). Adding a
*dataset* needs one entry in
`cfg.Datasets`.

Labels are computed by component lookup, not read from per-seed files. That works for any
point set, which is what lets the unclustered detections and TUH be trained on at all. Seeds further than
`Patch.seedTolerance` from every lesion are dropped from training: labelling them against pure
background would teach the network to ignore its own heat-map.

**Scoring must be whole-image, not per-segment.** Several detections land on one lesion, so
per-segment scoring would count that lesion once per detection and weight lesions by how many
detections they attracted. Whole-image scoring also charges for lesions missed entirely.
Per-segment scoring went with the `clustered` arm; both are at tag `archive/clustered-arm`.

### Input channels

Derived from `cnngvf.loadChannels`, never assumed. The current configuration builds **1**,
and the network sees 2:

| channel | source | enabled by |
|---|---|---|
| 1 | intensity, optionally median-filtered `[3 3]` | always; filter by `cfg.Denoise` (off) |
| 2 | Frangi vesselness, `fibermetric` scales `[2 4 6 8]` | `cfg.Vessel.enabled` (off) |
| last | Gaussian seed heat-map, `sigmaSeed = 32` | always, appended per patch |

**A network is tied to the channel count it trained with.** `cnngvf.inferProbMap` compares
`net.Layers(1).InputSize(3)` and errors readably.

### Patches and sampling

| parameter | value | why |
|---|---|---|
| `Patch.size` | **512** | 13 of 35 DRAC22 segments need a window > 256 px, and each scored badly at 256. |
| `Patch.jitter` | ±32 px | crop offset; the heat-map marks the seed, not the patch centre |
| `Patch.perSeed` | **4** | Training only. Sets the per-epoch patch budget: about `perSeed` × usable training seeds. With `balanceByImage` that budget is shared equally across images, drawn with replacement from each image's own seeds, so a seed gets about `perSeed` patches per epoch on average, not exactly. Each draw gets a fresh ±`Patch.jitter` offset and flip/rotation, so the patches differ every epoch. Why 4: 1 underfit, by +0.141 IoU paired; ~3 370 patches/epoch, ~5 h to train. |
| `Patch.balanceByImage` | **true** | equal patches per *image*, not per seed |
| `Patch.seedTolerance` | **20 px** | recovers 49 near-misses, rejects 32 detector errors |

**`balanceByImage` does two jobs on the combined domain.** Within a dataset it stops
detection-rich images dominating: sampling uniformly per seed once gave `1115.png` 39% of the
epoch. Across datasets it decides the mix — TUH holds 70% of the usable detections but only
40% of the images, so per-image balancing gives it **40% of the epoch, not 70%**. That is
intended (an image is one observation regardless of how many times the detector fired).

The budget is drawn **with replacement from each image's own seeds** — taking the first *N*
would pin a seed-rich image to the same subset in CSV order for the whole run.

### Network

The adopted networks are trained by the PyTorch trainer (`colab/rnvtorch.py`,
`cfg.Net.trainer = "pytorch"`), with the recipe in `cfg.Net`, which reaches it through
`run_export`'s bundle:

| parameter | value | why |
|---|---|---|
| network | `smp.Unet(Net.encoder)`, ImageNet weights, 2 input channels (intensity + seed heat-map), standardised per channel | with the two rows below, **+0.087 IoU** |
| `Net.encoder` | **`"efficientnet-b3"`** | |
| loss | **CE + (1 − soft Dice of the foreground pooled over the batch)**, equal weights | |
| augmentation | ±32 px jitter, flips, `rot90`, then affine p 0.75 (rotation ±15°, scale 0.9–1.1, shear ±5°) and CLAHE p 0.5 on intensity (clip 2.0, 128 px tiles) | |
| `Net.miniBatch` | **8** | the encoder's BatchNorm needs more than 2 |
| `Net.learnRate` → `Net.lrMin` | **1e-4 → 1e-6, cosine**, stepped per epoch | |
| precision | bf16 autocast in training, fp32 at inference | |
| `Net.epochsCv` | **40** | the loss still falls at epoch 20, and still slightly at 40 |
| output network | **the last epoch** | best-validation picked on 2 images' noise |
| inner validation | **none**: the 2 inner images train | selection cost training data |
| `Net.inferBatch` | **4** | Inference only: how many seed patches go through the network at once. It changes speed and GPU memory, never the result. Bounded because an image can carry 113 seeds. |

**The network, loss and augmentation** were adopted together and are worth **+0.087 IoU**
(CI [+0.056, +0.118]; 69 images, 5-fold CV, trained once). Precision rose more than recall
(+0.075 against +0.039): spill images fell from ×2.20 to ×1.17 of GT, under-covered ones rose
from ×0.52 to ×0.86 ([results/2026-11-11-paper-recipe.md](results/2026-11-11-paper-recipe.md)).
The three were tested as one bundle, so which carries the gain is unknown.

**The schedule** (40 epochs, cosine, last epoch, no inner validation) was adopted before them,
on the replica U-Net: **+0.025 IoU** over 20 epochs at a constant rate with best-validation
output ([results/2026-11-07-train-budget.md](results/2026-11-07-train-budget.md)). It worked by
making `P` more decisive ([results/2026-11-10-train-budget-pmaps.md](results/2026-11-10-train-budget-pmaps.md)).

**Stalls.** A stalled fold's training loss plateaus early, far above the others'. The stall
rule reads training loss only, before scoring: a fold whose last-five-epoch mean loss exceeds
**1.25 × the median** of the five folds' is retrained once with the next `--rng`, and that
retrain is used whatever it scores. The rule is relative because the absolute loss depends on
the loss function. Under the replica recipe 2 of 16 fold trainings stalled; under this one, 0
of 5 ([results/2026-11-07-train-budget.md](results/2026-11-07-train-budget.md),
[results/2026-11-11-paper-recipe.md](results/2026-11-11-paper-recipe.md)).

**The seedless network** (`rnvtorch.py train-seedless`) uses the same network, loss,
optimiser and schedule, with intensity only (no heat-map). It trains on whole 1024² frames with
every lesion as target, `Net.seedlessCopies` (10) augmented copies of each training image per
epoch, `Net.seedlessEpochs` (60) epochs and minibatch `Net.seedlessMiniBatch` (4). Its
augmentation adds a translation of up to ±5 % and uses 8 × 8 CLAHE tiles, 128 px on the frame
as on the seeded patch. It trains on every non-test image, including those without points.
No fold stalled in its one run ([results/2026-11-12-seedless-fallback.md](results/2026-11-12-seedless-fallback.md)).

**The replica U-Net stays in the trainer as a test fixture.** `rnvtorch.UNet` and
`generalized_dice` port MATLAB's `unet([512 512 2], 2, EncoderDepth=4)` and its loss, and
`test_parity.py` checks them against MATLAB's fold networks, which also checks the shared patch
code. No current network is that U-Net.

**The MATLAB trainer** (`cnngvf.trainNet`) runs the frozen arms' recipe: the replica U-Net,
`1 − generalizedDice`, Adam 1e-3 with L2 1e-4 on weights, minibatch 2 (4 died out of memory at
iteration 0 on an 8 GB GPU), 30 epochs (20 under `run_cv`), constant rate,
`OutputNetwork="best-validation"`. `cnngvf.arm` sets its minibatch and rate. For the current
configuration, `trainNet` and `run_cv` refuse.

`cnngvf.trainNet` calls `gpuDevice(1)` first so a stale allocation cannot starve the run, and
archives any existing network with a timestamp before saving.

### Post-processing

| parameter | value | why |
|---|---|---|
| `Post.tau` | **0.5** | pre-registered decision boundary; cross-fitting 0.40–0.70 loses 0.004 IoU ([results](results/2026-11-05-tau-crossfit.md)); re-fitted with `relaxFloor` on the adopted recipe's `P`: +0.0004, CI including 0 ([results](results/2026-11-11-tau-floor-refit.md)) |
| `Post.fallbackMode` | **`"relax"`** | shape from `P` for seeds the threshold missed; `"disk"` on the frozen arms |
| `Post.relaxFloor` | **0.20** | fixed after the 2026-09-16 sweep; re-tuning it jointly with the window loses 0.0098 cross-fitted ([results](results/2026-09-30-relax-grid.md)), and jointly with `tau` on the adopted `P` gains nothing ([results](results/2026-11-11-tau-floor-refit.md)) |
| `Post.relaxWindow` | 128 px | stops a low floor flooding a broad plateau; the best of 64 / 128 / 256 / 512 at every floor 0.10–0.25 ([results](results/2026-09-30-relax-grid.md)) |
| `Post.seedRadius` | 15 px | the disk when relax finds nothing claimable; near-inert under relax: r 0–30 all within 0.0003, 9 images change ([results](results/2026-10-02-minarea-disk.md)) |
| `Post.MorphElement` | `strel('disk', 2)` | `imclose` before `imfill` |
| `Post.MinArea` | 500 px | components below it cannot be claimed; must stay under the smallest component the network fires on (~700 px: `421`, `217`). Cross-fitted over 100–1000 and jointly with `seedRadius`: every fold picks 500 ([results](results/2026-10-02-minarea-disk.md)) |
| `Post.ClaimRule` | **`"boundary"`** | same rule as training labels; `"centroid"` on the frozen arms |
| `Post.hullLink` | **128 px** | seed hull pairwise bound, cross-fitted over 64–256; 0 turns it off, as on the frozen arms ([results](results/2026-09-30-seed-hull-d.md)) |
| `Post.seedlessCache` | **`<Output>/cv/probcache_seedless`** | the seedless network's CV `P`, read by stages 4a and 9; `""` turns both off, as on the frozen arms |
| `Post.fuseTauS` | **0.9** | stage 4a: the seedless `P` enters only above it. It won a 0.0009 tie against 0.5 and was then fixed before the replication ([results](results/2026-11-12-fusion-confirm.md)) |
| `Post.noPointTau` | **0.40** | cross-fitted on the images with 1–3 points; 0.30–0.90 all within 0.011 on the targets ([results](results/2026-11-12-seedless-fallback.md)) |

## Running it

```matlab
cd RNV-Claude/cnn
test_cnngvf                            % 9 checks; needs datasets/

run_batch                              % -> result/DRACTUH_unclustered/  (mostly in-sample)
run_cache                              % the reportable number: cached 5-fold CV P over 69 images

run_steps("458.png")                   % every intermediate, one DRAC22 image
run_steps("nv (21).png")               % ... or one TUH image
run_eval(dir)                          % re-score cached masks, ~1 s, no GPU

run_cv("unclustered")                  % a frozen arm trains and scores in MATLAB
```

Every runner takes nothing (the current configuration), a frozen arm's name, or a `cfg`
struct. For a frozen arm, `cfg.CV.skipTrain = true` makes `run_cv` reuse existing fold
networks — inference only, minutes instead of ~18 h, and how an interrupted run resumes.

**Verifying a change after `P`.** `run_cache` scores the fold-cached `P`
(`cfg.CV.probCache`, default `<Output>/cv/probcache`) through the current post-processing
and compares each image's IoU with `cfg.CV.reference`, the adopted configuration's record in
`results/reference/`. A refactor reproduces it to ~1e-15; anything larger is a result.

**Training is PyTorch, on Colab (`colab/`).** The current configuration's fold networks are
trained there with the recipe in `cfg.Net` (see Network). `run_export` writes everything
training and inference read (channels, labels, seeds, folds, the recipe) to
`<Output>/colab_bundle`, so the Python side never reads `datasets/`. A trained fold writes its
held-out `P` in the `probCache` format, and `run_cache` scores it with the leakage guard
unchanged. The port matches MATLAB to within 0.005 IoU on the 69 images, 5-fold CV
([results/2026-11-05-colab-parity-cv.md](results/2026-11-05-colab-parity-cv.md)).

**The adopted fold networks are PyTorch, and MATLAB does not load them.** Their `P` is the
cache at `<Output>/cv/probcache`, copied from the run in `<Output>/colab_bcedice/`; the
seedless network's is `<Output>/cv/probcache_seedless`, copied from `<Output>/colab_seedless/`. The caches
it replaced stay where they were made: `<Output>/colab_budget/` (the 2026-11-07 schedule on the
replica U-Net) and `result/archive/cv_matlab_until_2026-11-10/` (MATLAB). `run_cache` is how
this configuration is scored, and `run_cv` refuses it.

**`run_batch` and `run_steps` still load the MATLAB full-data network**
(`models/rnvNet_DRACTUH_unclustered.mat`, the frozen arms' recipe, 30 epochs). Their masks and
steps therefore show a network two adoptions old, not the adopted one. No full-data network has
been trained with the adopted recipe, and MATLAB cannot load an EfficientNet-B3 U-Net written
back from PyTorch the way it loaded the replica. Showing the adopted network there needs `P`
written by Python, or an ONNX import; not decided.

**The baseline for a new trained arm is the reference run** (`colab_bcedice/`, one run), judged
by the margin in Reporting rule 7. The two parity runs,
`cnn/result/DRACTUH_unclustered/colab_baseline/parity_rng0` and `parity_rng1`, are the
replica U-Net's noise-floor measurement. Their `P` caches are not in git; do not delete them.

**MATLAB R2026a.** Needs Image Processing, Deep Learning, Computer Vision and Parallel
Computing. R2025b on this machine has Image Processing only and fails immediately with a
message from `cnngvf.defaults`.

## Architecture

### Data contracts

Five, none enforced by a type.

**Seeds are `[cx cy]` = (column, row).** `cnngvf.loadPoints` reads CSV columns
*positionally*, so a header may be renamed but not reordered. Every consumer flips to
`(row, col)` at the point of use. Getting this backwards is silent: a transposed seed still
lands inside the image and still produces a mask.

**An image is addressed by name, resolved through the manifest.** `cnngvf.manifest` maps a
name to its image file, its mask and its dataset. Names must be unique across datasets — they
are the join key for the CSVs and for output filenames — and a collision is an error. The
image → mask rule is per-dataset data (`GtFrom`/`GtTo`, a regexprep on the file name):
identity for DRAC22, `nv (N).png` → `gt (N).tif` for TUH.

**Ground-truth labels are computed, not read.** A seed is labelled with the GT connected
component it falls in. That is what lets point sets with no per-seed files be trained on. The rule, tolerance included, is `cnngvf.seedLabels`, shared
by `prepareData` and `run_export` so the PyTorch trainer labels identically.

**`run_batch` clears its output directories before writing.** Per-seed masks are named by
coordinates, so a changed CSV would otherwise leave two generations side by side — and
`run_eval` recovers seeds from those filenames.

### Control flow

One image, one call — `cnngvf.segmentImage`:

```
imgPath ──► loadChannels ──► chans (h x w x C)
                               │
seeds (S x 2) ─────────────────┴──► inferProbMap ──► P        [seedPatch per seed]
                                                     │
                                                maskFromProb
                    P > tau ─► seedFallback ─► imclose ─► imfill ─► seedMasks ─► seedHull
                                                     │
                                              mask + perSeed{}
```

Above the stages sit three domain files: `manifest` (which images), `loadPoints` (which
seeds), `defaults`/`arm` (which configuration). `probCache` persists `P` per fold during
`run_cv`, so `run_cache` can score a post-processing change in seconds, and refuses to hand
back a `P` whose network trained on that image.

### Arms

An arm is one comparable configuration. `cnngvf.defaults()` **is** the current arm; the frozen
ones in `+cnngvf/arm.m` override only what differs:

| arm | datasets | points | channels | perSeed | fallback | claim | output |
|---|---|---|---|---|---|---|---|
| **`DRACTUH_unclustered`** | DRAC22 + TUH | unclustered | 1 (+heat-map) | 4 | relax 0.20 | boundary + seed hull 128 | `result/DRACTUH_unclustered/`; trains in PyTorch |
| `unclustered` | DRAC22 | unclustered | 1 (+heat-map) | 4 | disk | centroid | `result/unclustered/`; trains in MATLAB |
| `unclustered_boundary` | DRAC22 | unclustered | 1 (+heat-map) | 4 | disk | boundary | `result/unclustered_boundary/` |

The frozen arms keep their trained networks loadable, but their recorded numbers were measured
with the level set; on `master` they run the threshold pipeline and no longer reproduce those
records. Tag `archive/pre-cleanup` does. The `clustered` arm is at tag `archive/clustered-arm`:
its per-seed ground truth and clustered CSVs are not in the standardized `datasets/`.

Everything that must agree between training and inference is set in one place, so a runner
cannot disagree with the network it loads. Arms over the **same** domain share folds and
compare paired image-by-image; arms over different domains do not.

### Invariants that fail silently

- **The seed heat-map must track the seed through the training jitter.** If the seed always sat
  at the patch centre the channel would be constant, therefore ignorable, and the network would
  learn a plain segmenter. `test_cnngvf` asserts the peak position.
- **`OutputNetwork="best-validation"` selects on `cfg.Split.Val`.** Those images are *not* held
  out. Quoting them as validation once inflated an estimate by +0.089.
- **`Split.Val` is derived, never written down.** A literal list of four DRAC22 files once
  survived the dataset growing to 41 and TUH arriving. `cnngvf.valSplit` spreads the picks
  across the area range *and* across datasets, and `test_cnngvf` asserts every dataset is
  represented.
- **Inner-validation picks must be interior, not the head of the area-sorted list.** Taking the
  head hands checkpoint selection the largest lesions; in a one-epoch dry run that made a fold
  keep its untrained network and predict 0 px on every held-out image.
- **`balanceByImage` draws with replacement.** Taking the first *N* would pin a seed-rich image
  to the same CSV-order subset for the whole run.
- **Every score is tied to the CSVs in place when it was measured.** They live outside version
  control and have been revised repeatedly; `run_batch` stamps their mtimes into the P cache.

### Environment

`gpuDevice` enumerates CUDA devices only, so an onboard Intel GPU never appears; `cfg.GpuName`
is printed by every runner so the device is on the record. At patch 512 an 8 GB card has little
headroom for the MATLAB trainer — `miniBatch` 4 died out of memory at iteration 0, so the frozen
arms use 2 — and `trainNet` calls `gpuDevice(1)` first so a stale allocation cannot starve the
run. The PyTorch trainer fits minibatch 8 in bf16 on the same card. No `parfor`: several workers
contending for one laptop GPU is slower than serial.

## Reporting rules

1. **Quote cross-validated numbers only.** `run_batch` scores every domain image, but most of
   them trained the model.
2. **Never quote a single-split "validation" score as held-out.** `OutputNetwork="best-validation"`
   selects the checkpoint on `cfg.Split.Val`, so those images are not held out.
3. **Do not compare across protocols.** A CV number and a single-split number are not
   comparable in either direction.
4. **Do not compare across domains.** 17-image, 41-image and 68-image figures are three
   different questions. State the domain with every number.
5. **Every score is tied to the CSVs in place when it was measured.**
6. **Pre-register any threshold that sets the output.** State it before measuring; select it
   by cross-fitting if it must be tuned. Then quote the cross-fitted score and ship the value
   chosen on all images. The fixed value's score on those images is optimistic, because it
   won there ([CONTEXT.md](CONTEXT.md), "Cross-fitted"; worked example in
   [results/2026-09-30-seed-hull-d.md](results/2026-09-30-seed-hull-d.md)).
7. **A trained arm trains once (`--rng 0`) and is judged against the training noise floor.**
   Compare it paired with both baseline runs. Two retrains of one configuration differ by a
   95 % CI half-width of about 0.015 IoU (69 images, 5-fold CV), so a smaller mean difference
   is not a result in either direction ([CONTEXT.md](CONTEXT.md), "Noise floor";
   [results/2026-11-05-colab-parity-cv.md](results/2026-11-05-colab-parity-cv.md)).
   When only one run of the baseline exists, the margin replaces the second run: the paired
   ΔIoU needs a CI lower bound above 0 **and** a mean of at least +0.015
   ([results/2026-11-11-paper-recipe.md](results/2026-11-11-paper-recipe.md) first used it).

## Known limits and next steps

- **Inference-only post-processing is nearly exhausted on this `P`.** A claim-distance cap,
  a relax guard and d4 test-time augmentation were all tried and none adopted
  ([results/2026-09-17-refine.md](results/2026-09-17-refine.md)). The seed hull is the one
  exception, at +0.0087. Effort belongs in `P`.
- **The seed hull's 128 px was chosen on these 68 images.** It is cross-fitted, and it was
  adopted over its own precision guard. Confirming it needs images the pipeline has never
  scored. A closing at the heat-map scale recovered more on the largest lesions (+0.061 on
  the top quintile) but lost more on every smaller one
  ([results/2026-09-30-seed-hull.md](results/2026-09-30-seed-hull.md)).
- **The seed hull's precision cost comes from triangles joining two large components.**
  Their pixels are half lesion. Triangles joining a fallback-sized fragment are 82 % lesion,
  so fragments belong to their lesions. Blocking every cross-component triangle repairs the
  guard but costs 0.0028 IoU
  ([results/2026-10-01-seed-hull-claim.md](results/2026-10-01-seed-hull-claim.md)).
- **Precision and recall are now level** (0.742 and 0.735, 69 images, 5-fold CV). Under the
  replica U-Net precision trailed recall by about 0.13, and spill was the failure mode; the
  adopted recipe nearly halved the spill
  ([results/2026-11-11-paper-recipe.md](results/2026-11-11-paper-recipe.md)).
- **The large-lesion gap is model-side, not patch coverage.** 98% of the largest-quintile ground
  truth lies inside a 512 px patch, and 76% of its missed pixels have `P < 0.20`.
- **A fold can stall.** Under the replica U-Net, 2 of 16 PyTorch fold trainings plateaued far
  above the rest; one scored 0.067 below MATLAB on its fold
  ([results/2026-11-05-colab-parity-cv.md](results/2026-11-05-colab-parity-cv.md)). None of 5
  stalled under the adopted recipe. Check every arm's `log.csv` before reading its score.
- **Is `perSeed = 8` better still?** 4 restored the other arms' setting; it is not a measured
  optimum, and the 1 → 4 step was worth +0.141.
- **Stray localization points are confident, not weak.** 29 of 32 fire with `P ≈ 1`, so no
  `P`-based seed filter fits; the ~+0.03 oracle headroom belongs to the localization stage.
- **Tiny lesions are threshold-unstable.** A 0.8% shift in `P` can halve IoU on a ~1 000 px
  lesion by joining a neighbouring blob; expect large single-image swings in the smallest
  quintile from any change to `P`.
- **`tau` 0.5 is measured, not just pre-registered.** Cross-fitted over 0.40–0.70 it loses
  0.0039 IoU (CI [−0.0081, −0.0004], 69 images, 5-fold CV), and it loses on both PyTorch
  baseline caches too. The curve is flat over 0.45–0.60 and peaks at 0.5. The precision gap
  is not a threshold problem ([results/2026-11-05-tau-crossfit.md](results/2026-11-05-tau-crossfit.md)).
  **Re-fitted on the adopted recipe's `P`** jointly with `relaxFloor` (7 × 5 cells): +0.0004
  cross-fitted, CI [−0.003, +0.003], so both stay. Lower is better on both axes in-sample,
  by at most 0.004 ([results/2026-11-11-tau-floor-refit.md](results/2026-11-11-tau-floor-refit.md)).
- **`299.png` and `445.png` were upstream detector bugs** (445's only point was a copy of
  443's; the standardized CSVs give it none). `299` still scores ≈ 0 on its stray points.
  `445` now gets the seedless mask, but the seedless network finds nothing in its lesion
  (`P` = 0.000 inside the GT), so it scores 0 too. Re-localize both upstream.
- **Half of stage 4a's gain is two images.** `795` and `461`, very large DRAC22 lesions, carry
  about half of it on both seedless runs. A domain with fewer such lesions would see about
  +0.006. Bringing the seedless `P` in only at weak seeds gained nothing: under 90 % of them
  the seedless network sees nothing either
  ([results/2026-11-12-fusion-weak-seeds.md](results/2026-11-12-fusion-weak-seeds.md), [results/2026-11-12-fusion-confirm.md](results/2026-11-12-fusion-confirm.md)).
- **The seedless network is noisy on single images.** Two runs of one recipe score the three
  point-less targets 0.550 and 0.680 mean IoU; `445` goes from 0 to 0.363. Stage 9's figure
  rests on one run ([results/2026-11-12-fusion-confirm.md](results/2026-11-12-fusion-confirm.md)).
  Averaging the two runs gained +0.0030, CI including 0: stage 9 gains from the union, stage 4a
  loses at 0.9 from the agreement ([results/2026-11-12-seedless-ensemble.md](results/2026-11-12-seedless-ensemble.md)).
- **Most weak seeds end on the disk.** Of 156 seeds the threshold leaves empty, 87 find no
  relax component of `MinArea` and take the 15 px disk; 69 take relax.
- **The balanced-sampling gain is unverified.** Every cross-validated arm used it.
- **A trained CV costs about 5 h** on a Colab L4 (73–85 s per epoch, 40 epochs, 5 folds), and
  about 18 h for a frozen arm in MATLAB. Any post-`P` change scores off the cache with
  `run_cache` instead.
- **68 ground-truth images** is still thin for a U-Net;
  [briefs/design-study.md](briefs/design-study.md) suggests pretraining on ROSE or OCTA-500.
- **The 2026-09-17 seedless pilot failed** on `1 − generalizedDice` with tiles; the adopted
  loss and a pretrained encoder fixed that, and the seedless network is now stage 9
  ([results/2026-09-17-seedless.md](results/2026-09-17-seedless.md), [results/2026-11-12-seedless-fallback.md](results/2026-11-12-seedless-fallback.md)).
