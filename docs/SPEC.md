---
title: RNV Segmentation Pipeline — Specification
description: The working specification of the current pipeline — architecture, contracts, parameters, and where the measurement behind every non-obvious choice lives.
kind: specification
status: current
dataset: DRAC22 + TUH (68 images)
updated: 2026-09-25
---

# RNV Segmentation from Localization Points — Specification

How the current pipeline is wired, what it assumes, and why each non-obvious choice was
made. It describes the code on `master` and nothing else; how the pipeline got here is in
[results/](results/). Implementation: [`../cnn/`](../cnn/). Vocabulary:
[CONTEXT.md](CONTEXT.md). Background theory: [THEORY.md](THEORY.md). The current
cross-validated number is in the [README](../README.md).

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
question is RNV segmentation, not per-device performance. Area-stratified
round-robin dealing interleaves the datasets in practice — every fold holds both —
and `per_image.csv` carries a `Dataset` column so a pooled run can be broken out
afterwards without re-running.

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

Two images end up with ground truth but no usable seed — **`299.png` and `445.png`**. They
are still scored, and score ≈ 0. That is a localization-stage gap charged to the segmenter,
which is the honest accounting: whole-image scoring charges for lesions missed entirely.

**Data-quality note (2026-09-29).** Both are upstream localization bugs, not segmentation
failures. All four of `299.png`'s detections are errors, and **`445.png`'s single detection,
(878, 653), is a copy of `443.png`'s** — a copy error in the detector output, so other copies
may exist unchecked. Points placed from the ground truth were tried as training-only seeds
and reverted: two extra seeds could not move `P` measurably, and a second hand-edited CSV was
a maintenance risk. They are kept, unused, beside the images as
`DRAC_RNV_Points_Unclustered_gt_corrected.csv`. The fix belongs in the localization stage;
the pipeline reads only the detector's CSV. A mean over the 66 images with working
detections may be reported as a labelled secondary figure, never in place of the 68-image
headline.

### The TUH acquisition overlay

Every TUH image carries burned-in device graphics: a **magenta vertical and cyan
horizontal crosshair spanning the whole frame**, plus a corner watermark. After
`rgb2gray` these become bright, perfectly straight, high-gradient lines, and they cross
the middle of the frame where lesions are.

**They are deliberately left in.** They are removable (colored pixels are trivially
separable in RGB, before the grayscale conversion discards the information), but nothing
removes them today. If a run shows straight-line false positives, this is the first
suspect, and the fix is a masking step in `cnngvf.loadChannels` before `rgb2gray`.

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
                                                              BW = P > tau (0.5)
                                                                           │
                                          seed fallback: relax at 0.20 ◄───┘
                                                                           │
                        mask ◄─ seedMasks ◄─ imfill ◄─ imclose ◄───────────┘
```

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
The frozen `clustered` arm still uses both.

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

**Does:** `unet([512 512 C+1], 2, EncoderDepth=4)`, softmax over {background, RNV}; `P` is
the foreground channel. Per-seed patches are predicted in batches of `inferBatch`, then
stitched with a raised-cosine window so overlapping seeds leave no seams. Regions no seed
reaches stay `P = 0` by design.

**Matters most.** Every large gain in this project came from improving `P` — unclustered
seeds, the 512 px patch, `perSeed` 1 → 4 — and every post-processing lever was worth an order
of magnitude less. The remaining large-lesion misses are low `P` *inside* the patch, not
lesions outside it: the lever is training, not geometry
([results/2026-09-17-refine.md](results/2026-09-17-refine.md), step 5).

**Its failures cannot be repaired downstream.** A lesion the network never responds to is gone.

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

## Specification

All values from [`+cnngvf/defaults.m`](../cnn/+cnngvf/defaults.m), which is the current
configuration; the frozen arms' overrides live in [`+cnngvf/arm.m`](../cnn/+cnngvf/arm.m).

### Data

| item | value |
|---|---|
| images | 1024×1024 OCTA en-face — **68 with ground truth**, all seeded (41 DRAC22 + 27 TUH) |
| training seeds | the unclustered CSVs of both datasets — 874 rows, **842 usable** |
| inference seeds | **all 874** — rejection uses ground truth, so it cannot apply at inference |
| labels | the GT component containing each seed, or the nearest within `Patch.seedTolerance` |
| scoring | union mask vs the whole GT mask (`EvalMode = "image"`) |

`cnngvf.manifest` defines the domain by scanning each dataset's image directory and keeping
what has a mask, so adding images needs no code change. Adding a *dataset* needs one entry in
`cfg.Datasets`.

Labels are computed by component lookup, not read from `SegmentFilledGT/`. That reproduces all
35 DRAC22 segment files exactly, and it works for point sets those files are not keyed to —
which is what lets the unclustered detections and TUH be trained on at all. Seeds further than
`Patch.seedTolerance` from every lesion are dropped from training: labelling them against pure
background would teach the network to ignore its own heat-map.

**Scoring must be whole-image, not per-segment.** Several detections land on one lesion, so
per-segment scoring would count that lesion once per detection and weight lesions by how many
detections they attracted. Whole-image scoring also charges for lesions missed entirely.
`EvalMode = "segment"` exists only for the frozen `clustered` arm and is fenced to DRAC22 +
clustered points; `cnngvf.evaluate` errors with the reason otherwise.

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
| `Patch.perSeed` | **4** | 1 underfit, by +0.141 IoU paired; ~3 370 patches/epoch, ~5 h to train |
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

`unet([512 512 2], 2, EncoderDepth=4)`, loss `1 - generalizedDice`, Adam, lr 1e-3,
30 epochs (20 under `run_cv`), `OutputNetwork="best-validation"`.

| parameter | value | why |
|---|---|---|
| `Net.miniBatch` | **2** | 4 died out of memory at iteration 0 on an 8 GB GPU. Validation follows `MiniBatchSize` — `trainingOptions` has no `ValidationMiniBatchSize` for `adam`. |
| `Net.inferBatch` | **4** | an image can carry 113 seeds; they cannot all go through `predict` at once |

Generalized Dice, not cross-entropy: RNV is a small foreground fraction, and its
inverse-square-volume class weighting is what stops the network collapsing to all-background.

`cnngvf.trainNet` calls `gpuDevice(1)` first so a stale allocation cannot starve the run, and
archives any existing network with a timestamp before saving.

### Post-processing

| parameter | value | why |
|---|---|---|
| `Post.tau` | **0.5** | pre-registered decision boundary; never tuned on scored images |
| `Post.fallbackMode` | **`"relax"`** | shape from `P` for seeds the threshold missed; `"disk"` on the frozen arms |
| `Post.relaxFloor` | **0.20** | pre-registered; confirmed cross-fitted |
| `Post.relaxWindow` | 128 px | stops a low floor flooding a broad plateau |
| `Post.seedRadius` | 15 px | the disk when relax finds nothing claimable |
| `Post.MorphElement` | `strel('disk', 2)` | `imclose` before `imfill` |
| `Post.MinArea` | 500 px | components below it cannot be claimed |
| `Post.ClaimRule` | **`"boundary"`** | same rule as training labels; `"centroid"` on the frozen arms |

## Running it

```matlab
cd RNV-Claude/cnn
test_cnngvf                            % 9 checks; five need the datasets mounted

run_train                              % -> models/rnvNet_DRACTUH_unclustered.mat
run_batch                              % -> result/DRACTUH_unclustered/  (mostly in-sample)
run_cv                                 % the reportable number: 5 folds over 68 images

run_steps("458.png")                   % every intermediate, one DRAC22 image
run_steps("nv (21).png")               % ... or one TUH image
run_eval(dir)                          % re-score cached masks, ~1 s, no GPU
run_cache                              % cached CV P through the current post-processing

run_batch("unclustered")               % a frozen arm, by name
```

Every runner takes nothing (the current configuration), a frozen arm's name, or a `cfg`
struct. `cfg.CV.skipTrain = true` makes `run_cv` reuse existing fold networks — inference
only, minutes instead of ~18 h, and how an interrupted run resumes.

**Verifying a change after `P`.** `run_cache` scores the fold-cached `P`
(`cfg.CV.probCache`, default `<Output>/cv/probcache`) through the current post-processing
and compares each image's IoU with `cfg.CV.reference`, the adopted configuration's record in
`results/reference/`. A refactor reproduces it to ~1e-15; anything larger is a result.

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
image → mask rule is per-dataset data (`GtFrom`/`GtTo`, a regexprep on the stem): identity for
DRAC22, `nv (N)` → `gt (N)` for TUH.

**Ground-truth labels are computed, not read.** A seed is labelled with the GT connected
component it falls in. That is what lets point sets with no per-seed files be trained on.

**`cfg.EvalMode` picks the unit, and the units are not comparable.** On identical predictions,
recall once fell 0.681 → 0.527 purely from the mode change.

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
                                  P > tau ─► seedFallback ─► imclose ─► imfill ─► seedMasks
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
| **`DRACTUH_unclustered`** | DRAC22 + TUH | unclustered | 1 (+heat-map) | 4 | relax 0.20 | boundary | `result/DRACTUH_unclustered/` |
| `clustered` | DRAC22 | clustered | 2 (+heat-map) | 16 | disk | centroid | `result/` |
| `unclustered` | DRAC22 | unclustered | 1 (+heat-map) | 4 | disk | centroid | `result/unclustered/` |
| `unclustered_boundary` | DRAC22 | unclustered | 1 (+heat-map) | 4 | disk | boundary | `result/unclustered_boundary/` |

The frozen arms keep their trained networks loadable, but their recorded numbers were measured
with the level set; on `master` they run the threshold pipeline and no longer reproduce those
records. Tag `archive/pre-cleanup` does.

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
headroom — `miniBatch` 4 died out of memory at iteration 0, so it is 2, and `trainNet` calls
`gpuDevice(1)` first so a stale allocation cannot starve the run. No `parfor`: several workers
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
   by cross-fitting if it must be tuned.

## Known limits and next steps

- **Inference-only post-processing is exhausted on this `P`, measured.** A claim-distance cap,
  a relax guard and d4 test-time augmentation were all tried and none adopted
  ([results/2026-09-17-refine.md](results/2026-09-17-refine.md)). Effort belongs in `P`.
- **Precision is the weak side**, about 0.13 below recall. The failure mode is
  over-segmentation.
- **The large-lesion gap is model-side, not patch coverage.** 98% of the largest-quintile ground
  truth lies inside a 512 px patch, and 76% of its missed pixels have `P < 0.20`.
- **30 epochs may not be enough.** The `perSeed = 4` validation curve was still descending at
  epoch 29. Training longer is the cheapest untested lever.
- **Is `perSeed = 8` better still?** 4 restored the other arms' setting; it is not a measured
  optimum, and the 1 → 4 step was worth +0.141.
- **Stray localization points are confident, not weak.** 29 of 32 fire with `P ≈ 1`, so no
  `P`-based seed filter fits; the ~+0.03 oracle headroom belongs to the localization stage.
- **Tiny lesions are threshold-unstable.** A 0.8% shift in `P` can halve IoU on a ~1 000 px
  lesion by joining a neighbouring blob; expect large single-image swings in the smallest
  quintile from any change to `P`.
- **Under relax, `tau` 0.55–0.60 scores higher on the surface** (up to 0.519) than the
  pre-registered 0.5. Untested as a result; it would need cross-fitting.
- **The TUH acquisition overlay is still in the images.** Straight-line false positives are the
  symptom to watch for.
- **`299.png` and `445.png` have broken detector points** (445's is a copy of 443's) and
  score ≈ 0. Re-localize them in the upstream stage; see [Seed tolerance](#seed-tolerance).
- **The balanced-sampling gain is unverified.** Every cross-validated arm used it.
- **Cross-validation costs ~18 h** at `perSeed = 4` / 20 epochs. `cfg.CV.skipTrain` makes any
  post-`P` change inference-only.
- **68 ground-truth images** is still thin for a U-Net;
  [briefs/design-study.md](briefs/design-study.md) suggests pretraining on ROSE or OCTA-500.
- **A seedless `P` (tiles over the whole frame) failed its pilot gate**; the likely cause is
  `1 − generalizedDice` on all-background crops, so fix the loss before revisiting
  ([results/2026-09-17-seedless.md](results/2026-09-17-seedless.md)).
