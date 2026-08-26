# RNV Segmentation from Unclustered Localization Points — Specification

The best-performing configuration in this repo, with the measurement behind every
non-obvious choice. Implementation: [`../CNN+GVF/`](../CNN+GVF/). Theory: [CLAUDE.md](CLAUDE.md).
Original design: [spec.md](spec.md).

## Result

### Current configuration, full dataset

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

All values from [`+cnngvf/defaults.m`](../CNN+GVF/+cnngvf/defaults.m); `run_train_unclustered`
overrides `perSeed`.

### Data

| item | value |
|---|---|
| images | 1024×1024 OCTA en-face, `RNV-DRAC22NV/` — **41 with ground truth**, 39 of which carry seeds |
| training seeds | `DRAC_RNV_Points_Unclustered.csv` — 281 rows, 232 usable (49 fall outside every lesion) |
| inference seeds | same |
| labels | the `FilledGT` connected component containing each seed |
| scoring | union mask vs whole `FilledGT/<stem>.png` (`EvalMode = "image"`) |

The dataset grew from 17 to 41 ground-truth images during development. `cnngvf.prepareData`
and `cnngvf.cvFolds` both scan the ground-truth directory, so nothing needs updating when
images are added — but **`SegmentFilledGT/` still covers only the original 17**, so
`EvalMode = "segment"` fails on the newer ones until `separate_segments.py` is re-run. The
unclustered arm scores whole-image and is unaffected.

Any figure measured before the dataset grew is not comparable with one measured after. The
cross-validated 0.4163 was measured on 17 images.

Labels are computed by component lookup, not read from `SegmentFilledGT/`. Verified to
reproduce all 35 segment files exactly, and it works for point sets those files are not keyed
to. Seeds landing outside every lesion (24 of 185) are dropped from training — labelling them
against pure background would teach the network to ignore its own heat-map.

**Scoring must be whole-image, not per-segment.** Several detections land on one lesion, so
per-segment scoring would count that lesion once per detection and weight lesions by how many
detections they attracted. Whole-image scoring also charges for lesions missed entirely.

### Input channels

| channel | source |
|---|---|
| 1 | intensity, median-filtered `[3 3]` |
| 2 | Frangi vesselness, `fibermetric` scales `[2 4 6 8]` |
| 3 | Gaussian seed heat-map, `sigmaSeed = 32` |

Small vesselness scales because RNV vessels are thin; large scales blur the structure away.

### Patches and sampling

| parameter | value | why |
|---|---|---|
| `Patch.size` | **512** | 13 of 35 GT segments need a window > 256 px, and each scored badly at 256. At 512, 32 of 35 are fully covered. |
| `Patch.jitter` | ±32 px | crop offset; the heat-map marks the seed, not the patch centre |
| `Patch.perSeed` | **4** | set by `run_train_unclustered`; 4 crops × 137 seeds matches the clustered arm's 16 × 30 |
| `Patch.balanceByImage` | **true** | equal patches per *image*, not per seed |

**`balanceByImage` matters here and nowhere else.** Detection counts track lesion size, so
sampling uniformly per seed gave `1115.png` 39% of the epoch and the top two images 54%. The
budget is drawn **with replacement from each image's own seeds** — taking the first *N* would
pin a seed-rich image to the same subset in CSV order for the whole run.

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
test_cnngvf                % 6 checks, synthetic, no dataset or model needed

run_train_unclustered      % -> models/rnvNet_unclustered.mat  (~2.5 h)
run_batch_unclustered      % -> result/unclustered/            (~4 min)

run_eval(fullfile(cfg.Paths.Output,'unclustered'), cfg)   % re-score, ~1 s, no GPU

run_cv('unclustered')      % the defensible number (~8 h)
run_cv('clustered')        % the paired baseline   (~10 h)
```

**MATLAB R2026a.** Needs Image Processing, Deep Learning, Computer Vision and Parallel
Computing. R2025b on this machine has Image Processing only and fails immediately with a
message from `cnngvf.defaults`.

## Reporting rules

1. **Quote cross-validated numbers only.** `run_batch` scores all 17 images but 13 trained the
   model.
2. **Never quote a single-split "validation" score as held-out.** `OutputNetwork="best-validation"`
   selects the checkpoint on `cfg.Split.Val`, so those images are not held out. Doing this
   inflated one estimate by +0.089 (`461.png`: 0.731 single-split vs 0.307 cross-validated).
3. **Do not compare across protocols.** A CV number and a single-split number are not
   comparable in either direction.
4. **Every score is tied to the CSV in place when it was measured.** The localization files
   have been revised three times and live outside version control.

## Known limits and next steps

- **17 ground-truth images.** Thin for a U-Net; [spec.md](spec.md) suggests pretraining on
  ROSE or OCTA-500.
- **Precision is the weak side** (0.721 vs the baseline's 0.797). Four images regress under
  CV, worst `573.png` (−0.200) and `641.png` (−0.123); the failure mode is over-segmentation.
- **24 of 185 detections lie outside every lesion.** Dropped in training, but still generating
  patches at inference where they can only add false positives. Filtering them at inference is
  untried — though an *oracle* filter on the single-split run was worth only +0.015, so the
  ceiling is low.
- **Post-processing is exhausted.** The pipeline already beats an oracle that picks the best
  threshold per image (0.5315 vs 0.5068). Effort belongs in the probability map.
- **The balanced-sampling gain is unverified.** Both CV arms used it, so they measure the
  unclustered effect, not the balancing effect. Isolating it needs its own CV pair.
- **Four segments exceed even a 512 px window.** Only full-image tiled inference reaches them.
