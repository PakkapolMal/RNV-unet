# RNV Segmentation from Unclustered Localization Points — Specification

The best-performing configuration in this repo, with the measurement behind every
non-obvious choice. Implementation: [`../CNN+GVF/`](../CNN+GVF/). Theory: [CLAUDE.md](CLAUDE.md).
Original design: [spec.md](spec.md).

## Result

**5-fold cross-validated, every image scored out-of-sample, both arms on identical folds:**

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

## Specification

All values from [`+cnngvf/defaults.m`](../CNN+GVF/+cnngvf/defaults.m); `run_train_unclustered`
overrides `perSeed`.

### Data

| item | value |
|---|---|
| images | 1024×1024 OCTA en-face, `RNV-DRAC22NV/` |
| training seeds | `DRAC_RNV_Points_Unclustered.csv` |
| inference seeds | same |
| labels | the `FilledGT` connected component containing each seed |
| scoring | union mask vs whole `FilledGT/<stem>.png` (`EvalMode = "image"`) |

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
