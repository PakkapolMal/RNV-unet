---
title: Seedless Fallback for Images with No Localization Points
description: Pre-registered arm. A seedless whole-frame network, trained on the adopted recipe, segments only the images with zero localization points. Its threshold and component rule are cross-fitted on the images with 1–3 points, never on the three target images. Prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/colab (seedless trainer), Colab L4 (user runs), RNV-Claude/cnn (the no-point rule, its cross-fit)
depends-on: results/2026-11-11-paper-recipe.md, results/2026-11-11-tau-floor-refit.md, results/2026-09-17-seedless.md
status: executed 2026-11-12
---

# Seedless Fallback for Images with No Localization Points

> **Executed brief.** Kept as written. Outcome: all three gates hold. The targets score 0.550
> mean IoU, the proxy 0.467 cross-fitted, and the 66 other images are unchanged; the headline
> goes from 0.6125 to 0.6364: [../results/2026-11-12-seedless-fallback.md](../results/2026-11-12-seedless-fallback.md).

Supersedes arm C of the local brief `paper-adoption.md`. This brief adds the threshold and
component rule, which that one fixed at `tau`.

## Why

**Three images have no localization point at all:** `082`, `217` and `445`. They are DRAC22,
with GT of 2 408, 601 and 4 052 px. The pipeline gives them `P = 0` by construction, so they
score 0. The 66 images with points average 0.6403 IoU, and all 69 average 0.6125 (5-fold CV,
[2026-11-11](../results/2026-11-11-paper-recipe.md)). Each 0.1 of IoU on one of the three adds
0.0014 to the headline.

**"This image has no points" is known at inference, so it is not leakage.** A rule that
triggers on it cannot lower any of the three, which all score 0 today, and it touches no other
image.

**What it cannot reach:** `299`, `573` and `664` have points, all off-lesion. Recognising those
points as wrong needs GT.

**Why now:** the 2026-09-17 seedless pilot failed. That record named the loss as the first fix,
because `1 − generalizedDice` punishes every empty crop. The adopted recipe (CE + batch Dice,
pretrained encoder) is that fix ([2026-11-11](../results/2026-11-11-paper-recipe.md)).

## The network

`rnvtorch.py train-seedless`, same bundle, same folds, one run (`--rng 0`):

| | value |
|---|---|
| network | `SmpNet(cfg.Net.encoder, cin=1)`: the adopted network with intensity only, no heat-map |
| input | the whole 1024² frame, standardised per channel |
| target | FilledGT, `label > 0`: every lesion |
| training images | every image outside the fold's test images, **including the point-less ones** (labels need no points) |
| samples per epoch | 10 augmented copies of each training image |
| augmentation | flips p 0.5 each; affine p 0.75 (rotation ±15°, scale 0.9–1.1, shear ±5°, **translation ±5 %**); CLAHE p 0.5 (clip 2.0, **8 × 8 tiles**, 128 px on the frame as on the seeded patch) |
| loss, optimiser | `ce_dice_loss`; Adam 1e-4 → 1e-6 cosine, L2 1e-4 on conv weights, bf16 |
| minibatch | 4 (if it runs out of memory: 2, recorded as a deviation) |
| epochs | 60, last epoch, no validation |
| output | held-out `P` for every test image, `probCache` format, `<run_root>/probcache` |

**Code:**
- `affine_clahe` gains `tiles` and `shift` arguments. With the defaults (4 and 0) it draws
  exactly as now, so the seeded trainer is unchanged.
- `FrameSet` is the whole-frame dataset. `train_seedless` and `infer_frame` reuse
  `train_fold`'s optimiser, schedule, checkpoint and resume code. The shared part becomes one
  function, not a copy.

## The no-point rule, and its threshold

For an image with **zero points**:
1. read its seedless `P` (`cnngvf.probCache`, leakage guard included);
2. `BW = P > tauSeedless`, then `imclose` and `imfill` as in `maskFromProb`;
3. drop components below `MinArea` (500 px);
4. **keep:** either all remaining components, or only the one with the largest P mass
   (Σ`P` over the component).

**Why keep top-1 is a candidate:** a whole-frame network fires on normal vasculature too. The
seeded pipeline discards such blobs by claiming, and with no seeds, the strongest blob is the
only analogue. It relies on one domain fact: every image in this domain contains RNV (the
datasets are RNV-positive by construction). That is a property of the domain, not of an
image's GT, but it would not hold on a screening population. That caveat goes in SPEC if it
is adopted.

### The threshold suggestion: cross-fit on the few-point images

**The two choices are `tauSeedless` and keep.** Neither can be chosen on the three targets:
- with n = 3, a choice made on them would be scored on them;
- their GT is the evaluation.

**The proxy: images with 1–3 points**, scored with the no-point rule as if they had none (their
points ignored).
- **Selected by point count,** which is known at inference. No GT is used to select it.
- **It resembles the targets:** 18 images, 15 DRAC22, median GT 4 157 px, against 13 544 px
  domain-wide. A point-poor image tends to hold a small lesion, and the targets are small.
- **Their seedless `P` is out-of-fold,** like every cached `P`.

**Grid:** `tauSeedless` ∈ {0.30, 0.40, **0.50**, 0.60, 0.70, 0.80, 0.90} × keep ∈ {**all**,
top-1}: 14 cells. The default is (0.50, all).

**Selection, per fold:**
- a target image in fold *f* gets the cell with the best mean IoU over the proxy images
  **outside** fold *f*;
- ties within 0.001 go to the default, then to the cell nearest it (`tau` first, then all
  before top-1).

**What ships on adoption** is the cell chosen on all 18 proxy images (Reporting rule 6).

The proxy's own cross-fitted score (each fold's proxy images scored with the cell chosen on
the other folds' proxy images) is the honest estimate of what the rule does on images like
the targets. It rests on 18 images instead of 3.

## Verification

- `test_cnngvf` passes.
- **Nothing else moves:** with the rule on, `run_cache` reproduces the reference on the 66
  images with points, max |dIoU| ~1e-15.
- **The leakage guard holds:** it refuses a seedless `P` stamped with the wrong fold.
- **Smoke** (`test_parity.py stage3-seedless`): fold 1, 2 epochs with a resume. Checks:
  - the frame-level affine moves the label with the image;
  - translation stays within ±5 %;
  - `P` is written for the 14 test images and lies in [0, 1];
  - the projected CV time.
- **Stall rule** (SPEC, Network): the relative form, before scoring.

## Cost

- **Training:** not measured. 550 frames per epoch at 1024², batch 4, 60 epochs. Estimate
  about 1 h per fold, 5 h per CV on an L4. The smoke gives the number.
- **The cross-fit:** seconds, off the cache.

## Prediction (written before running)

- **No fold stalls.**
- **Proxy cross-fitted IoU 0.15–0.35.** Whole-frame false positives dominate small-lesion
  images.
- **The folds choose `tauSeedless` ≥ 0.6 and top-1.** False positives are the failure, and
  both choices cut them.
- **Three-target mean IoU 0.00–0.25,** so a headline change of +0.000 to +0.011. `217`
  (601 px, the smallest lesion in the domain) most likely stays at 0.
- **Diagnostic, all 69 images with the rule applied to every image:** IoU 0.35–0.48,
  below the seeded pipeline on both datasets.
- I give the gate about 30 %.

## Gate

Adopt only if all three hold:

1. **Gain:** the three-target mean IoU is at least **0.115**. That is +0.005 on the headline,
   the least that justifies a second network in the pipeline.
2. **It generalises:** the proxy's cross-fitted mean IoU is at least **0.20**. A rule that
   scores on 3 images but not on the 18 that resemble them is luck.
3. **Isolation:** the 66 images with points are unchanged, to about 1e-15.

This is a floor, not a significance test. n = 3 cannot carry one, and the rule cannot lower an
image. The training noise floor does not apply, because the other 66 images' `P` is the
reference's.

**On adoption:**
- the rule becomes a pipeline stage (`cnngvf.noPointMask`), with `cfg.Post.noPointTau`,
  `noPointKeep` and the seedless cache path as defaults;
- SPEC gets the stage and the RNV-positive caveat;
- a new reference CSV, `cfg.CV.reference`, and the README table;
- `run_batch` and `run_steps` need a full-data seedless network: a separate change.

**On rejection:**
- tag `archive/seedless-fallback` and delete the code;
- keep the seedless `P` cache, outside git, for later fusion work;
- record the proxy surface, which answers what a seedless network scores on small lesions in
  this domain.

## Report alongside, as description

- the 7 × 2 proxy surface;
- each target image's chosen cell, predicted area and IoU;
- the all-69 diagnostic per dataset;
- the seedless network's `P` against the seeded `P` on the 66 images with points: pixel
  agreement inside and outside GT. This is the groundwork for any later fusion arm.
