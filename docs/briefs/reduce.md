---
title: Reduced Pipeline — Implementation Instructions
description: Remove the GVF+DRLSE block, promote the existing threshold, and make the removal measurable as a paired inference-only ablation.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/CNN+GVF, MATLAB R2026a
status: done — executed 2026-09-03 to 2026-09-16
---

# Reduced Pipeline — Implementation Instructions

> **Executed brief.** Kept as written. Outcome:
> [../results/2026-09-16-reduce.md](../results/2026-09-16-reduce.md). The level set it made
> switchable was deleted on 2026-09-25; code at tag `archive/pre-cleanup`.

Read [UNCLUSTERED.md](../SPEC.md) first. It is the source of truth. Where this
document and UNCLUSTERED.md disagree, UNCLUSTERED.md wins and you should stop and say so.

## Objective

Replace stages 5–9 (force field, GVF diffusion, edge stopper, DRLSE) with a threshold on
`P` that the pipeline **already computes** inside `cnngvf.initPhi`.

**This is not a deletion task.** Build it as a switchable path so the removal can be
measured paired against the current pipeline on identical `P`. Deletion happens later,
after the numbers, in a separate change.

**No network changes. No retraining. Zero trainable parameters are added or removed.**
Every stage being cut has no weights. The comparison must be inference-only on cached `P`,
which is what makes it perfectly paired and free of training noise.

## New pipeline overview

```
image ─► loadChannels ──► intensity ──┐
                                      ├─► 512² seed-centred patch ─► U-Net ─► P
seed (x,y) ─► Gaussian heat-map ──────┘                                       │
                                                                              ▼
                                                                        BW = P > tau
                                                                              │
                          mask ◄─ seedMasks ◄─ imfill ◄─ imclose ◄────────────┘
                                (minArea 500)
```

Against the current pipeline, gone: `buildForce`, `gvfField`, edge stopper `g`, `drlse`,
the `muDR` distance-regularization term, the `alfa` balloon (already 0), and the **first**
of the two `seedMasks` calls — with no level set there is nothing to initialize, so
initialization-time claiming and post-evolution claiming collapse into one call.

Nine stages become five. Post-processing order is unchanged on purpose: `imclose` →
`imfill` → `seedMasks` is exactly what the frozen `threshold P > 0.60` arm used, so the
new path is directly comparable to that record.

## What to change

### 1. `+cnngvf/defaults.m` — add a `Post` block

```matlab
cfg.Post.mode         = "levelset";   % "levelset" | "threshold"
cfg.Post.tau          = 0.5;          % PRE-REGISTERED. Do not tune before measuring.
cfg.Post.diskFallback = false;        % plant seedRadius disks at seeds with no P > tau
cfg.Post.imclose      = true;         % imclose(disk 2) before imfill
```

`cfg.Post.mode = "levelset"` must reproduce today's behaviour byte-for-byte. Verify this
before touching anything else — if the refactor moves a single pixel in level-set mode,
the pairing is broken and every downstream number is worthless.

Keep `cfg.Drlse.tau` as-is. Do **not** repoint `initPhi` at `cfg.Post.tau`. Two separate
names, and assert `cfg.Post.tau == cfg.Drlse.tau` whenever `mode == "levelset"`, so the
level-set arm cannot silently drift when someone sweeps `Post.tau`.

### 2. New file `+cnngvf/maskFromProb.m`

The threshold path. One function, same shape as every other stage — takes `cfg`.

```
function [mask, perSeed] = maskFromProb(P, seeds, cfg)
```

Order of operations, exactly:

1. `BW = P > cfg.Post.tau`
2. If `cfg.Post.diskFallback` — for each seed whose 8-connected neighbourhood in `BW` is
   empty, stamp a disk of radius `cfg.Drlse.seedRadius` (15) at that seed. Reuse the
   radius constant from wherever `initPhi` reads it; do not hardcode 15 twice.
3. If `cfg.Post.imclose` — `imclose(BW, strel('disk', 2))`
4. `imfill(BW, 'holes')` — always, unconditional. The target is `FilledGT`.
5. `cnngvf.seedMasks(BW, seeds, minArea = 500)` — once, not twice.

Return contract is identical to the level-set path: `mask` logical `[1024 1024]`,
`perSeed` a cell array with one entry per seed. `run_eval` recovers seeds from per-seed
mask filenames, so the per-seed outputs are not optional.

### 3. `+cnngvf/segmentImage.m` — branch on `cfg.Post.mode`

```
"levelset"  : buildForce → gvfField → initPhi → drlse → imclose → imfill → seedMasks
"threshold" : maskFromProb
```

Branch as high as possible — immediately after `inferProbMap` returns `P`. Everything
upstream (`loadChannels`, patch extraction, heat-map, `inferProbMap`, raised-cosine
stitching) is shared and untouched.

In threshold mode the GVF mex must not be called at all. Assert it.

### 4. Cache `P` — the change that saves 18 hours per variant

Today nothing persists `P`, so every downstream variant costs a full cross-validation.
Fix that in `run_cv` and `run_batch`.

| item | requirement |
|---|---|
| dtype | `single`. **Not `uint8`** — 1/255 quantization is coarse against a fine `tau` sweep. `uint16` acceptable if size matters. |
| size | 68 × 1024² single ≈ 285 MB per CV run. Acceptable. |
| key | image name (unique across datasets — it is the manifest join key) |
| metadata | **fold index**, arm name, network file, timestamp, CSV mtimes |
| location | `result/<arm>/probcache/` |

Also persist **the five fold networks keyed by fold index**. `trainNet` currently archives
by timestamp, which cannot reconstruct which network owned which fold. Without this the
cache is unauditable.

**Leakage guard, non-negotiable:** on load, assert every cached `P` carries a fold index
whose held-out set contains that image. A cached `P` produced by a network that trained on
its own image is in-sample and must fail loudly, not score.

### 5. New runner `run_ablate.m`

Loads cached `P` and scores configurations without a GPU. Seconds, not hours.

Run the three switches as a **2³ factorial, all eight cells** — not sequentially. They
interact: `imclose` was justified as bridging gaps *the level set leaves*, and the disk
fallback was justified as giving a missed lesion *something to grow from*. Sequential
ablation will attribute their interaction to whichever ran first.

| # | mode | diskFallback | imclose |
|---|---|---|---|
| 1 | levelset | true | true | ← current pipeline, the reference cell |
| 2–8 | all remaining combinations | | |

Output per cell: mean/median IoU, F1, precision, recall, per-fold breakout, and the
`Dataset` column preserved so DRAC22/TUH can be split afterwards.

Paired statistics against cell 1: Wilcoxon signed-rank, exact sign test, bootstrap 95% CI
on the delta. Because `P` is identical across all eight cells, training variance is exactly
zero — these tests are unusually powerful here, and a small consistent delta is real.

### 6. `run_steps.m` — stage dumps

In threshold mode, skip `06_force` through `11_levelset_raw` rather than writing blanks,
and emit `11_binary_raw.png` (post-threshold, pre-morphology). Keep the `12_after_morph` /
`13_final_mask` numbering stable so the two modes' dumps line up side by side.

### 7. `+cnngvf/seedMasks.m` — fix the claiming rule before measuring

`seedMasks` claims by nearest **centroid**, unlimited distance. Training labels by nearest
**boundary** within 20 px. Two rules for the same question, and the centroid rule is the
bug `RNV-Claude/main`'s `selectSeedComponent` was fixed for — it misassigns seeds around
large concave lesions.

This matters more under the reduced pipeline, not less: components now come straight off
thresholded `P` and will be more fragmented than post-DRLSE components, so the claiming
rule carries more load.

**Fix it first, in its own commit, and re-run cell 1 afterwards.** If the fix lands in the
same change as the removal it confounds the result.

### 8. `test_cnngvf.m` — extend the eight checks

Add:

1. `Post.mode = "levelset"` reproduces the pre-change mask exactly on `458.png`.
2. `P` cache round-trips at `single` fidelity — no `uint8` path anywhere.
3. Every cached `P` has a fold index consistent with its held-out set.
4. Threshold mode never enters `gvfField` (assert the mex is not called).
5. `maskFromProb` returns the same `[mask, perSeed]` contract as the level-set path.
6. `assert(cfg.Post.tau == cfg.Drlse.tau)` holds in level-set mode.

Keep all eight existing checks passing, including the heat-map peak assertion — that one is
silent when wrong and is unrelated to this change.

## What NOT to touch

Nothing in this list changes. If a change here seems to require touching one of these,
stop and ask.

### Training — entirely untouched

| | |
|---|---|
| network | `unet(..., 2, EncoderDepth=4)`, weights, encoder depth |
| loss | `1 - generalizedDice` |
| optimizer | Adam, lr 1e-3, 30 epochs, `OutputNetwork="best-validation"` |
| channels | 2 (intensity + heat-map). `loadChannels` derives the count — never hardcode it. |
| patches | `Patch.size = 512`, `jitter = ±32`, `perSeed`, `balanceByImage = true` |
| seeds | `Patch.seedTolerance = 20` for training, **all 874 at inference** |
| batching | `Net.miniBatch = 2`, `Net.inferBatch = 4` |

**Do not retrain to evaluate this change.** Retraining destroys the pairing and buys
nothing — no trainable parameter is involved.

### Evaluation protocol

- `EvalMode = "image"`. Union mask vs whole GT. Never per-segment.
- Same folds, pooled across datasets, area-stratified round-robin.
- `299.png` and `445.png` still score ≈ 0 and still count. Do not exclude them.
- Integrity assertions stay: 68 rows, 68 unique, every fold scores exactly its held-out set.

### Shared upstream stages

`manifest`, `loadPoints`, `arm`, `rnv.loadImage`, `rnv.loadSeeds`, seed heat-map
(`sigmaSeed = 32`), `inferProbMap`, and the raised-cosine stitching window.

Seeds are `[cx cy]` = (column, row), read positionally. A transposed seed is silent.

### Do not delete yet

Leave in place until the ablation is reported: `buildForce`, `gvfField`, `initPhi`,
`drlse`, `cfg.Denoise`, `cfg.Vessel.enabled`, `Force.source = "blend"`,
`EvalMode = "segment"`, `lib/Fast-Gradient-Vector-Flow`. Dead-code removal is a separate,
later change. The frozen arms must stay reproducible.

### Do not add

No new preprocessing, no new channels, no loss change, no TTA, no sliding-window inference,
no post-processing beyond what is specified above. The single permitted addition is the
threshold — and it is not really an addition, since `initPhi` already computes
`BW0 = P > tau`.

## `tau` is now a decision variable — handle it carefully

Before, `tau` only seeded `phi0` and the level set washed it out. Now it sets the output
directly. **Sweeping `tau` on the scored images and quoting the best is leakage** — the same
class of error as quoting `Split.Val` as held-out, which inflated one estimate by +0.089.

Report three numbers, labelled, never conflated:

| | `tau` chosen how | quotable |
|---|---|---|
| **headline** | pre-registered `0.5`, fixed, all folds | yes |
| sensitivity | full sweep, mean F1 vs `tau` | yes, **as a curve** — never as its maximum |
| ceiling | oracle per-image `tau` | no — upper bound only |

`0.5` is the defensible prior: it is the softmax decision boundary, and the level set was
already documented as largely an iso-contour finder for `P ≈ 0.5`.

Do **not** select `tau` per fold on `cfg.Split.Val`. Legitimate in principle, but that set
is ~6 images and already carries checkpoint selection; a second selection on it will fit
noise.

A flat sensitivity curve across 0.4–0.65 means fixed `tau` is safe. A sharp one means a
calibration problem in `P`, not a threshold problem — record it and stop, do not patch it
with a tuned `tau`.

## Expected result — write this down before running

From the frozen record, identical post-processing, 17 images:

| | IoU | precision | recall | F1 |
|---|---|---|---|---|
| threshold `P > 0.60` | 0.5239 | 0.7393 | 0.6808 | 0.7088 |
| GVF+DRLSE | 0.5403 | 0.7413 | 0.7054 | 0.7229 |

Expect roughly **−0.014 F1** from the removal, with the loss almost entirely in recall
(precision delta is 0.002). Precision is the documented weak side, so the level set is
paying nothing on the axis that hurts.

If the measured delta is far from this, something is wrong with the port — check that
level-set mode still reproduces exactly, and that `tau` matches between the arms.

Note: the oracle figure 0.5068 in UNCLUSTERED.md is from the `seedMasks` ablation and has
**no** `seedMasks`. It is not the threshold comparator. 0.5239 is.

## Order of work

1. Fix `seedMasks` claiming (centroid → boundary). Own commit.
2. Add the `Post` block and the `segmentImage` branch. Verify level-set mode is byte-identical.
3. Add `P` caching + fold-keyed networks + the leakage guard. Extend `test_cnngvf`.
4. Run `run_cv("DRACTUH_unclustered")` once. This is also the first combined-domain number,
   which is owed regardless.
5. Write `maskFromProb` and `run_ablate`. Run all eight cells at pre-registered `tau = 0.5`.
6. Produce the `tau` sensitivity curve separately, as a curve.
7. Report. Only then open a deletion change.

## Reporting rules — unchanged, still binding

1. Cross-validated numbers only.
2. Never quote a single-split "validation" score as held-out.
3. Do not compare across protocols.
4. Do not compare across domains — state the domain with every number. 17, 41 and 68 images
   are three different questions.
5. Every score is tied to the CSVs in place when it was measured. Record their mtimes in the
   cache metadata.