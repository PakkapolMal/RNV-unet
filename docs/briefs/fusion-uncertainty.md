---
title: Fusing the Seedless P Where the Seeded P Is Uncertain
description: Pre-registered post-P arm. On images whose seeded P has a wide band of unsure pixels (high U), the seedless P is fused into the seeded P over the whole image (pixel max), before the unchanged pipeline. The U cut-off is a quantile set on the other folds' images. Grid, cross-fit, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-11-12-fusion-weak-seeds.md, results/2026-11-12-seedless-fallback.md, results/2026-11-10-train-budget-pmaps.md
status: pre-registered 2026-11-12, not yet run
---

# Fusing the Seedless P Where the Seeded P Is Uncertain

The image-level arm deferred by [fusion.md](fusion.md).

## Why

**The seedless-only lesion is not at weak seeds.**
- On the 66 images with points, 6.4 % of GT pixels are above 0.5 only in the seedless `P`
  ([2026-11-12](../results/2026-11-12-seedless-fallback.md)).
- The weak-seed arm found that under 90 % of weak seeds the seedless network sees nothing
  either ([2026-11-12 fusion](../results/2026-11-12-fusion-weak-seeds.md)).
- So those pixels lie mostly on lesions that confident seeds already reach. The likely places
  are frond edges and extensions the seeded outline stops short of.

**The weak-seed arm's own surface points the same way.** "Max then relax" gained more than
"seedless first" for the same seeds, because its ±128 px max also extended nearby lesions.
In-sample that was at most +0.0035, and the cross-fit kept under a third of it. This arm tests
the extension directly, over the whole image. It is a pointer, not evidence.

**Why gate it by uncertainty.** Fusing everywhere also extends lesions the seeded network
outlined well, which risks spill. The seeded `P` says, without GT, where its outline is unsure:
- **U** = (pixels with 0.2 < seeded `P` < 0.8) ÷ (pixels with seeded `P` ≥ 0.5), counted
  where the seeded network gave an answer (seeded `P` > 0.01);
- a low U means a decisive map, with almost every pixel near 0 or near 1;
- a high U means a wide band of unsure pixels around the lesions, typically a partial outline
  with weak fronds;
- across the 66 images with points, U is 0.16 at the 25th percentile, 0.32 at the median and
  0.70 at the 75th (GT-free).

**Seeds still decide what survives.** The fused `P` goes through the unchanged pipeline, and
claiming drops any seedless blob not connected to a claimed component.

**No training.** Both caches exist, out-of-fold with the same folds.

## Arm

For an image **with points** whose U is at or above the cut-off `c`:

    P_f = max(seeded P, seedless P where seedless P > τ_s, else 0)     over the whole image

Then run `cnngvf.maskFromProb` on `P_f`, unchanged: `tau` 0.5, relax 0.20 within 128 px,
`MinArea` 500, boundary claiming, hull 128.
- **U is computed on the seeded `P` alone,** before any fusion.
- **Images below the cut-off are unchanged,** and so are the 3 images with no points (stage 9).

**The cut-off is a quantile, set without the scored fold.** For held-out fold *f*, `c_f` is
the quantile of U over the images with points **outside** fold *f*. The images of fold *f*
with U ≥ `c_f` are fused. On adoption, the shipped `c` is that quantile over all 66 images,
written as a fixed number.

**Grid:** fraction fused q ∈ {top 10 %, top 25 %, top 50 %, **all**} × τ_s ∈ {0.5, 0.9},
plus **off**: 9 cells.
- "All" fuses every image with points. It answers whether the gate is needed at all.
- τ_s 0.9 admits only confident seedless pixels.

## Selection (cross-fitting)

- For each fold, choose the cell with the best mean IoU over the other four folds' images, and
  score it on the held-out fold. Both the choice and `c_f` come from the other folds.
- Ties within 0.001 go to off, then to the more conservative cell: smaller q first, then τ_s
  0.9 before 0.5.
- The fixed-cell surface is description only.

## Protocol

- 69 images (DRAC22 42, TUH 27), the two adopted 5-fold CV caches, whole-image scoring.
- **Sanity check:** off reproduces `reference/2026-11-12-seedless-fallback-69.csv` to ~1e-15.
- Paired per image against off. Percentile bootstrap on the mean Δ: 10 000 resamples,
  `rng(20260917)`, 95 % CI.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.
- **Code** (experiment):
  - `cnngvf.fuseUncertain(Pk, Ps, c, cfg)` builds `P_f`;
  - `cnngvf.seededUncertainty(Pk)` computes U;
  - cfg fields `Post.fuseQuantile` (0 = off) and `Post.fuseTauS`;
  - `run_cache` reads the seedless cache for images with points when fusion is on;
  - the runner `run_fusion_u_grid` sets the per-fold cut-offs and cross-fits.

## Prediction (written before running)

- **Fusing all images raises recall on Q4–Q5 and costs Q1–Q3 precision.** Its net ΔIoU is
  −0.003 to +0.006.
- **Gating helps.** The best cells fuse the top 25–50 % by U, at τ_s 0.9.
- **Cross-fitted ΔIoU +0.000 to +0.008.** The CI probably includes 0, and the folds split
  between neighbouring cells.
- **U ranks the poor seeded images, partly:** a Spearman correlation of −0.2 to −0.5 between U
  and seeded IoU over the 66 images. This uses GT, so it is a diagnostic only.
- I give the gate about 25 %. The weak-seed arm's in-sample gain shrank by two thirds under
  cross-fitting.

## Gate

Adopt a cell only if all three hold:

1. The cross-fitted ΔIoU against off has a 95 % CI lower bound above 0.
2. At least 4 of 5 folds choose the same cell. That cell ships, with `c` set on all 66 images.
3. **Precision guard:** on the cross-fitted masks, Q1–Q3 precision falls by no more than 0.01.
   Q4–Q5 recall is reported alongside.

**Otherwise** nothing changes. Tag the code `archive/fusion-uncertainty`, delete it, and
record the result.

**On adoption:**
- `cnngvf.fuseUncertain` becomes a stage before `maskFromProb`, with `c` and τ_s as defaults;
- `run_cache` reads both caches;
- a new reference CSV, `cfg.CV.reference`, the README table, and SPEC (the stage, the
  diagram, the fields).

## Report alongside, as description

- the 9-cell surface (IoU, precision, recall), per dataset;
- each fold's choice and its `c_f`;
- the fused images per cell, and how many pixels fusion added after claiming;
- precision and recall by quintile, cross-fitted against off;
- per-image movement: the largest gains and losses;
- the U against seeded IoU correlation (a GT diagnostic, never an input).

**Valid until either `P` changes.**
