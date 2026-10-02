---
title: MinArea, Cross-fitted
description: Pre-registered sweep of Post.MinArea on the current configuration (relax 0.20 at 128 px, boundary claiming, seed hull 128). Grid, prediction, selection rule and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-09-30-relax-grid.md, results/2026-09-30-seed-hull-d.md, results/2026-10-01-seed-hull-claim.md
status: pre-registered 2026-10-02, not yet run
---

# MinArea, Cross-fitted

## Why

`cfg.Post.MinArea = 500` has never been measured. The 2026-09-17 refine brief put `minArea`
sweeps out of scope. The one value does two jobs:

1. **Relax** (`cnngvf.seedFallback`): if the component relax finds under a seed is below
   `MinArea`, the seed gets the disk instead. 40 seeds took this path (2026-09-17).
2. **Claiming** (`cnngvf.seedMasks`): a component below `MinArea` cannot be claimed. A seed
   standing on one claims the nearest qualifying component by boundary distance.

**The disk interaction.** The fallback disk at `seedRadius` 15 is about 707 px. Above that,
a lone disk is unclaimable, so `MinArea` > ~710 acts like dropping the disk for isolated
seeds. Dropping the disk has lost twice: −0.0006 (2026-09-18) and arm RN, −0.0045
(2026-10-01). The smallest ground-truth lesion in the domain is 601 px.

## Arm

No code change. `MinArea` alone is varied, keeping one shared value for both jobs.
Everything else is `cnngvf.defaults()`: `tau` 0.5, relax 0.20 at 128 px, boundary claiming,
`hullLink` 128.

**Grid:** `MinArea` ∈ {100, 250, 500, 750, 1000} px. 500 is the default. 750 and 1000 lie
above the disk's area on purpose, to measure the disk interaction.

## Selection (cross-fitting)

The rule [the relax grid](../results/2026-09-30-relax-grid.md) used. For each fold, choose the
value with the best mean IoU on the other four folds' images, and score it on the held-out
fold. Values within 0.001 IoU of the best count as tied, and **the default (500) wins a tie**,
then the value closest to it. Fixed-value results are a surface only, never a result.

## Prediction (written before running)

- 750 and 1000 lose recall on the smallest GT-area quintile, as no-disk did. Each scores
  0.002–0.006 below 500 on the surface.
- 100 and 250 let relax keep components of 100–500 px instead of a disk, and make small
  thresholded blobs claimable: on Q1–Q2, precision rises a little and recall falls a little.
  Within ±0.002 of 500 on the surface.
- Cross-fitted ΔIoU vs the default: −0.003 to +0.002, 95 % CI including 0. At least 4 folds
  choose 500 or 250. **The gate fails and 500 stays.**

## Protocol

68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` via `run_cache`, whole-image scoring.
The comparator is the default (500), which must reproduce the reference
`2026-09-30-hull-128.csv` to ~1e-15. The cross-fitted per-image IoU is paired against it.
Percentile bootstrap on the mean Δ, 10 000 resamples, `rng(20260917)`, 95 % CI; exact
two-sided sign test, ties excluded. Breakouts by dataset, fold, and rank-based GT-area
quintile (`ceil(rank · 5 / n)`). For each value: the number of images whose mask changes
vs 500.

## Gate

Adopt a value other than 500 only if all three hold:

1. cross-fitted ΔIoU 95 % CI lower bound > 0;
2. precision guard on the cross-fitted masks: mean precision over the three smallest
   GT-area quintiles falls by no more than 0.01 vs the default;
3. at least 4 of 5 folds choose the same value, which becomes the default.

Otherwise 500 stays, now measured, and the result is recorded as
`docs/results/2026-10-02-minarea.md`.
