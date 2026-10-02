---
title: MinArea × Fallback Disk Radius, Cross-fitted
description: Pre-registered 2D sweep of Post.MinArea and Post.seedRadius (including 0) on the current configuration, seed hull unchanged. Grid, prediction, selection rule and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-10-02-minarea.md, results/2026-09-16-reduce.md, results/2026-10-01-seed-hull-claim.md
status: executed 2026-10-02
---

# MinArea × Fallback Disk Radius, Cross-fitted

> **Executed brief.** Kept as written. Outcome: every fold chose (500, 15), gate 1 failed, the
> default stays. Predictions 1 and 3 were wrong: `421` / `217` are fired seeds, not disks.
> [../results/2026-10-02-minarea-disk.md](../results/2026-10-02-minarea-disk.md).

## Why

[2026-10-02-minarea](../results/2026-10-02-minarea.md) showed that `MinArea` and the disk are
coupled. A lone disk of radius r covers about πr² px (r 10 ≈ 317, 15 ≈ 707, 20 ≈ 1257,
30 ≈ 2827) and can only be claimed when that is at least `MinArea`. Above 707 px, the two
smallest lesions (`421.png`, `217.png`) went to IoU 0. `seedRadius` 15 was never measured
under relax. The 2026-09-16 disk-only sweep (no relax) preferred r 30–40.

`seedRadius` is read only by `cnngvf.seedDisks`, after `P`, so `run_cache` scores every
cell out-of-sample without retraining.

## Arm

No code change. `cfg.Post.MinArea` and `cfg.Post.seedRadius` vary; everything else is
`cnngvf.defaults()`: `tau` 0.5, relax 0.20 at 128 px, boundary claiming, `hullLink` 128 with
the fill-any-triangle rule. Every seed stays a hull vertex whether or not it got a disk.

**r = 0** needs no code: the disk is at most one pixel, always below `MinArea`, so it is never
claimable. Such a seed claims the nearest qualifying component by boundary distance, or
nothing. This differs from 2026-09-18 (the seed was dropped) and from 2026-10-01 RN (which
also changed the hull rule).

**Grid:** `MinArea` ∈ {250, 500, 750, 1000} × `seedRadius` ∈ {0, 10, 15, 20, 30}: 20 cells,
rectangular. 100 is left out because it was inert next to 250. Cells whose disk area is below
`MinArea` double as a check against r = 0.

## Selection (cross-fitting)

For each fold, choose the cell with the best mean IoU on the other four folds' images, and
score it on the held-out fold. Cells within 0.001 IoU of the best count as tied. The default
(500, 15) wins a tie; otherwise the smallest radius wins, then the smallest `MinArea`, which
is the tied cell that adds the fewest pixels. Fixed cells are a surface only, never a result.

## Prediction (written before running)

The disk serves only seeds relax could not serve: about 40, whose relax component fell under
`MinArea` or whose own `P` was at or below the floor. Stray detections fire with `P ≈ 1`, so
they never get disks.

1. **(500, r 0):** −0.002 to −0.006 on the surface. Recall falls on Q1, and `217.png` goes
   to 0: its whole 708 px prediction is the disk.
2. **(500, r 20):** within ±0.003 of the default. **(500, r 30):** −0.001 to −0.005, from the
   precision cost on small lesions. The disk-only optimum (r 30–40) does not carry over
   under relax.
3. **`MinArea` 750 / 1000 with r ≥ 20:** a disk of 1257 px or more partly rescues `421` and
   `217` (IoU ~0.4–0.5 instead of 0), but those cells stay below the default, where those
   images score 0.84–0.87.
4. **Cells whose disk is too small to claim** (r 10 at `MinArea` ≥ 500, r 15 at ≥ 750) come
   out nearly but not exactly equal to r 0 at the same `MinArea`: a disk joined to a
   neighbour by `imclose` can still be claimed. The size of the difference is reported.
5. **Cross-fitted:** at least 3 folds choose (500, 15) or a grid neighbour. ΔIoU vs the
   default lies in −0.004 to +0.002, 95 % CI including 0. **The gate fails, and (500, 15)
   stays.**

## Protocol

68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` via `run_cache`, whole-image scoring.
The comparator is the default (500, 15), which must reproduce `2026-09-30-hull-128.csv` to
~1e-15. The cross-fitted per-image IoU is paired against it. Percentile bootstrap on the mean
Δ, 10 000 resamples, `rng(20260917)`, 95 % CI; exact two-sided sign test, ties excluded.
Breakouts by dataset, fold, and rank-based GT-area quintile (`ceil(rank · 5 / n)`). For each
cell: images changed vs the default. For `421.png` and `217.png`: IoU in every cell.
Budget: about 11 min, about 30 s per cell.

## Gate

Adopt a cell other than (500, 15) only if all three hold:

1. cross-fitted ΔIoU 95 % CI lower bound > 0;
2. precision guard on the cross-fitted masks: mean precision over the three smallest GT-area
   quintiles falls by no more than 0.01 vs the default;
3. at least 4 of 5 folds choose the same cell, which becomes the default.

Otherwise (500, 15) stays, and the result is recorded as
`docs/results/2026-10-02-minarea-disk.md`.
