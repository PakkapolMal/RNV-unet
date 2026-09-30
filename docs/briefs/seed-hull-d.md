---
title: Seed Hull, Cross-fitted Link Distance
description: Pre-registered plan to re-run the seed triangle fill with a larger pairwise bound, chosen by cross-fitting over a grid. Prediction, grid, selection rule and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: briefs/seed-hull.md, results/2026-09-30-seed-hull.md, results/2026-09-30-relax-grid.md
status: pre-registered 2026-09-30, not yet run
---

# Seed Hull, Cross-fitted Link Distance

Follow-up to [seed-hull.md](seed-hull.md). Read its record,
[../results/2026-09-30-seed-hull.md](../results/2026-09-30-seed-hull.md), first.

## Why run it again

At a pairwise bound of `d` = 64 px (2 · `sigmaSeed`), the triangle fill was inert: +0.0002
IoU, because only 15 % of Delaunay triangles qualified and 97 % of their area was already
in the mask. The fill starts to act only at larger `d`. The table below is GT-free, on the
cached 5-fold CV `P` of the 68-image domain:

| `d` (px) | pixels added | % of mask | images changed |
|---|---|---|---|
| 64 | 4 104 | 0.14 % | 14 |
| 96 | 41 149 | 1.37 % | 26 |
| 128 | 176 097 | 5.87 % | 30 |
| 192 | 367 722 | 12.26 % | 36 |
| 256 | 545 552 | 18.18 % | 39 |

**Hypothesis.** The closing arm recovered inter-frond gaps on the largest lesions (+0.061
IoU on the top quintile), but lost more everywhere else. The triangle fill adds only
*inside* the seed cloud and never touches a lesion's rim. At a larger `d` it may keep the
large-lesion gain at a smaller precision cost.

**Risks.** As `d` grows, the union of qualifying triangles approaches the convex hull of the
seeds. It then bridges genuine concavities and separate lesions, and from about 128 px it
starts linking detector errors (7 lie 100–200 px from any lesion).

## Why cross-fitted

The 2σ derivation behind 64 does not extend to any larger value. A larger `d` would be
chosen after seeing 64 fail and after looking at `nv (17)`, which makes it a threshold tuned
on the scored images. It is therefore cross-fitted, never picked.

## Arm

`cnngvf.seedHull` in triangle mode only, restored from tag `archive/seed-hull`, applied after
claiming as before. The pairwise bound becomes its own field, `cfg.Post.hullLink` (px),
instead of being derived from `sigmaSeed`. A triangle of the seeds' Delaunay triangulation
is filled when all three of its seeds are pairwise within `hullLink`. Holes enclosed by
filled triangles are filled.

**Grid:** `hullLink` ∈ {64, 96, 128, 192, 256}. The cap of 256 is `Patch.size` / 2, the
farthest the network sees from a seed.

## Selection (cross-fitting)

The rule is the one [the relax grid](../results/2026-09-30-relax-grid.md) used. For each fold,
choose the `d` with the best mean IoU on the other four folds' images, then score that `d`
on the held-out fold. Values within 0.001 IoU of the best count as tied, and the **smallest**
tied `d` wins, since it adds the least. The fixed-`d` results are reported as a surface
only, never as a result, because each of them is tuned on the images it scores.

## Prediction (written before running)

- Most folds choose `d` in 128–192.
- Cross-fitted ΔIoU vs `"none"` lies between +0.003 and +0.010, with the 95 % CI touching 0.
- The precision guard holds, because the fill never touches a lesion's rim.

## Protocol

68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` via `run_cache`, whole-image scoring,
`tau` 0.5, relax 0.20. The comparator is `"none"`, which reproduces the reference
`2026-09-17-relax-0.20.csv`. The cross-fitted per-image IoU is paired against it. Percentile
bootstrap on the mean Δ, 10 000 resamples, `rng(20260917)`, 95 % CI (one arm, so no
correction); exact two-sided sign test, ties excluded.

## Gate

Adopt only if all three hold:

1. cross-fitted ΔIoU 95 % CI lower bound > 0;
2. precision guard on the cross-fitted masks: mean precision over the three smallest
   GT-area quintiles falls by no more than 0.01;
3. at least 4 of 5 folds choose the same `d`, which becomes the default.

Otherwise reject: tag `archive/seed-hull-d`, delete the switch, record the result.

Report alongside, as description rather than evidence: the fixed-`d` surface, GT-area
quintile and per-dataset breakdowns, the folds' choices, and the five motivating images.
