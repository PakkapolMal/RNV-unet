---
title: tau × relaxFloor, Re-fitted on the Adopted Recipe's P
description: Pre-registered joint cross-fit of Post.tau and Post.relaxFloor on the cached CV P of the 2026-11-11 recipe (pretrained EfficientNet-B3 U-Net, CE + Dice). Grid, selection rule, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-11-11-paper-recipe.md, results/2026-11-05-tau-crossfit.md, results/2026-09-30-relax-grid.md, results/2026-09-16-reduce.md
status: executed 2026-11-11
---

# tau × relaxFloor, Re-fitted on the Adopted Recipe's P

> **Executed brief.** Kept as written. Outcome: gates 1 and 2 failed. Cross-fitted ΔIoU is
> +0.0004, CI [−0.0027, +0.0033], and three of five folds agree. Both thresholds stay:
> [../results/2026-11-11-tau-floor-refit.md](../results/2026-11-11-tau-floor-refit.md).

## Why

Both thresholds were fitted to `P` from networks trained with `1 − generalizedDice` alone:
- `tau = 0.50`: cross-fitting 0.40–0.70 lost 0.0039 IoU on the MATLAB reference cache and on
  both PyTorch baseline caches ([2026-11-05](../results/2026-11-05-tau-crossfit.md));
- `relaxFloor = 0.20`: re-tuning it jointly with the window lost 0.0098 cross-fitted on the
  68-image domain ([2026-09-30](../results/2026-09-30-relax-grid.md)).

The adopted recipe adds a cross-entropy term ([2026-11-11](../results/2026-11-11-paper-recipe.md)).
CE gives graded probabilities where Dice saturates, so the distribution of `P` around 0.5 and
in the 0.1–0.3 band that relax reads has changed. Both earlier records say they hold only
until `P` changes, and SPEC asks for a re-fit after a loss change.

They are fitted jointly because they interact: `tau` decides which seeds the threshold leaves
empty, and only those seeds fall back to relax at `relaxFloor`.

## Arm

No pipeline code changes; only `cfg.Post.tau` and `cfg.Post.relaxFloor` vary. Everything
else is `cnngvf.defaults()`: relax window 128 px, `MinArea` 500, boundary claiming,
`hullLink` 128.

**Grid:** `tau` ∈ {0.40, 0.45, **0.50**, 0.55, 0.60, 0.65, 0.70} × `relaxFloor` ∈ {0.10, 0.15,
**0.20**, 0.25, 0.30}: 35 cells. The default is (0.50, 0.20).

- `tau` covers the earlier tau grid.
- `relaxFloor` covers the flat region of the relax grid (0.10–0.20) and two steps above. A
  less saturated `P` may need a higher floor to avoid flooding. The 0.05 floor flooded in
  2026-09-30 and is left out.
- Every cell has `relaxFloor < tau`, so relax always reads below the threshold.

**Runner:** `run_postfit_grid`, the archived `run_tau_grid` (tag `archive/tau-crossfit`)
generalized to a grid of cells. Every cell goes through `run_cache`, so every `P` is
out-of-sample, with output pointed at a scratch directory.

## Selection (cross-fitting)

This is the rule of both earlier cross-fits:
- for each fold, choose the cell with the best mean IoU on the other four folds' images;
- score that cell on the held-out fold;
- cells within 0.001 IoU of the best count as tied, and the default wins a tie. Among other
  tied cells, the one listed first in the grid order wins: `tau` nearest 0.50 first, then
  `relaxFloor` nearest 0.20.

The fixed-cell surface is reported as description only, never as a result.

## Protocol

- **Cache:** the reference, `<Output>/cv/probcache` (the 2026-11-11 recipe; 69 images, DRAC22
  42 + TUH 27, 5-fold CV). It is the only cache of this recipe, so there is no robustness cache.
- **Sanity check:** the default cell reproduces
  `reference/2026-11-11-paper-recipe-69.csv`, max |dIoU| ~1e-15.
- **Comparison:** paired per image against the default cell on the same cache. A post-`P`
  comparison on one cache carries no training noise, so the noise floor does not apply.
- **Statistics:** percentile bootstrap on the mean Δ, 10 000 resamples, `rng(20260917)`, 95 %
  CI. Counts of images up and down.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.

## Prediction (written before running)

- **The `tau` curve is flat around 0.50,** within ±0.005 IoU over 0.45–0.60 at floor 0.20.
- **The floor matters more than it used to.** With CE, more of `P` sits between 0.1 and 0.3,
  so a low floor floods more. I expect the best floor at 0.20–0.30.
- **Folds choose `tau` 0.45–0.55 and floor 0.20–0.25,** without 4 of 5 agreeing on one cell.
- **Cross-fitted ΔIoU: −0.005 to +0.005,** with a CI including 0.
- About a 25 % chance the gate passes. The default most likely stays.

## Gate

Adopt a new cell only if all three hold:

1. The cross-fitted ΔIoU against the default has a 95 % CI lower bound above 0.
2. At least 4 of 5 folds choose the same cell. That cell becomes the default.
3. **Guard, both sides:** on the cross-fitted masks, Q1–Q3 precision and Q4–Q5 recall each
   fall by no more than 0.01. Precision and recall are now level (0.742 / 0.735), and a cell
   that trades one failure mode for the other is not a fit.

**Otherwise** both thresholds stay. Tag the runner `archive/tau-floor-refit`, delete it, and
record the result.

**On adoption:** change the defaults, write a new reference CSV, and update the README table,
SPEC and `cfg.CV.reference`.

**Report alongside, as description:**
- the 7 × 5 fixed-cell surface (IoU), with precision and recall at floor 0.20 and `tau` 0.50;
- each fold's choice;
- precision and recall by quintile, cross-fitted against the default;
- per-dataset breakdowns;
- the cross-device check: choose on one dataset, score the other.

**Valid until `P` changes.**
