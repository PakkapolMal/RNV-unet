# Re-fitting tau × relaxFloor on the new P gains nothing: +0.0004 cross-fitted, both stay (2026-11-11)

Brief: [../briefs/tau-floor-refit.md](../briefs/tau-floor-refit.md).

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- the reference cache, `cv/probcache` (the 2026-11-11 recipe:
  [2026-11-11-paper-recipe.md](2026-11-11-paper-recipe.md)), 5-fold CV;
- whole-image scoring by `run_cache`, one run per cell;
- grid `tau` ∈ {0.40 … 0.70, step 0.05} × `relaxFloor` ∈ {0.10, 0.15, 0.20, 0.25, 0.30}, 35 cells;
- each fold's cell chosen on the other four folds, ties within 0.001 to the default (0.50, 0.20);
- paired against the default on the same cache, bootstrap 10 000 resamples, `rng(20260917)`.

**Sanity check: passes.** The default cell reproduces
[reference/2026-11-11-paper-recipe-69.csv](reference/2026-11-11-paper-recipe-69.csv), max |dIoU|
5.6e-16.

Runner: `run_postfit_grid`, at tag `archive/tau-floor-refit`.

## Result: the gate fails, and both defaults stay

| | IoU | F1 | precision | recall |
|---|---|---|---|---|
| default (0.50, 0.20) | 0.6125 | 0.7273 | 0.7418 | 0.7348 |
| cross-fitted | 0.6128 | 0.7275 | 0.7369 | 0.7422 |

**Cross-fitted ΔIoU +0.0004, 95 % CI [−0.0027, +0.0033];** 36 images up, 24 down.

| fold | chosen (`tau`, floor) | IoU, other 4 folds | IoU, held-out fold | default, held-out fold |
|---|---|---|---|---|
| 1 | (0.45, 0.10) | 0.6020 | 0.6655 | 0.6646 |
| 2 | (0.50, 0.10) | 0.6160 | 0.6060 | 0.6061 |
| 3 | (0.45, 0.10) | 0.6239 | 0.5796 | 0.5794 |
| 4 | (0.45, 0.10) | 0.6399 | 0.5168 | 0.5109 |
| 5 | (0.40, 0.15) | 0.5954 | 0.7026 | 0.7081 |

**Gate:**
1. **CI lower bound above 0: fails** (−0.0027).
2. **4 of 5 folds on one cell: fails.** Three folds chose (0.45, 0.10).
3. **Guards: hold.** Q1–Q3 precision −0.0036 (limit −0.01); Q4–Q5 recall +0.0103.

## The surface (description, not a result)

Mean IoU of each fixed cell, scored on the images it is chosen on:

| `tau` \ floor | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|
| 0.40 | **0.6161** | 0.6156 | 0.6144 | 0.6117 | 0.6082 |
| 0.45 | 0.6149 | 0.6142 | 0.6129 | 0.6102 | 0.6059 |
| 0.50 | 0.6139 | 0.6137 | *0.6125* | 0.6090 | 0.6044 |
| 0.55 | 0.6105 | 0.6102 | 0.6090 | 0.6049 | 0.5996 |
| 0.60 | 0.6112 | 0.6108 | 0.6096 | 0.6054 | 0.5998 |
| 0.65 | 0.6099 | 0.6096 | 0.6085 | 0.6053 | 0.5990 |
| 0.70 | 0.6088 | 0.6085 | 0.6074 | 0.6038 | 0.5979 |

- **The whole surface spans 0.018, and the best cell is +0.0036 over the default.** That
  gain is in-sample. Cross-fitted, 90 % of it disappears.
- **Lower is better on both axes, monotonically for the floor.** At every `tau`, IoU falls
  from floor 0.10 to 0.30. The best cell sits at the grid's corner. The grid was not extended
  past it: the in-sample headroom is under 0.004, and extending a grid toward the best cell
  on the scored images is the tuning the cross-fit exists to prevent.
- **The trade runs as expected.** Raising `tau` or the floor buys precision and costs
  recall: from (0.40, 0.10) to (0.70, 0.30), precision goes 0.734 → 0.754 and recall
  0.750 → 0.701.
- **The datasets pull apart.** Choosing on TUH and scoring DRAC22 picks (0.50, 0.15):
  +0.0009 on DRAC22. Choosing on DRAC22 and scoring TUH picks (0.40, 0.20): −0.0044 on TUH.

By quintile, cross-fitted against the default:

| quintile | Δ precision | Δ recall | ΔIoU |
|---|---|---|---|
| Q1 | −0.004 | +0.004 | −0.001 |
| Q2 | −0.003 | +0.004 | −0.001 |
| Q3 | −0.004 | +0.008 | −0.001 |
| Q4 | −0.009 | +0.011 | +0.001 |
| Q5 | −0.004 | +0.009 | +0.004 |

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| `tau` curve flat within ±0.005 over 0.45–0.60 at floor 0.20 | held: 0.6129 to 0.6090 |
| the floor matters more, best at 0.20–0.30 | **half wrong:** it matters (0.30 costs up to 0.009), but the best floor is the lowest, 0.10 |
| folds choose `tau` 0.45–0.55 and floor 0.20–0.25, no 4-of-5 agreement | `tau` held (0.40–0.50); **floor wrong** (0.10–0.15); no agreement held |
| cross-fitted ΔIoU −0.005 to +0.005, CI including 0 | held: +0.0004, [−0.0027, +0.0033] |
| about a 25 % chance of passing | failed |

## Reading

- **Both thresholds stay: `tau` 0.50, `relaxFloor` 0.20.** On the new `P`, re-fitting them is
  worth +0.0004 IoU (69 images, 5-fold CV, CI [−0.003, +0.003]).
- **The new `P` does not want a higher floor.** I expected CE to spread `P` into the
  0.1–0.3 band so that a low floor floods. The opposite holds: a lower floor helps slightly
  at every `tau`. The relax component it adds is lesion more often than not.
- **Post-processing is still not where the gains are.** Under three different `P` (MATLAB,
  the replica PyTorch runs, this recipe), cross-fitting `tau` has moved the score by less
  than 0.004 in either direction.
- **Valid until `P` changes.**
