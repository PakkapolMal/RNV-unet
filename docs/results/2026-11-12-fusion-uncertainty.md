# Fusing the seedless P on every image is worth +0.012 IoU cross-fitted, but the folds split on τ_s (2026-11-12)

Brief: [../briefs/fusion-uncertainty.md](../briefs/fusion-uncertainty.md).

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- the two adopted 5-fold CV caches, seeded `cv/probcache` and seedless
  `cv/probcache_seedless`;
- each image with points scored off and fused (pixel max, τ_s 0.9 and 0.5), then the unchanged
  pipeline;
- 9 cells: off, and the fraction fused q ∈ {10, 25, 50, 100 %} × τ_s;
- for each held-out fold, both the U cut-off and the cell come from the other folds' images;
- paired against off, bootstrap 10 000 resamples, `rng(20260917)`.

Runner: `run_fusion_u_grid`; stages `cnngvf.seededUncertainty`, `cnngvf.fuseUncertain`.

**Checks:**
- **sanity:** off reproduces
  [reference/2026-11-12-seedless-fallback-69.csv](reference/2026-11-12-seedless-fallback-69.csv),
  max |dIoU| 5.6e-16;
- **consistency:** the (all, 0.9) cell re-run through `run_cache` matches the runner exactly
  (max |dIoU| 0).

## Result: gates 1 and 3 hold, gate 2 fails

| | IoU | precision | recall |
|---|---|---|---|
| off (current) | 0.6364 | 0.7667 | 0.7625 |
| cross-fitted | 0.6479 | 0.7618 | 0.7818 |

**Cross-fitted ΔIoU +0.0116, 95 % CI [+0.0039, +0.0221];** 42 images up, 14 down, 13
unchanged.

| fold | chosen q | τ_s | images fused | IoU, other 4 folds | IoU, held-out fold | off, held-out fold |
|---|---|---|---|---|---|---|
| 1 | all | 0.9 | 14 | 0.6426 | 0.6713 | 0.6646 |
| 2 | all | 0.5 | 13 | 0.6591 | 0.6111 | 0.6061 |
| 3 | all | 0.9 | 13 | 0.6439 | 0.6662 | 0.6357 |
| 4 | all | 0.9 | 13 | 0.6648 | 0.5842 | 0.5726 |
| 5 | all | 0.5 | 13 | 0.6349 | 0.7114 | 0.7081 |

(Images fused counts each fold's images with points. Folds 2–4 each hold one image with no
points, which keeps stage 9.)

**Gate:**
1. **CI lower bound above 0: holds** (+0.0039).
2. **4 of 5 folds on one cell: fails.** All five folds chose to fuse every image. On τ_s they
   split 3 (0.9) to 2 (0.5).
3. **Precision guard: holds.** Q1–Q3 precision −0.0033 (limit −0.01). Q4–Q5 recall +0.0343.

**Where the folds split, the cells barely differ.** Over all 69 images, (all, 0.9) scores
0.6484 and (all, 0.5) 0.6493: 0.0009 apart, inside the pre-registered 0.001 tie tolerance. The
cell chosen on all 69 images, which would ship under Reporting rule 6, is (all, 0.9) by the tie
rule.

**The gain is concentrated.** `795` (+0.284) and `461` (+0.139), both large DRAC22 lesions,
carry 53 % of it. Without them the mean is +0.0056. 27 images moved by more than 0.01.

## The surface (description, not a result)

Cut-off set on all 66 images with points, scored on all 69:

| q | τ_s | U cut-off | images fused | IoU | precision | recall | DRAC22 | TUH |
|---|---|---|---|---|---|---|---|---|
| off | | | 0 | 0.6364 | 0.7667 | 0.7625 | 0.5797 | 0.7245 |
| 10 % | 0.9 | 1.071 | 7 | 0.6370 | 0.7668 | 0.7633 | 0.5801 | 0.7256 |
| 10 % | 0.5 | 1.071 | 7 | 0.6369 | 0.7664 | 0.7633 | 0.5797 | 0.7258 |
| 25 % | 0.9 | 0.708 | 17 | 0.6369 | 0.7662 | 0.7639 | 0.5797 | 0.7259 |
| 25 % | 0.5 | 0.708 | 17 | 0.6363 | 0.7650 | 0.7642 | 0.5788 | 0.7256 |
| 50 % | 0.9 | 0.318 | 33 | 0.6406 | 0.7658 | 0.7689 | 0.5864 | 0.7250 |
| 50 % | 0.5 | 0.318 | 33 | 0.6408 | 0.7632 | 0.7717 | 0.5870 | 0.7246 |
| all | 0.9 | | 66 | 0.6484 | 0.7636 | 0.7806 | 0.5975 | 0.7277 |
| all | 0.5 | | 66 | 0.6493 | 0.7590 | 0.7863 | 0.5987 | 0.7281 |

- **Gating by U does not help. Fusing more images is better at every step,** and fusing every
  image is best. The image that gains most, `795`, has one of the lowest U values (0.05).
- **U does rank poor seeded images** (Spearman −0.45 with the seeded IoU over 66 images, a GT
  diagnostic). But the gain from fusion does not follow U: a decisive seeded map can still stop
  short of a lesion's extent.
- **Fusion adds little area.** Per image, the median is 190 px added to the final mask at τ_s
  0.9 and 370 px at 0.5. Claiming drops the rest.

By quintile, cross-fitted against off:

| quintile | Δ precision | Δ recall | ΔIoU |
|---|---|---|---|
| Q1 | −0.002 | +0.012 | +0.009 |
| Q2 | −0.007 | +0.007 | −0.001 |
| Q3 | −0.001 | +0.009 | +0.006 |
| Q4 | −0.008 | +0.020 | +0.010 |
| Q5 | −0.007 | +0.049 | +0.037 |

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| fusing all images: net ΔIoU −0.003 to +0.006 | **wrong, above:** +0.012 to +0.013 in-sample |
| gating helps; best cells fuse the top 25–50 % at τ_s 0.9 | **wrong:** fusing all is best at both τ_s; U-gating adds nothing |
| cross-fitted ΔIoU +0.000 to +0.008, CI likely including 0; folds split between neighbouring cells | **magnitude wrong, split right:** +0.0116, CI [+0.004, +0.022]; folds split on τ_s only |
| Spearman(U, seeded IoU) −0.2 to −0.5 | held: −0.45 |
| Q4–Q5 recall rises, Q1–Q3 precision falls | held: +0.034 and −0.003 |
| about a 25 % chance of passing | failed on gate 2 alone |

## Reading

- **By its pre-registered gate, the arm is not adopted.** Gate 2 fails: no single cell won 4
  of 5 folds.
- **The failure is on the minor dimension.** Every fold chose to fuse every image. The τ_s split
  is between two cells 0.0009 apart on all 69 images, inside the tie tolerance. The question the
  arm asked, whether fusing helps, got a unanimous yes from the folds and a CI above 0.
- **What the gain is.** Fusing the seedless `P` into every image with points, then claiming as
  usual, recovers lesion extent the seeded outline stops short of. It is largest on the biggest
  lesions (Q5 recall +0.049). Half of it rests on two images, and the other half is spread over
  the rest (+0.0056 without them).
- **The U gate is not needed.** If fusion is adopted, the rule is the simpler one: fuse every
  image with points, and drop U.
- **Adopting it would override gate 2,** as the seed hull was adopted over its precision guard
  ([2026-09-30-seed-hull-d.md](2026-09-30-seed-hull-d.md)). That is a decision, not a result,
  and it is recorded as one if taken.
- **Valid until either `P` changes.**
