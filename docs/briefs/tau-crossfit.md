---
title: tau, Cross-fitted
description: Pre-registered cross-fit of Post.tau on the cached CV P (levers analysis R4). Grid, selection rule, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-09-17-refine.md, results/2026-09-16-reduce.md, results/2026-09-30-relax-grid.md, results/2026-11-05-colab-parity-cv.md, research_analysis/rnv-score-levers-2026-10-29.md
status: pre-registered 2026-11-05, not yet run
---

# tau, Cross-fitted

## Why

`cfg.Post.tau = 0.5` is pre-registered as the softmax decision boundary and has never been
tuned. The 2026-09-17 refine record found that, under relax at 0.20, the `tau` curve rises
above 0.5:

| `tau` | 0.50 | 0.55 | 0.60 | 0.65 |
|---|---|---|---|---|
| mean IoU | 0.5097 | 0.5157 | 0.5189 | 0.5182 |

That curve is 68 images, 5-fold CV `P`, before the seed hull. It was recorded and not tuned,
because the curve's maximum is chosen on the images it scores.

Precision is the weak side (0.6410 against recall 0.6975 in the reference), and the largest
in-scope loss is spill on small lesions. A higher `tau` trades recall for precision.

Levers analysis R4 asks for a cross-fit. SPEC, "Known limits", lists it as untested.

**What has changed since that curve:**
- the domain: the standardized 69 images, not the old 68;
- boundary claiming;
- the seed hull at 128 px, which fills between seeds after claiming and may give back some of
  the recall a higher `tau` costs.

Its numbers therefore do not predict this run.

## Arm

No code change to the pipeline. Only `cfg.Post.tau` varies. Everything else is
`cnngvf.defaults()`: relax at 0.20 within 128 px, `MinArea` 500, boundary claiming,
`hullLink` 128.

`tau` enters at one point, `BW = P > tau` in `cnngvf.maskFromProb`. It also decides which
seeds fall back to relax, because relax fires only on a seed the threshold left empty.

**Grid:** `tau` ∈ {0.40, 0.45, **0.50**, 0.55, 0.60, 0.65, 0.70}. 0.50 is the default.

The grid goes below 0.5 because the threshold-only curves (2026-09-16, §3 and §7) peaked
there. It goes up to 0.70 to cover the far side of the refine curve's peak at 0.60–0.65.

## Selection (cross-fitting)

This is the relax grid's rule:
- for each fold, choose the `tau` with the best mean IoU on the other four folds' images;
- score that `tau` on the held-out fold;
- values within 0.001 IoU of the best count as tied, and the default, 0.50, wins a tie.

The fixed-`tau` results are reported as a curve only, never as a result.

## Caches

1. **Primary:** the reference cache, `<Output>/cv/probcache`. It holds the MATLAB fold
   networks of 2026-10-23, and reproduces
   `reference/2026-10-23-standardized-69.csv` at `tau` 0.50.
2. **Robustness:** the two PyTorch baseline caches,
   `colab_baseline/parity_rng0/probcache` and `parity_rng1/probcache`. These are the
   trainer future arms use.

Each cache is cross-fitted on its own. On a fixed cache a post-`P` comparison carries no
training noise. The robustness caches ask whether the gain survives a retrain, which the
noise floor (2026-11-05) makes the open question.

## Protocol

- 69 images (DRAC22 42, TUH 27), 5-fold CV `P`, whole-image scoring through `run_cache`.
- Compare paired per image against `tau` 0.50 on the same cache.
- Percentile bootstrap on the mean Δ: 10 000 resamples, seeded `rng(20260917)`, 95 % CI
  (one arm, so no correction). Report counts of images up and down.
- Size quintiles are by GT area over the 69 images: Q1 smallest, Q5 largest, 14/14/14/14/13.

**Sanity check:** on the primary cache, `tau` 0.50 reproduces the reference CSV, max
|dIoU| ~1e-15.

## Prediction (written before running)

- **The curve rises above 0.50** on every cache. Its maximum is at 0.55–0.65.
- **Folds choose 0.55–0.65.** At least 4 of 5 agree on the primary cache.
- **Cross-fitted ΔIoU on the primary cache: +0.003 to +0.010.** The CI may touch 0. The
  seed hull, added since the refine curve, takes back some of what the threshold gains.
- **Precision rises by about 0.02, recall falls.** Recall on the two largest quintiles falls
  most, because 76 % of Q5's missed pixels already sit below 0.20 and a higher `tau` adds
  more. I expect the recall guard (gate 3) to be the binding one.
- **On the PyTorch caches the sign is the same.** Their magnitudes are within ±0.005 of the
  primary's.

## Gate

Adopt a new `tau` only if all four hold:

1. On the primary cache, the cross-fitted ΔIoU 95 % CI lower bound is above 0.
2. On the primary cache, at least 4 of 5 folds choose the same `tau`. That value becomes the
   default.
3. **Recall guard**, on the primary cache's cross-fitted masks: mean recall over Q4–Q5 falls
   by no more than 0.01.
4. On **both** PyTorch baseline caches, the cross-fitted mean ΔIoU is above 0. No CI is
   required, but a gain that vanishes on a retrained network does not carry to future arms.

**Otherwise** `tau` stays 0.50. Tag the runner `archive/tau-crossfit`, delete it, and record
the result. On adoption: change the default, write a new reference CSV and update the README
table, SPEC and `cfg.CV.reference`.

**Report alongside, as description, not evidence:**
- the fixed-`tau` curve per cache;
- the folds' choices;
- precision and recall by quintile;
- per-dataset breakdowns;
- the cross-device check: choose on one dataset, score the other.

**Valid until `P` changes.** Any change to the loss or training (R2, R3) moves the `P`
distribution, so this result applies only to the networks it was measured on. Re-run it
after any change to `P` that is adopted.
