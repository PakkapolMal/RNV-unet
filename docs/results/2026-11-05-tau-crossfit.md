# Cross-fitting `tau` loses 0.004 IoU: 0.50 is already the best value (2026-11-05)

Brief: [../briefs/tau-crossfit.md](../briefs/tau-crossfit.md) (levers analysis R4). Domain:
the standardized 69 images (DRAC22 42 + TUH 27). Protocol: 5-fold CV `P`, whole-image
scoring through `run_cache`.

Only `Post.tau` varied, over {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}. Everything else is
`cnngvf.defaults()`: relax at 0.20 within 128 px, boundary claiming, `hullLink` 128.

For each fold, the `tau` with the best mean IoU on the other four folds was scored on the
held-out fold. Ties within 0.001 went to 0.50. Results are paired per image against `tau`
0.50. Bootstrap: 10 000 resamples, `rng(20260917)`.

Three caches were each cross-fitted on their own:
- **Primary:** the reference cache (MATLAB fold networks, 2026-10-23).
- **Robustness:** the two PyTorch baseline caches
  ([parity CV](2026-11-05-colab-parity-cv.md)).

Code: `run_tau_grid.m`, at tag `archive/tau-crossfit`.

**Sanity check passes:** on the primary cache, `tau` 0.50 reproduces
`reference/2026-10-23-standardized-69.csv` with max |dIoU| 4.4e-16.

## Result: the gate fails, and `tau` stays at 0.50

| cache | IoU at `tau` 0.50 | cross-fitted | ΔIoU | 95 % CI | up / down | folds' choices |
|---|---|---|---|---|---|---|
| **reference (MATLAB)** | 0.5047 | 0.5008 | **−0.0039** | [−0.0081, −0.0004] | 9 / 15 | 0.50, 0.50, 0.50, 0.55, 0.45 |
| PyTorch RNG 0 | 0.5002 | 0.4922 | −0.0080 | [−0.0140, −0.0023] | 15 / 32 | 0.50, 0.60, 0.60, 0.45, 0.60 |
| PyTorch RNG 1 | 0.5010 | 0.4939 | −0.0070 | [−0.0116, −0.0031] | 4 / 21 | 0.50, 0.50, 0.60, 0.45, 0.50 |

| gate | needed | measured | |
|---|---|---|---|
| 1. CI lower bound > 0 (primary) | > 0 | −0.0081 | fail |
| 2. ≥ 4 of 5 folds agree (primary) | 4 | 3 (on 0.50, the default) | fail |
| 3. Q4–Q5 recall falls ≤ 0.01 (primary) | ≥ −0.01 | −0.0038 | pass |
| 4. mean Δ > 0 on both PyTorch caches | > 0 | −0.0080, −0.0070 | fail |

Cross-fitting is significantly **worse** on all three caches, and every CI lies entirely
below 0. Where the folds pick anything but 0.50, the choice does not hold up on the
held-out fold. On the primary cache, fold 4 picked 0.55 (0.4673 against 0.4724 at 0.50) and
fold 5 picked 0.45 (0.4456 against 0.4611).

## Prediction: wrong

The brief predicted:
- the curve rises above 0.50, peaking at 0.55–0.65;
- folds choose 0.55–0.65;
- a gain of +0.003 to +0.010.

Instead, on the current pipeline and domain, **the curve peaks at 0.50**:

| `tau` | 0.40 | 0.45 | **0.50** | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|---|
| reference | 0.4920 | 0.5032 | **0.5047** | 0.5046 | 0.5007 | 0.4945 | 0.4920 |
| PyTorch RNG 0 | 0.4893 | 0.4986 | 0.5002 | 0.4992 | 0.5008 | 0.4880 | 0.4789 |
| PyTorch RNG 1 | 0.4929 | 0.5000 | 0.5010 | 0.5008 | 0.5018 | 0.4970 | 0.4903 |

Curves are mean IoU over 69 images. They are tuned on the images they score, so they are
description, not results.

**The curve is flat from 0.45 to 0.60.** On all three caches it varies by under 0.0045
across that range, with no consistent peak. The PyTorch caches' highest point (0.60) is only
0.0006–0.0008 above 0.50, which is selection noise, as the cross-fit shows.

**The 2026-09-17 rise above 0.50 is gone.** That curve gained +0.009 at 0.60, on the old 68
images, before the seed hull. Two changes made since are the likely reasons, not separated
here:
- The seed hull was adopted at 128 px. A higher `tau` breaks large lesions into fragments,
  and the hull already fills between seeds.
- The standardized `datasets/` changed the domain and the folds.

## Description: precision and recall

On the primary cache, cross-fitted against 0.50:
- precision 0.6410 → 0.6389;
- recall 0.6975 → 0.6937.

So the expected shift of about +0.02 precision did not happen. Recall on Q4–Q5 fell 0.0038,
which passes the guard.

| quintile (GT area) | n | IoU at 0.50 | IoU cross-fitted |
|---|---|---|---|
| Q1 (smallest) | 14 | 0.3329 | 0.3306 |
| Q2 | 14 | 0.4772 | 0.4760 |
| Q3 | 14 | 0.5309 | 0.5324 |
| Q4 | 14 | 0.5766 | 0.5669 |
| Q5 (largest) | 13 | 0.6138 | 0.6054 |

**Per dataset**, cross-fitted vs 0.50:
- DRAC22: 0.4291 vs 0.4357;
- TUH: 0.6122 vs 0.6121.

**Cross-device**, choosing on one dataset and scoring the other:
- choose on TUH, score DRAC22: picks 0.50, the default itself;
- choose on DRAC22, score TUH: picks 0.55, 0.6053 vs 0.6121 at 0.50.

## What this settles

- **`tau` 0.50 stays pre-registered.** It is now measured as well: on these networks no
  other value beats it out of sample.
- **The precision gap is not a threshold problem.** It does not respond to one global
  `tau`. The spill on small lesions has to be fixed in `P`, which is levers analysis R3
  (Dice + cross-entropy).
- **Valid until `P` changes.** Re-run this cross-fit if a loss or training change is
  adopted.
