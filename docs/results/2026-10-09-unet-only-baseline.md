---
title: U-Net-only baseline — the network's thresholded output with no post-processing
kind: measurement-record
dataset: DRAC22 + TUH, 68 images
protocol: 5-fold CV, out-of-sample cached P, whole-image scoring
updated: 2026-10-09
---

# U-Net-only baseline

**Question.** For the paper: how much does the network alone score, and how much does the
post-processing after `P` add?

**Prediction.** None was written down. This is a descriptive baseline, not an experiment:
it introduces no threshold and changes no code.

**Protocol.**
- **Domain:** 68 images (DRAC22 41, TUH 27), the same 5 folds as the current configuration,
  using the `DRACTUH_unclustered` fold networks from 2026-09-03.
- **P:** the cached out-of-sample `P` (`result/DRACTUH_unclustered/cv/probcache`). Each
  image's `P` comes from the fold network that held it out.
- **U-Net only:** `mask = P > Post.tau` with `tau` = 0.5, as pre-registered and not tuned.
  There is no seed fallback, `imclose`, `imfill`, seed claiming, `MinArea` or seed hull.
- **Full:** `cnngvf.maskFromProb` on the same `P`. This reproduces the reference
  [reference/2026-09-30-hull-128.csv](reference/2026-09-30-hull-128.csv) to a
  max |dIoU| of 5e-16.
- **Scoring:** `cnngvf.evaluate`, image mode. Paired bootstrap with 10 000 resamples and
  seed 20261009, plus a two-sided exact sign test.
- **Script:** scratch MATLAB, not committed. Per-image scores are in
  [2026-10-09-unet-only-baseline.csv](2026-10-09-unet-only-baseline.csv).

**What "U-Net only" means here.** The network is still **seed-conditioned**. It sees the
image plus a localization heat-map in 512 px patches centred on each detection, so `P` is 0
wherever no patch reaches. This baseline removes the post-processing, not the seeds. A
seedless U-Net that segments from the image alone has no cross-validated result (see
[2026-09-17-seedless.md](2026-09-17-seedless.md), which stopped at a fold-1 pilot).

## Result: post-processing adds +0.10 IoU, all of it recall

68 images, 5-fold CV, means over images:

| | IoU (median) | F1 | precision | recall |
|---|---|---|---|---|
| U-Net only, `P > 0.5` | 0.4201 (0.4339) | 0.5643 | 0.6336 | 0.6256 |
| full pipeline | **0.5207** (0.5375) | **0.6572** | 0.6268 | **0.7762** |

Paired ΔIoU (full − U-Net) is **+0.1006**, 95 % CI [+0.065, +0.138]. Full is better on 53
images, worse on 13 and tied on 2 (sign test p = 7e-7).

| subset (5-fold CV) | n | U-Net IoU | full IoU | Δ |
|---|---|---|---|---|
| DRAC22 | 41 | 0.4276 | 0.4831 | +0.056 |
| TUH | 27 | 0.4087 | 0.5777 | +0.169 |

- **The gain is recall.** Recall rises by 0.15 and precision barely moves (−0.007). The
  post-processing fills in lesion the threshold leaves out (seed fallback, hole filling,
  the seed hull). It does not mainly clean up false positives.
- **TUH gains three times as much as DRAC22.** TUH lesions are larger and more
  multi-component, which is where the fill stages act.
- **`299` and `445` score 0 either way.** They have no usable seed, so `P` never
  reaches the lesion, and post-processing cannot create one.
