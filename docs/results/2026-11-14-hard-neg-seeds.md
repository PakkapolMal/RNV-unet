---
title: Hard-Negative Seeds and Multi-Scale Seed Conditioning
description: Rejected before training. On the adopted network's cache, only 5 of 30 far points fire, and perfect stray-point suppression is worth at most +0.0008 IoU against a +0.015 gate. The empty-mask handoff to stage 9 moves no image. The 5-fold CV was not run.
kind: measurement-record
status: rejected by pre-training oracle bound
dataset: DRAC22 + TUH (69)
updated: 2026-11-14
---

# Hard-Negative Seeds and Multi-Scale Seed Conditioning

Brief, predictions and gate: [../briefs/2026-11-14-hard-neg-seeds.md](../briefs/2026-11-14-hard-neg-seeds.md),
committed before any code (`b43de6a`). Code: tag `archive/hard-neg-seeds`.

**Question.** Should the seeded network train on all-background patches at rejected detections
and on bright background vasculature, and see the seed heat-map at every decoder scale? And
should an image whose points claim nothing take the stage 9 mask?

**Decision.** Rejected without training, by the user, on the oracle bound below. The gate asks
for +0.0150 IoU, but the mechanism the arm targets has +0.0008 of headroom on the current
pipeline. A pass would have had to come from effects the proposal did not predict. The arm was
built and its bundle exported. The ~6.3 h Colab run was not made, so the gate was never scored.

## The premise, measured

These figures come from the 2026-11-13 caches (`cv/probcache`, the adopted EfficientNet-B3
network; `cv/probcache_seedless`). Distance is the true boundary distance at the rounded point,
the rule of `cnngvf.seedLabels`.

**The 871 points by distance to the nearest lesion:**

| 0 px | ≤ 20 px | 20–50 px | 50–100 px | > 100 px |
|---|---|---|---|---|
| 790 | 47 | 4 | **1** | 29 |

The one point at 50–100 px is `664.png`'s only point, at 55 px. SPEC's "nothing between 50 and
100 px" describes the earlier 874-point domain.

**Far points (> 50 px) and the seeded `P`.** There are 30 such points, on 22 images: 26 DRAC22
and 4 TUH.

| seeded `P` | far points |
|---|---|
| at the point > 0.9 | **5 of 30**: `135`, `419`, `1115`, `1116`, `nv (2)` |
| at the point > 0.5 | 6: the five above, plus `664` (0.87) |
| max within 32 px > 0.5 | 15 |

The "29 of 32 fire with `P ≈ 1`" in SPEC was measured on the replica U-Net
([2026-09-17](2026-09-17-refine.md)). It does not describe the adopted network.

**The images the proposal cited:**

| image | IoU | what the cache shows |
|---|---|---|
| `098` | 0.179 | no far point: its three points lie 0–1 px from the lesion |
| `134` | 0.322 | one far point at 230 px, with `P` ≤ 0.01 within 32 px |
| `299` | 0.198 | four far points, with `P` ≤ 0.01 at each. The score comes from these points claiming the lesion at a distance. |
| `573` | 0 | two far points, with `P` ≤ 0.01. Nothing is claimed, so `PredPx` is 0. |
| `664` | 0.253 | one point at 55 px, with `P` 0.87 |

## Oracle bounds

The oracles use GT at inference, so they measure headroom only and are never reportable.
`run_cache` scoring, paired against `reference/2026-11-13-relax-no-disk-69.csv`, which
reproduces to 5e-16.

| | mean IoU | ΔIoU | up / down |
|---|---|---|---|
| reference | 0.6604 | | |
| **E**: empty mask → stage 9 (`emptyToSeedless`), alone | 0.6604 | **0.0000** | 0 / 0 |
| **O2**: every far point's patch predicts nothing; the points still claim; with E | 0.6612 | **+0.0008** | 5 / 2 |
| O1: far points removed before inference | 0.6711 | +0.0107 | 10 / 2 |

- **E moves no image.** `573.png` is the only image with points and an empty mask. Its seedless
  `P` peaks at `P_s` = 0.011, and at 0.002 inside the GT. So stage 9 returns 0 px there too.
  With the flag on, `run_cache` gives 0.660423, max |dIoU| 5e-16.
- **O2 is the ceiling for what hard-negative training can do** with the post-processing
  unchanged. In O2, `P` is set to 0 where only far points' patches reach. It moves `1115`
  +0.016, `135` +0.034, `299` +0.058, `nv (19)` +0.033 and `nv (2)` +0.021. It lowers `1116`
  (−0.094) and `664` (−0.016). The ceiling is small for two reasons:
  - only 5–6 far points fire;
  - a silent point still claims the nearest component, which has had no distance cap since
    2026-11-13.
- **O1 needs a decision only GT can make.** No training change keeps a point from claiming. O1
  also costs `125` 0.756 → 0.530, because there the stray point's claim helps.

## Against the predictions

None was scored, because the run was not made. Claude's brief already gave the gate about 5 %
odds on the bound above.

## What was built (tag `archive/hard-neg-seeds`)

| part | what it does |
|---|---|
| `cnngvf.negativeSeeds` | detections more than 50 px from every lesion, and background ≥ the 75th non-lesion intensity percentile as far out; exported by `run_export` |
| `rnvtorch.negative_rows` | 20 % of the epoch as all-background patches |
| `SmpNet(multi_scale_seed)` | the max-pooled heat-map fed into every decoder block |
| `cachedMask` | the empty-mask handoff (`Post.emptyToSeedless`) |

The checks that passed on the branch:
- `test_cnngvf`, 13 checks;
- `test_parity.py`, stages 1–2, on both bundles;
- `test_parity.py selftest`. Its multi-scale check was not run locally, because
  `segmentation_models_pytorch` is not installed.

The exported bundle (`colab_bundle_hardneg`) was deleted.

**Re-test only if** stray points start firing again under a new network, or claiming gains a
rule that a silent point cannot satisfy. Under the current pipeline, the headroom is in the
localization stage.
