---
title: Hard-Negative Seeds and Multi-Scale Seed Conditioning
description: Pre-registered trained arm. The seeded network trains with 20 % all-background patches at rejected detections and bright background vasculature, and sees the seed heat-map again at every decoder scale. An image whose points claim nothing takes the stage 9 mask. One run (--rng 0); the gate was written before any code. Measured before writing, the stated mechanism's oracle ceiling on the current pipeline is +0.0008 IoU.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn (export, stage 9 handoff), RNV-Claude/colab (trainer), Colab L4 (user runs)
depends-on: results/2026-11-13-relax-no-disk-fusion.md, results/2026-11-12-seedless-fallback.md, results/2026-11-11-paper-recipe.md, results/2026-09-17-refine.md
status: rejected before training 2026-11-14
---

# Hard-Negative Seeds and Multi-Scale Seed Conditioning

> **Rejected brief.** Kept as written. Not trained: the oracle ceiling measured below, +0.0008
> IoU against a +0.015 gate, ended it before the Colab run, by the user's decision.
> The empty-mask handoff moves no image:
> [../results/2026-11-14-hard-neg-seeds.md](../results/2026-11-14-hard-neg-seeds.md).

## Why (as proposed)

**The training patches.** `Patch.seedTolerance = 20` drops every detection more than 20 px from
a lesion (837 kept, 34 dropped). So **every training patch has a lesion under the heat-map
peak**. The proposal is that the network learns the shortcut "bump = foreground" and ignores
whether the vasculature under the bump is neovascular.

**The inference symptom.** The proposal cites stray points firing with `P ≈ 1` (SPEC, "Known
limits") and these images:
- **stray points only:** `299` 0.198, `573` 0.000 (`PredPx` 0), `664` 0.253;
- **mixed:** `098` 0.179, `134` 0.322.

**A second gap.** When an image with points ends with an empty mask, the pipeline outputs that
empty mask. "The mask is empty" is known at inference, so stage 9 could take over.

**References:**
- **Sofiiuk et al., ICIP 2022 (RITM):** click-conditioned networks shortcut to the click unless
  they also train on negative clicks.
- **Lin et al., CVPR 2020 (First Click Attention):** click guidance carried past the first
  encoder layer keeps deep decoder blocks localised.
- **Tun & Aimmanee, IEEE Access 2024; Sujarae et al., lab manuscript 2025:** the upstream
  detector's false positives fall on dense, branching normal vasculature, not on dark
  background. That is the distribution the negatives below sample.

## Measured before writing this brief

On the 2026-11-13 caches (`cv/probcache`, `cv/probcache_seedless`), before any code. Distances
are true boundary distances at the rounded point, the rule `cnngvf.seedLabels` uses. These use
GT and describe the reference. They are not results of the arm.

**The 871 points by distance to the nearest lesion:**

| 0 px | ≤ 20 px | 20–50 px | 50–100 px | > 100 px |
|---|---|---|---|---|
| 790 | 47 | 4 | **1** | 29 |

SPEC's "nothing between 50 and 100 px" describes the 874-point domain. On these points, one
sits in that band: `664`'s only point, at 55 px.

**The premise does not hold on the current `P`.** "29 of 32 fire with `P ≈ 1`" was measured
on the replica U-Net ([2026-09-17](../results/2026-09-17-refine.md)), and SPEC still carries it.
On the adopted recipe's cache, the 30 points beyond 50 px (on 22 images; 26 DRAC22, 4 TUH) give:

| seeded `P` | far points |
|---|---|
| at the point > 0.9 | **5**: `135`, `419`, `1115`, `1116`, `nv (2)` |
| at the point > 0.5 | 6: the five above, plus `664` (0.87) |
| max within 32 px > 0.5 | 15 |

**What happens on the cited images:**
- **`098`** has no far point. Its three points lie 0–1 px from the lesion, so its 0.179 is not a
  stray-point effect.
- **`134`'s** one far point (230 px) has `P` ≤ 0.01 within 32 px.
- **`299`'s** four far points have `P` ≤ 0.01 at the point. All of its 0.198 comes from those points
  claiming the lesion at a distance ([2026-11-13](../results/2026-11-13-relax-no-disk-fusion.md)).
- **`573`'s** two far points have `P` ≤ 0.01. Nothing is claimed, so the mask is empty.
- **`664`'s** point is 55 px out, with `P` 0.87. Under the > 50 px rule below, it becomes a
  Source A negative.

**Ceilings on this cache.** The two oracles (O) use GT at inference, so they measure headroom
only and are never reportable.

| | mean IoU | ΔIoU | up / down |
|---|---|---|---|
| reference (reproduced to 5e-16) | 0.6604 | | |
| **E**: empty mask → stage 9, alone | 0.6604 | **0.0000** | 0 / 0 |
| **O2**: every far point's patch predicts nothing; the points still claim; with E | 0.6612 | **+0.0008** | 5 / 2 |
| O1: far points removed before inference | 0.6711 | +0.0107 | 10 / 2 |

- **E changes nothing.** `573` is the only image with points and an empty mask. Its seedless
  `P` peaks at 0.011, so stage 9 returns nothing there either.
- **O2 is what perfect hard-negative training can do** with the post-processing unchanged. In
  O2, `P` is set to 0 where only far points' patches reach. Its ceiling is small for two
  reasons:
  - only 5–6 far points fire;
  - a silent point still claims the nearest component, which has had no distance cap since
    2026-11-13.
- **O1 needs a decision at inference that only GT makes.** No training change stops a point from
  claiming. O1 also takes `125` from 0.756 to 0.530, because its stray point's claim helps.

**What this means for the gate.** The gate asks for +0.015. The stated mechanism has +0.0008 of
headroom. A pass would have to come from elsewhere, from the negatives or the decoder
conditioning changing `P` at *true* points. The run can still show that, so the arm stands as
specified. These figures were measured after the proposal and changed nothing in the arm. They
inform only the predictions below.

## Arm

There are three changes, tested as one bundle. As with the
[paper recipe](../results/2026-11-11-paper-recipe.md), which training change carries a gain
will not be known. E acts after `P`, so its share is split off on the new cache.

### 1. Negative seeds (training only)

| | rule |
|---|---|
| positives | unchanged: labelled at tolerance 20. The positive list is drawn first from the same rng, so it is the reference run's list. |
| **Source A** | detections more than **50 px** from every lesion: 30 points, including `664`'s. The 4 points at 20–50 px are neither positive nor negative. |
| **Source B** | pixels more than 50 px from every lesion with intensity ≥ the **75th percentile** of that image's non-lesion pixels |
| where computed | `cnngvf.negativeSeeds`, in MATLAB, exported per image by `run_export`. Ground truth stays on the MATLAB side, as the labels do. The trainer only samples. |
| count | `round(negSeedFrac / (1 − negSeedFrac) × positives)`: 25 % of the positives, 20 % of the epoch |
| spread | equal across every non-test image with a candidate, the remainder to random images. This includes the images with no usable point. |
| per draw | a uniform Source A point with p 0.5 where the image has one, else a uniform Source B pixel |
| when | drawn once per run, like the positives. Jitter and augmentation are fresh every epoch. |
| target | all background |
| augmentation | the positives': ±32 px jitter, flips, `rot90`, affine p 0.75, CLAHE p 0.5. The heat-map peaks at the negative point. |
| `cfg.Patch.negSeedFrac` | **0.20**; 0 on the frozen arms |

**The loss is unchanged.** In an all-negative minibatch, the pooled foreground Dice is
0 / (ΣY + 1e-7) = 0. That is finite, and Dice passes no gradient, so CE carries the batch. The
trainer's 1e-7 is in fp32 and stays. Changing it to 1e-6 would alter the reference recipe for
nothing. A unit check asserts the loss and its gradient are finite.

### 2. Multi-scale seed conditioning

- **`cfg.Net.multiScaleSeed`** is **true**. The encoder is unchanged: 2 channels, ImageNet
  weights.
- Each `smp.Unet` decoder block's first convolution takes one more input channel: the raw seed
  heat-map, `adaptive_max_pool2d` to that block's size. It is concatenated after the upsampled
  and skip features. That conv is initialised as smp initialises its decoder.
- A bundle without the field trains the plain network. `test_parity.py`'s replica U-Net is not
  touched.

### 3. Empty mask → stage 9

- **`cfg.Post.emptyToSeedless`** is **true**; false on the frozen arms.
- In `cnngvf.cachedMask`, an image with points whose stages 3–8 return an empty mask takes
  `cnngvf.noPointMask` of its seedless `P`. The read goes through the same leakage guard as
  stage 9. No GT is involved.

On `exp/hard-neg-seeds`, all three default on, so the bundle `run_export` writes from
`cnngvf.defaults()` *is* the arm. The reference reproduces with the three set off.

## Protocol

| | |
|---|---|
| training | `rnvtorch.py train`, 5 folds from `cnngvf.cvFolds`, one run, **`--rng 0`**, the adopted recipe otherwise (40 epochs, cosine 1e-4 → 1e-6, minibatch 8, last epoch) |
| seedless network | unchanged: `cv/probcache_seedless` serves stages 4a and 9 |
| stall rule | before any IoU is read: a fold whose last-five-epoch mean loss exceeds 1.25 × the five folds' median retrains once with `--rng 1`, and that retrain is used whatever it scores |
| scoring | `run_cache` on the new cache, all three flags on |
| reference | `reference/2026-11-13-relax-no-disk-69.csv`, 0.6604. It must reproduce first: the old cache with the flags off, max \|dIoU\| ~1e-15. |
| statistics | paired per image; percentile bootstrap on the mean Δ, 10 000 resamples, `rng(20260917)`, 95 % CI; exact two-sided sign test |
| the 63 | images with at least one point labelled at tolerance 20 (`seedLabels` comp > 0). Reference 0.6900. A GT-defined subset, used as a guard only, never a headline. |
| 1:1 mean | the mean of the DRAC22 and TUH means. Reference (0.6167 + 0.7285) / 2 = **0.6726**. For comparison with Sujarae et al.; the pooled 69-image mean stays the headline. |
| tuned | nothing. 0.20, 50 px, the 75th percentile and max-pool injection are fixed here. |

## Predictions (written before any code)

**Claude:**
- No fold stalls.
- Far points firing (`P` > 0.5 at the point) fall from 6 to 0–2.
- E fires on at most 2 images and moves the headline by no more than ±0.001. `573`'s seedless
  `P` is empty whatever the seeded network does.
- **ΔIoU −0.015 to +0.010, point estimate −0.003.** Most of the movement lands on the
  threshold-unstable small lesions (Q1), not on the stray-point images.
- **Δ63 −0.012 to +0.008.** Recall falls more than precision rises. Negatives on bright
  vasculature teach the network to lower `P` on vessel-like texture, and weak true points sit
  on exactly that.
- The 1:1 mean moves with the pooled mean, within 0.005 of it.
- **The gate passes with about 5 % odds.**

**The user:** the proposal's Why. Stray points stop firing, and the arm clears the gate without
costing the 63 images.

## Gate

Preconditions. A violation voids the run; it is neither a pass nor a fail.

- **Stall check:** all five `log.csv` files are checked before any IoU is read.
- **Zero leakage:**
  - GT labels positives and negatives only inside each training fold.
  - Inference sees all 871 detections unchanged.
  - Stage 9 triggers only on `nnz(mask) == 0`.

Adopt only if both of these hold:

1. **Gain:** the mean paired ΔIoU over 69 images is **≥ +0.0150**, *and* its 95 % CI lower bound
   is **> 0**.
2. **Recall guard:** ΔIoU on the 63 images with a usable point is **≥ −0.0050**.

**On adoption:**
- the three flags stay on;
- a new reference CSV and `cfg.CV.reference`;
- SPEC: the network, the sampling table, the stage 9 trigger, and the seed-tolerance numbers;
- the README table and the notebook.

**On rejection:**
- write the record;
- tag `archive/hard-neg-seeds`;
- delete the code;
- merge the records only.

**Either way:**
- SPEC's "29 of 32 fire with `P ≈ 1`" and its empty 50–100 px band are corrected from the
  measurements above. They describe the reference, not the arm.

## Report alongside, as description

- **E's share:** the new cache scored with `emptyToSeedless` off.
- **Far points:** `P` at each point and the max within 32 px, against the 6 and 15 above.
- **Weak points:** points the threshold leaves empty, against the reference count. This is the
  direct test of the recall worry.
- per dataset, the 1:1 mean, GT-area quintiles, and the largest gains and losses;
- the six images without a usable point;
- s/epoch and CV time. Negatives add 25 % patches per epoch, so expect about 6.3 h per CV on an
  L4, against 5 h.

## Running it

```matlab
cd RNV-Claude/cnn
test_cnngvf
cfg = cnngvf.defaults();
off = cfg; off.Post.emptyToSeedless = false;
run_cache(off)                       % the reference reproduces, ~1e-15
run_cache(cfg)                       % E alone on the old cache: 0 images move
run_export(cfg, fullfile(cfg.Paths.Output, 'colab_bundle_hardneg'))
```

On Colab, upload the bundle as `MyDrive/rnv/colab_bundle_hardneg/` and the branch's `colab/`.
Then run:

```
!rsync -a {DRIVE}/colab_bundle_hardneg/ /content/bundle/
!python {DRIVE}/colab/test_parity.py /content/bundle
!python {DRIVE}/colab/test_parity.py stage3 /content/bundle {DRIVE}/runs/smoke_hardneg 2
for FOLD in range(1, 6):
    !python {DRIVE}/colab/rnvtorch.py train /content/bundle {DRIVE}/runs/hardneg_rng0 --fold {FOLD} --rng 0 --workers 2
```

Download `runs/hardneg_rng0/` to `cnn/result/DRACTUH_unclustered/colab_hardneg/`. Apply the
stall rule to its `fold*/log.csv`, then score:

```matlab
cfg = cnngvf.defaults();
run_cache(cfg, fullfile(cfg.Paths.Output, 'colab_hardneg', 'probcache'))
```

**Valid until the seedless `P` or the post-processing changes.**
