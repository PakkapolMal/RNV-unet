# A pretrained EfficientNet-B3 U-Net with CE + Dice and affine + CLAHE is worth +0.087 IoU and passes its gate (2026-11-11)

Brief: `docs/briefs/paper-adoption.md`, arm AB. It is kept local, because it builds on an
unpublished manuscript; the recipe below is complete without it.

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- 5-fold CV with the reference's folds;
- whole-image scoring by `run_cache` with `cnngvf.defaults()` post-processing;
- paired per image against the reference
  ([reference/2026-11-07-train-budget-69.csv](reference/2026-11-07-train-budget-69.csv));
- percentile bootstrap over images, 10 000 resamples, `rng(20260917)`, 95 % CI.

**The arm** is one bundle (`--recipe paper`), trained once (`--rng 0`) on a Colab L4. Its
cache is in `result/DRACTUH_unclustered/colab_bcedice/`.

| | reference | **arm** |
|---|---|---|
| network | MATLAB-replica U-Net, depth 4, no BN, from scratch | `smp.Unet("efficientnet-b3")`, ImageNet weights, 2 input channels, 13 M parameters |
| input | zero-centred | zero-centred and scaled to unit SD per channel |
| loss | `1 − generalizedDice`, per patch | CE + (1 − soft Dice pooled over the batch) |
| minibatch, rate | 2; 1e-3 → 1e-5 cosine | 8; 1e-4 → 1e-6 cosine |
| precision | fp32 | bf16 autocast in training |
| augmentation | ±32 px jitter, flips, `rot90` | the same, then affine p 0.75 (rotation ±15°, scale 0.9–1.1, shear ±5°) and CLAHE p 0.5 on intensity |

Unchanged: 40 epochs, the last epoch as output, no inner validation, `perSeed` 4,
`balanceByImage`, inference and post-processing.

**Gate:** the margin, decided before the run. A single-run baseline needs a paired ΔIoU CI
lower bound > 0 **and** a mean ≥ +0.015. This record adds that rule to SPEC Reporting rule 7.

**Per-image scores:** [2026-11-11-paper-recipe.csv](2026-11-11-paper-recipe.csv).

## Checks before scoring

- **Stall rule, relative form (brief):** no fold stalled. The last-five-epoch mean training
  losses are 0.044, 0.047, 0.047, 0.044 and 0.044. The median is 0.044 and the threshold
  (1.25 × median) is 0.055.
- **Provenance:** all 69 cached maps come from `rng 0 epoch 40`, recipe `paper`.
- **Folds:** the bundle's folds match [../folds.csv](../folds.csv), and every map is stamped
  with its own held-out fold. `run_cache`'s leakage guard passed.
- **Images without points:** `082`, `217` and `445` still predict 0 px, as they must with no
  seed.

## Result: all three gate conditions hold

| | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| reference | 0.5256 | 0.5527 | 0.6545 | 0.6672 | 0.6955 |
| **arm** | **0.6125** | 0.6494 | 0.7273 | 0.7418 | 0.7348 |

| paired ΔIoU | mean | 95 % CI | per-image SD | up / down / tied | Wilcoxon p |
|---|---|---|---|---|---|
| **arm − reference** | **+0.0869** | **[+0.0562, +0.1183]** | 0.133 | 48 / 16 / 5 | 5.7e-7 |

**Gate:**
1. **Margin: holds.** The CI lower bound is +0.056 and the mean is +0.087, about 6× the
   +0.015 margin.
2. **Trained cleanly: holds.** No fold stalled.
3. **Precision guard: holds.** Mean precision over Q1–Q3 (42 images) rose from 0.5863 to
   0.6729 (+0.087). The guard allows a fall of up to 0.01.

The five tied images are those `P` cannot reach: the three with no points, and `299` and
`573`, whose only points are off-lesion. They score 0 in both runs.

### Where the gain sits

By GT-area quintile (Q1 smallest, 14/14/14/14/13), against the reference:

| quintile | GT px | ΔIoU | Δ precision | Δ recall | share of the gain | median pred ÷ GT |
|---|---|---|---|---|---|---|
| Q1 | 601 – 4 157 | +0.054 | +0.075 | −0.019 | 13 % | 0.88 → 0.89 |
| Q2 | 4 176 – 10 374 | +0.122 | +0.113 | +0.075 | 28 % | 1.10 → 0.99 |
| Q3 | 10 645 – 23 655 | +0.054 | +0.071 | −0.004 | 13 % | 1.05 → 0.95 |
| Q4 | 23 663 – 50 314 | +0.094 | +0.099 | +0.019 | 22 % | 1.10 → 0.97 |
| Q5 | 51 180 – 432 602 | +0.113 | +0.009 | +0.133 | 24 % | 0.86 → 0.95 |

Grouped by the reference's own failure mode (the 66 images with points; the bands of
[2026-11-10](2026-11-10-train-budget-pmaps.md)):

| reference pred ÷ GT | n | ΔIoU | Δ precision | Δ recall | median pred ÷ GT |
|---|---|---|---|---|---|
| spill, above ×1.5 | 11 | **+0.191** | +0.256 | −0.063 | ×2.20 → ×1.17 |
| under-coverage, below ×0.67 | 9 | **+0.153** | −0.002 | +0.184 | ×0.52 → ×0.86 |
| in between | 46 | +0.055 | +0.051 | +0.038 | ×0.99 → ×0.97 |

| | n | reference | arm | ΔIoU | 95 % CI | F1 | precision | recall |
|---|---|---|---|---|---|---|---|---|
| DRAC22 | 42 | 0.4670 | 0.5404 | +0.073 | [+0.031, +0.121] | 0.592 → 0.659 | 0.629 → 0.696 | 0.605 → 0.647 |
| TUH | 27 | 0.6168 | 0.7245 | +0.108 | [+0.072, +0.146] | 0.753 → 0.834 | 0.727 → 0.813 | 0.836 → 0.871 |

These per-dataset numbers break out the pooled folds and are not a separate protocol.

| fold | n | reference | arm | Δ | images up |
|---|---|---|---|---|---|
| 1 | 14 | 0.5535 | 0.6645 | +0.111 | 12 |
| 2 | 14 | 0.5450 | 0.6061 | +0.061 | 10 |
| 3 | 14 | 0.5292 | 0.5794 | +0.050 | 6 |
| 4 | 14 | 0.4999 | 0.5109 | +0.011 | 7 |
| 5 | 13 | 0.4985 | 0.7081 | +0.210 | 13 |

- **Both failure modes shrink.** Spill images fall from ×2.20 to ×1.17 of GT through
  precision. Under-covered images rise from ×0.52 to ×0.86 through recall.
- **The size sweep is broad.** Every quintile gains; Q2 and Q5 most. Small lesions gain
  through precision and the largest through recall, so `P` is better placed rather than
  uniformly higher or lower.
- **Folds disagree.** The gain runs from +0.011 (fold 4) to +0.210 (fold 5, all 13 up). Every
  fold is positive.
- **Per-image movement:**
  - largest gains: `419` (+0.50), `795` (+0.46), `443` (+0.44), `nv (27)`, `1236`, `nv (16)`,
    `nv (3)`, `428` (+0.24 to +0.36);
  - largest losses: `576` (−0.19), `461` (−0.16), `1116` (−0.11), then five within −0.05.
    `576` was the budget run's largest gain.
- **Correlation with the reference:** per-image IoU correlates at 0.83, against 0.94–0.95
  between the budget run and its baselines. The new network errs differently, not just less.
- **Over the 66 images with points,** the mean is 0.6403 against 0.5495 (secondary figure).

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| no fold stalls | held |
| mean IoU 0.525–0.565, ΔIoU −0.005 to +0.040 | **wrong, above the range:** 0.6125, +0.087 |
| the gain comes through Q1–Q3 precision | held in part: Q1–Q3 precision +0.087, but Q5 recall (+0.133) carries as much |
| DRAC22 gains more than TUH | **wrong:** TUH +0.108, DRAC22 +0.073 |
| risk: ImageNet features weaken the heat-map, spill grows | **did not happen:** spill images fell from ×2.20 to ×1.17 of GT |
| about a 40 % chance of passing the gate | passed |

## Reading

- **The arm is adopted by its gate.** It gives 0.6125 IoU, +0.087 over the reference
  (69 images, 5-fold CV, CI [+0.056, +0.118]).
- **The bundle cannot be split.** The arm was trained once, so the shares of the encoder, the
  loss and the augmentation are unknown, as the brief accepted.
- **One run, but the margin is wide.** The training noise floor is about ±0.015, and the lower
  bound sits 0.041 above the margin. A second run would size the run-to-run spread of this
  recipe; it is not needed for the decision.
- **`tau` and `relaxFloor` were fitted to the old `P`.** With a CE term, `P` is a different
  distribution. Re-fitting both is the follow-up brief that adoption calls for.
- **What adoption requires,** per the brief:
  - the recipe becomes the trainer's only recipe, and the option goes;
  - the settings move into `cfg.Net` and the bundle;
  - a new reference CSV, `cfg.CV.reference`, and the README "Current result" table;
  - the arm's cache becomes `cv/probcache`;
  - `run_batch` and `run_steps` still load the MATLAB full-data network. MATLAB cannot load
    an EfficientNet-B3 U-Net written back from PyTorch the way it loaded the replica, so
    showing the adopted network there needs a Python-side `P` or an ONNX import. Decide that
    at adoption.
