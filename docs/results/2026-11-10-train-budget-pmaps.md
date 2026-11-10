# The training budget sharpens `P`, and the gain lands on the baseline's errors (2026-11-10)

Follows [2026-11-07-train-budget.md](2026-11-07-train-budget.md). That record proposed that
longer, decaying training "sharpens `P` overall" and left it untested. This record tests it.

**Domain and protocol:**
- Domain: the standardized 69 images (DRAC22 42 + TUH 27).
- `P`: the out-of-sample 5-fold CV `P` of each run. Fold 3 of the budget arm is its
  `--rng 1` retrain.
- Masks and scores: `run_cache` with `cnngvf.defaults()` post-processing.

**Comparison:** the budget arm against the per-image mean of the two PyTorch parity runs
(`parity_rng0`, `parity_rng1`).

**Code:** scratch Python, not committed. It reads the cached `P`, the GT masks in
`datasets/` and the three runs' `per_image.csv`.

**Prediction.** None was written. This analysis was run after the result was known, so it
explains that result and does not test a gate.

## `P` is more decisive

The table gives the mean over the 69 images. "Up / down" counts the images where the
budget arm's value is higher or lower than the baseline mean. p is a paired Wilcoxon test.

| `P` measure | budget | baseline | Δ | up / down | p |
|---|---|---|---|---|---|
| share of the `P` > 0.2 area with 0.2 < `P` < 0.8 | 0.676 | 0.763 | −0.086 | 5 / 61 | 2e-11 |
| mean `P` inside the GT | 0.560 | 0.511 | +0.049 | 53 / 11 | 4e-8 |
| mean `P` in a 64 px ring outside the GT | 0.090 | 0.101 | −0.011 | 28 / 38 | 0.13 |
| pixels with `P` > 0.5 outside the GT, ÷ GT area | 0.474 | 0.558 | −0.084 | 44 / 22 | 0.35 |
| share of GT pixels with `P` < 0.2 | 0.263 | 0.275 | −0.012 | 33 / 31 | 0.34 |

- **The sharpening is confirmed.** Fewer pixels are undecided on 61 of 66 images, and
  probability inside the lesion rises on 53.
- **Background suppression is only a trend.** The ring and the out-of-GT area fall, but
  neither change is significant.
- **The deep misses are untouched.** About a quarter of GT pixels stay below `P` 0.2. No
  amount of extra training reached them.

## The gain sits on the images the baseline got wrong

Images are grouped by the baselines' mean predicted ÷ GT area. The 3 images with no
localization points are left out, since every run scores 0 there.

| baseline on the image | n | ΔIoU | Δ precision | Δ recall | median area ratio, baseline → budget |
|---|---|---|---|---|---|
| spill: above ×1.5 | 15 | **+0.061** | +0.078 | −0.011 | ×2.12 → ×1.64 |
| under-coverage: below ×0.67 | 10 | **+0.052** | +0.011 | +0.065 | ×0.49 → ×0.55 |
| about right: ×0.67–×1.5 | 41 | +0.007 | −0.004 | +0.013 | ×0.97 → ×0.99 |

- **Spill is cut through precision, and under-coverage is filled through recall.** That
  answers the 2026-11-07 prediction miss ("mostly Q4–Q5 recall"): spill sits on small and
  mid-sized lesions, and it is fixed by precision, not recall.
- **Images the baseline already got right barely move.**

## It is not one retrain's luck

| | median per-image \|ΔIoU\| | images up by > 0.05 | images down by > 0.05 |
|---|---|---|---|
| budget − baseline mean | 0.036 | 25 | 6 |
| RNG 0 − RNG 1 (noise floor) | 0.034 | 11 | 15 |

- **Typical movement:** per-image changes are about as large as between two identical
  retrains.
- **Direction:** the budget arm's changes are lopsided upwards; the noise pair's are not.

**Fold 3, the retrained fold,** gains +0.016, against +0.027 for the other four folds
(Mann-Whitney p = 0.74). Using `--rng 1` on that fold did not shape the result.

**Fold 4 gains +0.070 against the baseline mean.** That number is inflated by RNG 0's
stalled fold 4. Against RNG 1 alone the gate already held (2026-11-07).

**By subgroup:**
- Dice: +0.025 overall, +0.032 on DRAC22 and +0.013 on TUH.
- By localization-point count, ΔIoU is flat: +0.021 to +0.032 in every bin that has
  points.

## Training had not converged

| fold | loss at epoch 20 | loss at epoch 40 | slope over the last 5 epochs |
|---|---|---|---|
| 1 | 0.267 | 0.233 | −0.0007 / epoch |
| 2 | 0.291 | 0.261 | −0.0006 |
| 3 (`--rng 1`) | 0.298 | 0.261 | −0.0005 |
| 4 | 0.281 | 0.249 | −0.0005 |
| 5 | 0.272 | 0.236 | −0.0006 |

The 20-epoch baselines ended at 0.28–0.31 (excluding RNG 0's stalled fold, at 0.39). The
loss is still falling at the bottom of the cosine schedule.

## Reading

- **Training longer makes `P` more decisive.** That fixes spill through precision and
  under-coverage through recall. This is the mechanism behind the +0.025 IoU of
  2026-11-07.
- **The deep misses (`P` < 0.2 inside the GT) are not reached.** A gain there needs
  something other than more training of the same kind.
- **Two follow-ups:**
  - A sharper `P` may move the best `tau` and `relaxFloor`. That is the planned re-fit
    brief.
  - The loss is still falling, so a longer budget (60 epochs, or `perSeed` 8) is a
    candidate next lever.
