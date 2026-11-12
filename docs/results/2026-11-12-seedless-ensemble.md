# Averaging the two seedless networks gains +0.003 IoU, CI including 0: one run stays (2026-11-12)

Brief: [../briefs/seedless-ensemble.md](../briefs/seedless-ensemble.md).

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- the seedless `P` replaced by the mean of the two seedless runs (rng 0, the adopted cache;
  rng 1, [2026-11-12](2026-11-12-fusion-confirm.md)), in both stage 4a and stage 9;
- thresholds unchanged (`fuseTauS` 0.9, `noPointTau` 0.40), one cell, nothing selected;
- whole-image scoring by `run_cache`, paired against
  [reference/2026-11-12-fusion-69.csv](reference/2026-11-12-fusion-69.csv);
- bootstrap 10 000 resamples, `rng(20260917)`.

**Per-image scores:** [2026-11-12-seedless-ensemble.csv](2026-11-12-seedless-ensemble.csv).

**The averaged cache:** both runs stamp every image with the same fold. For the median image,
14 % of the non-trivial pixels lie near 0.5, where only one run is confident. The cache is kept
outside git, in `result/DRACTUH_unclustered/seedless_ensemble/`.

## Result: the gate fails

| | IoU | precision | recall | DRAC22 | TUH |
|---|---|---|---|---|---|
| reference (rng 0 seedless) | 0.6484 | 0.7636 | 0.7806 | 0.5975 | 0.7277 |
| averaged seedless | 0.6514 | 0.7760 | 0.7819 | 0.6021 | 0.7280 |

**ΔIoU +0.0030, 95 % CI [−0.0042, +0.0149];** 26 images up, 33 down.

**Gate:**
1. **CI lower bound above 0: fails** (−0.0042).
2. **Precision guard: holds.** Q1–Q3 precision +0.0194.

## Where it moves

| part | images | contribution to the 69-image mean |
|---|---|---|
| stage 4a (images with points) | 66 | **−0.0023** |
| stage 9 (images with no points) | 3 | **+0.0052** |

- **Stage 9 gains, all of it from `445`:** 0 → 0.358 (2 209 px predicted). `082` goes 0.788 →
  0.795 and `217` 0.863 → 0.857.
- **Stage 4a loses, on the largest lesions:** `461` −0.068, `144` −0.032, `795` −0.026,
  `1115` −0.015, `125` −0.014. At `fuseTauS` 0.9 the mean admits only pixels both runs agree
  on, so fusion adds less where the runs differ.
- **By quintile:** Q1 +0.022 (`445` is in Q1); Q2–Q4 within ±0.002; Q5 −0.009 (recall
  −0.011).

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| stage 4a loses a little: −0.004 to +0.002 | held: −0.0023 |
| `445` rises to about 0.2–0.4; `082` and `217` within ±0.03 | held: 0.358; +0.007 and −0.006 |
| headline −0.003 to +0.006, CI including 0 | held: +0.0030, [−0.0042, +0.0149] |
| about a 30 % chance of passing | failed |

## Reading

- **The adopted seedless cache stays the rng 0 run alone.** Averaging the two runs is worth
  +0.0030 IoU (69 images, 5-fold CV, CI including 0).
- **The two stages want opposite things from an ensemble.** Stage 9 gains from a union-like
  map (either run is enough at 0.40). Stage 4a loses from an agreement-like one (both runs
  needed at 0.9).
- **Splitting the stages is not supported by this run.** An ensemble for stage 9 only, with
  rng 0 alone for 4a, would add the +0.0052. But its whole effect is one image, `445`. That
  split was seen here, not pre-registered, and scoring it on these images would be circular.
  It needs new point-less images.
- **The `445` swing is training noise, not a fix.** One run sees half of it, the other none.
  A stage 9 figure resting on one seedless run should be read with that spread.
- **Valid until a seedless `P` changes.**
