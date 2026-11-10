# Training twice as long with cosine decay is worth +0.025 IoU and passes its gate (2026-11-07)

> **Follow-up:** the "not tested here" reading of the mechanism is tested in
> [2026-11-10-train-budget-pmaps.md](2026-11-10-train-budget-pmaps.md): `P` is sharper, and the gain
> sits on the baseline's spill and under-coverage images.

Brief: [../briefs/train-budget.md](../briefs/train-budget.md) (levers analysis R2). Domain:
the standardized 69 images (DRAC22 42 + TUH 27). Protocol: 5-fold CV with the reference's
folds, whole-image scoring by `run_cache` with `cnngvf.defaults()` post-processing.

**The arm** is one bundle, trained once on a Colab L4 with the PyTorch trainer:
- 40 epochs instead of 20;
- cosine learning rate, 1e-3 → 1e-5;
- the last epoch as the output network;
- the inner-validation images returned to training.

Everything else is unchanged from the parity runs.

**Baselines:** the two PyTorch parity runs, `parity_rng0` and `parity_rng1`
([2026-11-05](2026-11-05-colab-parity-cv.md)). Both were re-scored here and reproduce that
record (0.5002 and 0.5010).

**Statistics:** the bootstrap uses 10 000 resamples over the 69 images, `rng(20260917)`.

**Per-image scores:** [2026-11-07-train-budget.csv](2026-11-07-train-budget.csv).

## Stall rule: fold 3 stalled and was retrained

The stall rule was applied to every fold's `log.csv` before any scoring.

| fold | mean training loss, last five epochs | |
|---|---|---|
| 1 | 0.234 | |
| 2 | 0.262 | |
| 3, `--rng 0` | **0.381** | stalled: about 0.40 from epoch 3 to 40 |
| 3, retrain with `--rng 1` | 0.262 | used, as the rule requires |
| 4 | 0.250 | |
| 5 | 0.237 | |

- **The stall threshold is 0.35.**
- **The stalled run left no trace on disk.** The retrain was copied over the stalled fold's
  folder, so the stalled `log.csv` and `P` no longer exist. The 0.381 figure is from the
  log read on 2026-11-07, quoted in commit `867efc3`.
- **Provenance of the scored `P`:** every cached `P` names its network. The 14 fold-3 maps
  come from `fold 3 rng 1 epoch 40`; all others come from `rng 0 epoch 40`. All carry the
  arm's options.
- **Stalls so far:** 2 of 16 PyTorch fold trainings, both at the first attempt (parity
  RNG 0 fold 4, budget fold 3).

## Result: all three gate conditions hold

| run | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| PyTorch parity RNG 0 | 0.5002 | 0.5267 | 0.6296 | 0.6431 | 0.6870 |
| PyTorch parity RNG 1 | 0.5010 | 0.5252 | 0.6298 | 0.6591 | 0.6738 |
| **budget arm** | **0.5256** | 0.5527 | 0.6545 | 0.6672 | 0.6955 |
| MATLAB reference (description only) | 0.5047 | 0.5556 | 0.6328 | 0.6410 | 0.6975 |

| paired ΔIoU | mean | 95 % CI | per-image SD | up / down | Wilcoxon p |
|---|---|---|---|---|---|
| **budget − RNG 0** | **+0.0254** | **[+0.0077, +0.0438]** | 0.077 | 38 / 26 | 0.0095 |
| **budget − RNG 1** | **+0.0246** | **[+0.0083, +0.0411]** | 0.070 | 45 / 19 | 0.0026 |
| budget − mean of the two | +0.0250 | [+0.0096, +0.0406] | 0.066 | 45 / 19 | 0.0022 |
| budget − MATLAB reference (description only) | +0.0209 | [+0.0046, +0.0381] | 0.073 | 40 / 24 | — |
| RNG 0 − RNG 1 (noise floor) | −0.0008 | [−0.0153, +0.0140] | 0.062 | 29 / 35 | — |

**Gate:**
1. **Beats both baselines: holds.** Both CI lower bounds are above 0 (+0.0077, +0.0083).
   The margin over RNG 0 is not an artefact of that run's stalled fold 4: against RNG 1,
   which has no stalled fold, the lower bound is +0.0083.
2. **Trained cleanly: holds.** After the stall rule, every fold ends at 0.234–0.262.
3. **Precision guard: holds.** Mean precision over Q1–Q3 (42 images) is 0.5863, against
   0.5767 for the mean of the two baselines (RNG 0 0.5628, RNG 1 0.5907). It rose by
   +0.0096; the guard allows a fall of up to 0.01.

**The gain is about 1.7× the noise floor.** The CI half-width of a single retrain is about
0.015.

### Where the gain sits

The table compares against the mean of the two baselines, by GT-area quintile (Q1
smallest, 14/14/14/14/13):

| quintile | GT px | ΔIoU | Δ precision | Δ recall | share of the gain |
|---|---|---|---|---|---|
| Q1 | 601 – 4 157 | +0.008 | +0.006 | +0.013 | 7 % |
| Q2 | 4 176 – 10 374 | +0.035 | +0.008 | +0.020 | 28 % |
| Q3 | 10 645 – 23 655 | +0.027 | +0.015 | +0.009 | 22 % |
| Q4 | 23 663 – 50 314 | +0.030 | +0.024 | +0.023 | 24 % |
| Q5 | 51 180 – 432 602 | +0.026 | +0.029 | +0.010 | 19 % |

| | n | budget | RNG 0 | RNG 1 | MATLAB reference |
|---|---|---|---|---|---|
| DRAC22 | 42 | 0.4670 | 0.4343 | 0.4385 | 0.4357 |
| TUH | 27 | 0.6168 | 0.6026 | 0.5982 | 0.6121 |

These per-dataset numbers break out the pooled folds and are not a separate protocol.

| fold | n | budget | RNG 0 | RNG 1 | MATLAB reference |
|---|---|---|---|---|---|
| 1 | 14 | 0.5535 | 0.5364 | 0.5144 | 0.5440 |
| 2 | 14 | 0.5450 | 0.5315 | 0.5465 | 0.5313 |
| 3 | 14 | 0.5292 | 0.5032 | 0.5231 | 0.5116 |
| 4 | 14 | 0.4999 | 0.4058 | 0.4545 | 0.4724 |
| 5 | 13 | 0.4985 | 0.5258 | 0.4636 | 0.4611 |

- **The gain is broad.** Q2–Q5 each gain about +0.03, and only Q1 is near flat. DRAC22
  gains more than TUH.
- **Precision rises with size.** In Q4–Q5 it rises more than recall (+0.026 against +0.017).
- **The median predicted ÷ GT area barely moves:** 0.99, against 1.02 and 0.96.
- **Per-image movement:**
  - largest gains: `576` (+0.23), `nv (1)`, `nv (3)`, `559`, `134`, `664`, `nv (9)`, `419`
    (+0.10 to +0.14);
  - largest losses: `312` (−0.15), `421`, `nv (12)`, `nv (14)`, `nv (16)` (−0.11 to −0.12).
- **Correlation with the baselines:** per-image IoU correlates at 0.944 and 0.954.

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| no fold stalls | **wrong:** fold 3 stalled and needed the retrain |
| healthy folds end at training loss 0.22–0.27 | held: 0.234–0.262 |
| mean IoU 0.510–0.525, ΔIoU +0.010 to +0.025 per baseline | at the top edge: 0.5256; +0.0254 and +0.0246 |
| the gain comes mostly from Q4–Q5 recall | **wrong:** Q4–Q5 carry 43 % of the gain, and their precision rose more than their recall (+0.026 vs +0.017) |
| Q1–Q3 precision moves by less than ±0.01 | held, just: +0.0096 (Q3 alone +0.015) |
| about an even chance of passing the gate | passed |

## Reading

- **R2 is adopted by its gate.** Training twice as long with cosine decay, a last-epoch
  output and the inner images in training gives 0.5256 IoU. That is +0.025 over each PyTorch
  baseline (69 images, 5-fold CV, both CIs above 0), and +0.021 over the MATLAB reference
  (description only: a different trainer).
- **The bundle cannot be split.** It was trained once, so which of the four changes carries
  the gain is unknown, as the brief accepted.
- **The mechanism is not what the brief predicted.** The gain is broad across sizes, and it
  comes from precision as much as from recall. A plausible reading is that longer, decaying
  training sharpens `P` overall, rather than lifting the low-`P` interiors of large lesions.
  This is not tested here.
- **Stalls are now a known trainer risk:** 2 of 16 fold trainings. The pre-registered stall
  rule handled this one. The rule should stay in any future training brief.
- **What adoption requires,** per the brief:
  - the options become the trainer's defaults (switches removed), with the settings in
    `cnngvf.defaults()` and the bundle;
  - a new reference CSV, `cfg.CV.reference`, and the README "Current result" table;
  - writing the PyTorch weights back into MATLAB networks, so that `run_batch` and
    `run_steps` use the adopted networks. The check: MATLAB inference reproduces this cache.
- **Re-fitting `tau` and `relaxFloor` on the new `P`** is a follow-up brief.
