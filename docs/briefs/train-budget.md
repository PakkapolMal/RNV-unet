---
title: Training Budget, LR Decay, Last-epoch Output
description: Pre-registered single-arm test of levers analysis R2 on the PyTorch trainer. Twice the epochs, cosine learning-rate decay, the last epoch as the output network, and the inner-validation images returned to training. Bundle, stall rule, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/colab (trainer flags), Colab L4 (user runs), RNV-Claude/cnn (scoring)
depends-on: results/2026-11-05-colab-parity-cv.md, results/2026-09-01-filters-perseed-levelset.md, results/2026-11-05-tau-crossfit.md, research_analysis/rnv-score-levers-2026-10-29.md
status: pre-registered 2026-11-05, not yet run
---

# Training Budget, LR Decay, Last-epoch Output

## Why

Every remaining in-scope loss is in `P`. The two post-`P` checks just closed say the same:
- `tau` is already optimal (2026-11-05);
- the TUH overlay is harmless (2026-11-05).

The trainer has never been given more than 20 epochs under CV, and its own logs say that is
too few.

**Evidence from the two PyTorch baseline runs** (69 images, 5-fold CV, 10 fold trainings):
- **The loss is still falling at the end.** Training loss is still dropping at epoch 20 in
  every healthy fold, by about 0.001–0.002 per epoch over the last five.
- **Best-validation picks on noise.** It picked epochs 7–17. The inner validation is 2
  images (about 11 seeds), and its loss swings far more than it trends: fold 5 of RNG 0
  ranges 0.38–0.58 over epochs 3–20.
- **Selection costs training data.** The 2 inner-validation images per fold train nothing.

**Earlier MATLAB evidence:**
- `perSeed` 1 → 4, which is mostly "more gradient steps on freshly jittered patches", was
  worth +0.141 IoU.
- At `perSeed = 4` the validation curve was still descending at epoch 29, with a wobble of
  ±0.03 (2026-09-01).

## Arm: one bundle, judged as one

Levers analysis R2 bundles four changes. Training once per arm (SPEC, Reporting rule 7)
cannot separate them, so they are tested together and adopted or rejected together.

| | baseline (parity) | **arm** |
|---|---|---|
| epochs | 20 | **40** |
| learning rate | constant 1e-3 | **cosine 1e-3 → 1e-5**, stepped once per epoch (`CosineAnnealingLR`, `T_max` = 40, `eta_min` = 1e-5) |
| output network | best validation on the inner images | **last epoch** |
| inner-validation images | held out for selection | **returned to training** |
| everything else | `perSeed` 4, `balanceByImage`, minibatch 2, Adam with L2 1e-4 on weights, fp32 with TF32 off, the same folds | unchanged |

The epoch's patch list is still drawn once (`prepareData`'s rule). It now includes the
returned images' seeds, so it grows by about 4 % per fold. The test images of each fold are
unchanged and never seen.

**Not changed, so there is one arm and not several:**
- `perSeed` 8, which the levers analysis offers as an equivalent;
- piecewise decay;
- TF32 or mixed precision;
- the post-processing values (`tau`, `relaxFloor`, `hullLink`), which stay at their adopted
  values. A change to `P` may shift their optimum. Re-fitting them is a separate brief if
  this arm is adopted.

## Implementation

All the changes go into `colab/rnvtorch.py` as `train` options. The defaults reproduce the
parity runs exactly.
- `--epochs 40` already exists.
- `--lr-schedule {constant, cosine}`, default `constant`.
- `--output {best-validation, last}`, default `best-validation`.
- `--no-inner-val`: the inner images train. Requires `--output last`, because nothing is
  left to select on.

With no validation set, `log.csv` keeps the `val_loss` column empty and adds `lr`.

**Verification before any Colab run:**
1. `test_parity.py`, stages 1–2, still passes. The shared code is unchanged.
2. A local 2-epoch smoke run of fold 1 with all four options. Checks:
   - the patch count equals the parity count plus the inner images' usable seeds × `perSeed`
     share, computed independently;
   - the learning rate follows the cosine;
   - `net.pt` holds the last epoch;
   - resume works mid-schedule (stop after 1, resume to 2) and restores the scheduler state.

**Colab:** the notebook gets a stage cell:

```
!python {DRIVE}/colab/rnvtorch.py train /content/bundle {DRIVE}/runs/budget_rng0 \
    --fold {FOLD} --rng 0 --epochs 40 --lr-schedule cosine --output last --no-inner-val --workers 2
```

**Cost:** about 300 s/epoch × 40 ≈ 3.3 h per fold and about 17 h per CV on an L4. That is
about the GPU time of both parity CVs together (about 16 h), so check the L4's CU rate in
Colab before starting. One CV, `--rng 0` only.

## Stall rule (pre-registered; uses only training loss)

The parity runs had one stalled fold in ten: RNG 0, fold 4. Over the last five epochs its
mean training loss was 0.392, while the nine healthy folds' last-five means were
0.279–0.320.

**Before any scoring,** a fold is stalled if its mean training loss over its last five epochs
exceeds **0.35**, which lies between the two groups. A stalled fold is retrained once from scratch with `--rng 1`, and that
retrain is used whatever its score. If the retrain also stalls, the arm is reported as a
training failure and nothing is scored.

The rule reads only training loss, never a held-out image, so it selects nothing on the
outcome.

## Protocol

- 69 images (DRAC22 42, TUH 27), 5-fold CV, whole-image scoring by `run_cache` with
  `cnngvf.defaults()` post-processing.
- Compare paired per image against **each** baseline run (`colab_baseline/parity_rng0` and
  `parity_rng1`).
- Percentile bootstrap over images: 10 000 resamples, `rng(20260917)`, 95 % CI.
- Report against the MATLAB reference as description only: it is a different trainer.
- Size quintiles are by GT area: Q1 smallest, 14/14/14/14/13.

## Prediction (written before running)

- **No fold stalls.** Healthy folds end at training loss 0.22–0.27.
- **Mean IoU 0.510–0.525.** That is ΔIoU of +0.010 to +0.025 against each baseline run.
- **The gain comes mostly from Q4–Q5 recall**, where 76 % of Q5's missed pixels sit below
  `P` 0.20.
- **Q1–Q3 precision changes by less than ±0.01.**
- I give the gate below about an even chance. The predicted gain straddles the noise floor
  (CI half-width about 0.015 for a single retrain), so a real +0.01 may not clear it.

## Gate

Adopt only if all three hold:

1. **Beats both baselines:** paired ΔIoU against each of `parity_rng0` and `parity_rng1`
   has a 95 % CI lower bound > 0.
2. **Trained cleanly:** no fold is stalled after the stall rule.
3. **Precision guard:** mean precision over Q1–Q3 falls by no more than 0.01 against the
   mean of the two baselines' Q1–Q3 precision.

**Otherwise reject.** Tag the trainer options `archive/train-budget`, delete them and record
the result.

**On adoption:**
- The options become the trainer's defaults and the switches go. The settings move into
  `cnngvf.defaults()` (`cfg.Net`) and reach the trainer through `run_export`'s bundle, so the
  defaults stay the configuration.
- The arm's per-image scores become a new `docs/results/reference/*.csv`, with
  `cfg.CV.reference` and the README "Current result" table updated.
- **Before that adoption is complete,** `run_batch` and `run_steps` must load the adopted
  networks. Today they load MATLAB networks. The architecture is unchanged, so the PyTorch
  weights can be written back into a MATLAB `unet` through `net.Learnables` and the input
  mean (the reverse of `UNet.load_matlab`). That needs its own small change and check:
  MATLAB inference on the written-back weights reproduces the arm's cached `P` (CPU, about
  1e-6).
- Re-fitting `tau` and `relaxFloor` on the new `P` is a follow-up brief, not part of this
  gate.
