---
title: Relax window × floor grid
kind: measurement-record
dataset: DRAC22 + TUH (68)
updated: 2026-09-30
---

# Relax window × floor grid

The relax fallback's window (±128 px) was never measured, and its floor (0.20) was swept
only at that one window ([2026-09-16](2026-09-16-reduce.md)). This grid measures both.

## Pre-registration (written before running)

**Grid.** `relaxWindow` ∈ {64, 128, 256, 512} × `relaxFloor` ∈ {0.05, 0.10, 0.15, 0.20, 0.25,
0.30}: 24 cells. `tau` stays 0.5; everything else is `cnngvf.defaults()`.

**Protocol.** 68 images (DRAC22 41, TUH 27), whole-image scoring, the cached 5-fold CV `P`
scored with `run_cache`, so every `P` comes from the fold network that held that image out.
The cell is **cross-fitted**: for each fold, choose the cell with the best mean IoU on the
other four folds, then score it on the held-out fold. Cells within 0.001 IoU of the best
count as tied, and the current default (128, 0.20) wins a tie. The full 24-cell surface is
reported as a surface, never as a result: it is tuned on the images it scores.

**Comparator.** The current default (128, 0.20) at a fixed setting: 0.5097. Paired per-image
ΔIoU, bootstrap 95 % CI (10 000 resamples).

**Gate.** Adopt a new cell only if both hold:

1. cross-fitted ΔIoU vs the default > 0, with the 95 % CI excluding 0;
2. at least 4 of 5 folds choose the same cell, which becomes the new default.

Otherwise the default stays, and the window is now measured rather than assumed.

**Sanity check.** The four cells at window 128 and floors 0.05 / 0.10 / 0.20 / 0.30 must
reproduce 2026-09-16: 0.4863 / 0.5101 / 0.5097 / 0.4909.

**Prediction.** The window matters little at floors ≥ 0.15, where the component under a seed
is local anyway. At 0.05–0.10 a larger window lets the component flood a plateau: precision
falls. Window 64 clips large lesions: recall falls on the biggest quintile. The best surface
cell lies at window 128–256, floor 0.10–0.20, within ±0.005 of 0.5097. Cross-fitted ΔIoU vs
the default: −0.005 to +0.005, CI including 0. The gate fails and the default stays.

## Result

**The gate fails. The default stays (128, 0.20).** Tuning both parameters costs more in
selection than it gains: cross-fitted, the grid scores **0.4999** against the fixed default's
**0.5097**, ΔIoU −0.0098, 14 up / 19 down, 95 % CI [−0.0228, +0.0030]. The folds do not
agree on a cell (below), so criterion 2 fails as well.

All numbers: 68 images (DRAC22 41, TUH 27), cached 5-fold CV `P`, whole-image scoring,
`tau` 0.5. Code: `run_relax_grid.m` at tag `archive/relax-grid`, scored through `run_cache`.

**Sanity check passes.** At window 128, floors 0.05 / 0.10 / 0.20 / 0.30 give 0.4863 / 0.5101 /
0.5097 / 0.4909, the 2026-09-16 values. The default cell matches the reference CSV to 5e-16
per image.

### Cross-fitted

| | IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| default (128, 0.20), fixed | 0.5097 | 0.5299 | 0.6489 | 0.6311 | 0.7579 |
| grid, cell cross-fitted per fold | 0.4999 | 0.5276 | 0.6383 | 0.6020 | 0.7783 |

| fold | chosen (window, floor) | IoU on the other 4 folds | IoU on the held-out fold |
|---|---|---|---|
| 1 | 128, 0.10 | 0.5366 | 0.4078 |
| 2 | 128, 0.15 | 0.4877 | 0.6076 |
| 3 | 128, 0.10 | 0.5063 | 0.5250 |
| 4 | 64, 0.10 | 0.5185 | 0.4508 |
| 5 | 128, 0.15 | 0.5141 | 0.5052 |

Per dataset, cross-fitted vs default: DRAC22 0.4642 vs 0.4792, TUH 0.5542 vs 0.5561.
Cross-device: choose on TUH, score DRAC22 → (128, 0.15), 0.4759 vs default 0.4792. Choose
on DRAC22, score TUH → (128, 0.20), the default itself.

### Surface (tuned on the images it scores; not a result)

Mean IoU, 68 images:

| window \ floor | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|
| 64 | 0.4986 | 0.5055 | 0.5055 | 0.4996 | 0.4914 | 0.4835 |
| **128** | 0.4863 | 0.5101 | **0.5124** | *0.5097* | 0.5012 | 0.4909 |
| 256 | 0.4646 | 0.4970 | 0.5057 | 0.5045 | 0.4984 | 0.4899 |
| 512 | 0.4580 | 0.4936 | 0.5036 | 0.5027 | 0.4977 | 0.4916 |

Precision / recall by floor move as expected at every window: 0.05 gives recall 0.77–0.83
at precision 0.53–0.60; 0.30 gives recall 0.69–0.72 at precision 0.65–0.66. Per dataset,
per cell, and precision / recall per cell: `surface.csv` from the runner.

### Against the prediction

- **Right:** the surface peak is (128, 0.15), +0.0027 over the default, inside ±0.005. Larger
  windows at low floors flood: at floor 0.05, precision falls 0.60 → 0.53 from window 64 to 512.
  Window 64 costs recall: at floor 0.20 it is 0.727 against 0.758 at 128. The gate fails.
- **Wrong:** the window matters at floors ≥ 0.15 too. At 0.15 and 0.20, 128 beats both
  neighbours by 0.005–0.010, so it is an interior optimum and not an arbitrary choice that
  happened to work. The cross-fitted loss (−0.0098) is also larger than predicted, and fold 4,
  which chose window 64, contributes to it.

### What this settles

- **±128 px is now measured, not assumed:** it is the best window at every floor from 0.10
  to 0.25 on this `P` (at 0.30 window 512 edges it, 0.4916 vs 0.4909).
- **The floor surface is flat between 0.10 and 0.20 (0.5101–0.5124),** so which of them the
  folds pick is noise. Here, selecting a cell per fold cost 0.0098 IoU, which is why the
  fixed setting is kept.
- Caveat on "pre-registered": 0.20 was fixed after the 2026-09-16 floor sweep had been seen,
  and that sweep scored these same 68 images. The cross-fitted figures (0.5061 floor only,
  2026-09-17; 0.4999 floor × window, here) are the honest estimates of a tuned relax.
