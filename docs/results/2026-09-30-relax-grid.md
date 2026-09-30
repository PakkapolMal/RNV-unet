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
