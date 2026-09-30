---
title: Relax floor 0.15 vs 0.20, paired per image
kind: measurement-record
dataset: DRAC22 + TUH (68)
updated: 2026-09-30
---

# Relax floor 0.15 vs 0.20, paired per image

Follow-up to [2026-09-30-relax-grid](2026-09-30-relax-grid.md), which called the gap between
floors 0.10–0.20 at window 128 noise. Is (128, 0.15) at 0.5124 really indistinguishable from
the default (128, 0.20) at 0.5097?

**Post hoc.** 0.15 was chosen as the grid's surface peak on these same images, so this
comparison is tilted in its favour. It can show that 0.15 is *not* better. It cannot
establish that it is.

**Prediction (before running).** ΔIoU ≈ +0.0027, 95 % CI including 0.

**Protocol.** 68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` through `run_cache`,
whole-image scoring, `tau` 0.5, window 128. Paired per-image Δ = 0.15 − 0.20. Percentile
bootstrap 95 % CI on the mean, 10 000 resamples, `rng(20260917)`. Exact two-sided sign test,
ties excluded. Script: a scratch `.m` file, not committed. It calls `run_cache` twice and
does the arithmetic below.

## Result

| metric | 0.20 | 0.15 | Δ | 95 % CI | up / down / tie | sign *p* |
|---|---|---|---|---|---|---|
| IoU | 0.5097 | 0.5124 | +0.0027 | [−0.0033, +0.0095] | 17 / 16 / 35 | 1.000 |
| F1 | 0.6489 | 0.6500 | +0.0011 | [−0.0039, +0.0069] | 17 / 16 / 35 | 1.000 |
| precision | 0.6311 | 0.6155 | −0.0156 | [−0.0221, −0.0098] | 0 / 33 / 35 | <0.001 |
| recall | 0.7579 | 0.7791 | +0.0212 | [+0.0127, +0.0310] | 32 / 0 / 36 | <0.001 |

**On IoU it is noise; underneath it is a real trade.** The floor changes the mask on 33
images and leaves 35 identical. On every changed image, 0.15 has lower precision and higher
recall (except one image whose recall is unchanged), and the two effects cancel in IoU.

- **The whole mean gain comes from three images.** `nv (3)` +0.126, `nv (27)` +0.080 and
  `nv (21)` +0.062, all TUH, together contribute +0.0039 to the mean, more than the +0.0027
  total. Without them, 0.15 is slightly worse. Median Δ is 0.
- **The devices disagree:** DRAC22 −0.0033, CI [−0.0075, +0.0009], 6 up / 10 down. TUH
  +0.0117, CI [−0.0015, +0.0264], 11 up / 6 down. This matches 2026-09-16, where DRAC22 was
  best at 0.20 and TUH at 0.10.
- **Folds:** +0.0070, +0.0072, −0.0051, +0.0002, +0.0041.
- Largest losses: `nv (16)` −0.049, `535` −0.042, `nv (14)` −0.038, `144` −0.035,
  `nv (24)` −0.031.

## Conclusion

The IoU difference between 0.15 and 0.20 is not distinguishable from zero on 68 images, even
with selection tilting the comparison towards 0.15. The default stays at 0.20. The floor is
a precision / recall dial, not an IoU lever. If TUH-like data dominates a future domain, a
lower floor is worth a pre-registered test there.
