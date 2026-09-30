---
title: Seed hull claim rule, with and without the fallback disk
kind: measurement-record
dataset: DRAC22 + TUH (68)
updated: 2026-10-01
---

# Seed hull claim rule, with and without the fallback disk

Brief, prediction and gate: [../briefs/seed-hull-claim.md](../briefs/seed-hull-claim.md),
committed before any run. Code: tag `archive/seed-hull-claim`.

**Question.** The 128 px seed hull was adopted over a failed precision guard
([2026-09-30-seed-hull-d](2026-09-30-seed-hull-d.md)). Does filling only triangles whose
seeds claimed the same component (arm R) repair that without losing IoU? And does also
dropping the fallback disk (arm RN) do better?

**Protocol.** 68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` through `run_cache`,
whole-image scoring, `tau` 0.5, relax 0.20, `hullLink` 128. Each arm is paired per image
against the current configuration, which reproduces `2026-09-30-hull-128.csv` to 5e-16.
Percentile bootstrap, 10 000 resamples, `rng(20260917)`, 97.5 % CIs (Bonferroni over two
arms); exact two-sided sign test. Script: a scratch `.m` file, not committed.

## Result: both arms fail non-inferiority. Reject both.

| arm | metric | current | arm | Δ | 97.5 % CI | up / down / tie |
|---|---|---|---|---|---|---|
| **R** | IoU | 0.5207 | 0.5179 | −0.0028 | [−0.0065, +0.0004] | 3 / 10 / 55 |
| | precision | 0.6268 | 0.6308 | +0.0040 | [−0.0001, +0.0115] | 7 / 6 / 55 |
| | recall | 0.7762 | 0.7693 | −0.0069 | [−0.0124, −0.0026] | 0 / 13 / 55 |
| **RN** | IoU | 0.5207 | 0.5162 | −0.0045 | [−0.0098, −0.0002] | 5 / 11 / 52 |
| | precision | 0.6268 | 0.6322 | +0.0054 | [+0.0005, +0.0131] | 11 / 5 / 52 |
| | recall | 0.7762 | 0.7661 | −0.0101 | [−0.0175, −0.0043] | 0 / 16 / 52 |

| gate | R | RN |
|---|---|---|
| 1. ΔIoU CI lower bound > −0.002 | −0.0065, **fail** | −0.0098, **fail** |
| 2. precision, three smallest GT-area quintiles, vs no hull 0.5402 (limit 0.5302) | 0.5400, pass | 0.5420, pass |

**RN − R** (description): ΔIoU −0.0017, 95 % CI [−0.0038, −0.0000], 2 up / 7 down; precision
+0.0014 [+0.0000, +0.0036]; recall −0.0032. RN excludes 40 seeds, as in 2026-09-18.

### Against the predictions

- **R, main:** predicted ΔIoU −0.001 to +0.003, non-inferior. Measured −0.0028, which fails.
  **Wrong.**
- **R, guard:** predicted back within −0.01, at 60 % confidence. It came back almost exactly
  to no hull, at −0.0002. **Right.** The hull's precision cost on smaller lesions came
  entirely from the triangles this rule blocks.
- **RN − R:** Claude bet on IoU within ±0.001; the user bet RN wins on precision and IoU.
  RN wins on precision (+0.0014) and loses on IoU (−0.0017). Each bet was half right.

### What the blocked triangles were

The rule removed pixels on 13 images. Each removed pixel is attributed to the class of
triangle that had filled it (GT used here for diagnosis):

| class | removed px | fraction in GT |
|---|---|---|
| fragment join (smaller component < 2 000 px) | 28 557 | **0.819** |
| lesion-to-lesion (both components ≥ 2 000 px) | 27 171 | 0.501 |
| *the whole current hull, for comparison* | 175 212 | 0.771 |

- **Fragment joins are the hull's best pixels.** 82 % are lesion, above the hull's average.
  The fallback-disk fragments (709 px) are almost entirely lesion: `nv (13)` 1.00, `nv (3)`
  1.00, `1115` 0.97. `211` is the exception at 0.46. This **refutes the open hypothesis**
  that joining fallback-disk fragments lowers precision: those fragments sit inside the
  lesion, in its inter-frond gaps. The blind impression from the GT-free sheet was right.
- **Lesion-to-lesion joins are a coin flip,** at 50 % GT. That still raises IoU: at a mean
  IoU near 0.52, adding pixels that are more than about a third lesion helps. It costs
  precision.
- **The one bad fragment join** is `429`'s 1 683 px blob, at 24 % GT. It is a thresholded
  blob across a vessel, not a disk, and blocking it gains +0.046. Only `429` (+0.046), `458`
  (+0.011) and `nv (23)` (+0.003) improve under R.
- **Largest losses under R:** `nv (3)` −0.059, `nv (20)` −0.056, `1236` −0.030, `nv (17)`
  −0.028, `461` −0.022, `nv (13)` −0.020.
- **Size and device:** by GT-area quintile, Q5 −0.013, Q3 −0.005, Q2 +0.004, Q1 and Q4 ≈ 0.
  TUH −0.0069, 95 % CI [−0.0134, −0.0016]; DRAC22 −0.0001.

## Conclusion

Neither arm is adopted, and `hullLink` 128 with all triangles stays. The claim rule does what
it was built for on precision: the guard the hull failed is back to no-hull level. But the
triangles it blocks are mostly good fills, so it costs 0.0028 IoU, which is more than the
margin allowed.

Two things are now known and were not before:

1. **Fallback-disk fragments belong to their lesions.** A future change should *keep*
   joining them. Dropping the disk (RN) removes lesion.
2. **The hull's precision cost comes from lesion-to-lesion triangles,** which are half
   lesion. Distinguishing the good half needs something other than claims, since both
   sides are real lesion components. A rule chosen after seeing this split would need its
   own brief, and confirmation beyond these 68 images.
