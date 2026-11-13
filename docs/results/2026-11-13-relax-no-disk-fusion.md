---
title: Relax Without the Disk, Under Fusion
description: The pre-registered arm (no disk, no claim) fails both gates. A variant found while implementing it (no disk, the seed still claims the nearest component) was adopted after scoring, by the user's decision; most of its gain is three images.
kind: measurement-record
status: pre-registered arm rejected; post-hoc variant adopted
dataset: DRAC22 + TUH (69)
updated: 2026-11-13
---

# Relax Without the Disk, Under Fusion

Brief, predictions and gate: [../briefs/relax-no-disk-fusion.md](../briefs/relax-no-disk-fusion.md),
committed before any run. Pre-registered arm's code: tag `archive/relax-no-disk-fusion`.

**Question.** Under the current pipeline (fusion at stage 4a, stage 9), should a seed relax
cannot serve get nothing instead of the 15 px disk?

**Protocol.** As in the brief: 69 images, cached 5-fold CV `P`, `run_cache`, paired per image
against `reference/2026-11-12-fusion-69.csv` (reproduced to 4e-16). Percentile bootstrap,
10 000 resamples, `rng(20260917)`, 95 % CI; exact two-sided sign test. Dice is F1. Script: a
scratch `.m` file, not committed.

## Result

| arm | metric | reference | arm | Δ | 95 % CI | up / down / tie |
|---|---|---|---|---|---|---|
| **A: no disk, no claim** (pre-registered) | IoU | 0.6484 | 0.6510 | +0.0026 | [−0.0059, +0.0126] | 19 / 18 / 32 |
| | Dice | 0.7621 | 0.7625 | +0.0005 | [−0.0075, +0.0082] | 19 / 18 / 32 |
| | precision | 0.7636 | 0.7887 | +0.0252 | [+0.0116, +0.0418] | 31 / 6 / 32 |
| | recall | 0.7806 | 0.7601 | −0.0205 | [−0.0408, −0.0074] | 0 / 30 / 39 |
| **B: no disk, nearest claim** (post hoc) | IoU | 0.6484 | **0.6604** | +0.0120 | [+0.0009, +0.0258] | 22 / 16 / 31 |
| | Dice | 0.7621 | 0.7726 | +0.0105 | [−0.0010, +0.0244] | 22 / 16 / 31 |
| | precision | 0.7636 | 0.7900 | +0.0264 | [+0.0117, +0.0453] | 33 / 5 / 31 |
| | recall | 0.7806 | 0.7782 | −0.0024 | [−0.0133, +0.0106] | 2 / 29 / 38 |

| gate | A | B (not pre-registered) |
|---|---|---|
| 1. ΔIoU CI lower bound > −0.002 | −0.0059, **fail** | +0.0009, pass |
| 2. Δrecall ≥ −0.005 | −0.0205, **fail** | −0.0024, pass |

**Arm A is rejected.**

### How B arose

The first flag only skipped the `seedDisks` call. A seed with no disk stands on background, so
`seedMasks` lets it claim the nearest qualifying component under the boundary rule, which has
no distance cap. That is arm B. The brief, and both earlier tests, specified A: the seed is
also kept out of claiming. A was then implemented as in `fe4c3c0` and scored. B was scored
first, by mistake, so its result was seen before anything about it was fixed.

### Arm B is adopted post hoc, by the user's decision

Claude advised against it: B was chosen after scoring, and no unseen data exists to confirm
it. **Most of B's gain is three images:**

| image | seeds | reference | B | note |
|---|---|---|---|---|
| `125` | 6 | 0.479 | 0.756 | seed 5 is stray (2026-09-17 refine); B also adds 5 130 px |
| `428` | 3 | 0.653 | 0.895 | B removes 709 px; the mask falls from 2 525 px to 1 816 against GT 1 808 |
| `299` | 4 | 0 | 0.198 | no usable seed; stray seeds now claim the lesion at a distance; B adds 1 450 px |

Together they are +0.0104 of the +0.0120. **The other 66 images: +0.0017, CI [−0.004,
+0.006].** For `125` and `299`, the gain comes from a seed claiming a component it does not
lie on, which is the uncapped nearest claim, not the removal of a disk.

Largest losses under B: `098` −0.123, `nv (20)` −0.031, `500` −0.027, `211` −0.017.

## Against the predictions

- **Claude:** ΔIoU −0.006 to −0.001, recall about −0.008, gate odds about 10 %. A: ΔIoU +0.0026,
  recall −0.0205. **Wrong on IoU, right that recall fails the gate.**
- **The user:** IoU, Dice and precision rise. A: precision +0.025, a real gain; IoU and Dice
  rise within noise. **Right on precision, not shown on IoU and Dice.**

## Diagnostics

- **Fusion does not change which seeds reach the disk.** 85 seeds on 39 images reach it with
  fusion, 87 without; fusion serves 2. The brief's hypothesis is wrong.
- **What the disk adds.** Against A, the disk adds 61 958 px to the final mask, 36.3 % inside
  GT. Against B, removing it takes away 43 000 px, 29.0 % inside GT. So most disk pixels are
  false positive, as the user saw. But they hold enough lesion that dropping the seed outright
  costs recall on 30 images.
- **B by GT-area quintile (ΔIoU / Δprecision / Δrecall):** Q1 +0.037 / +0.076 / +0.012;
  Q2 +0.002 / +0.011 / −0.017; Q3 +0.017 / +0.030 / +0.001; Q4 +0.003 / +0.011 / −0.006;
  Q5 0.000 / +0.003 / −0.003. Per dataset: DRAC22 +0.0192, TUH +0.0009.

## Decision

- Relax no longer plants the disk. A seed it cannot serve claims the nearest component in
  `seedMasks`. `seedDisks` stays for `fallbackMode = "disk"`, which the frozen arms use.
- New reference: `reference/2026-11-13-relax-no-disk-69.csv`, mean IoU 0.6604.
- **Open risk.** The uncapped nearest claim now decides the output for about 85 seeds. A cap on
  the claim distance is the obvious next question. It needs a pre-registered brief, and it
  will be judged on these same 69 images.
