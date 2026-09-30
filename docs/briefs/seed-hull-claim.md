---
title: Seed Hull Claim Rule, With and Without the Fallback Disk
description: Pre-registered plan to fill only seed triangles whose seeds claimed the same component, with and without the relax fallback's disk. Prediction, gate and decision rule written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: briefs/seed-hull-d.md, results/2026-09-30-seed-hull-d.md, results/2026-09-18-relax-no-disk.md
status: pre-registered 2026-10-01, not yet run
---

# Seed Hull Claim Rule, With and Without the Fallback Disk

Follow-up to [seed-hull-d.md](seed-hull-d.md). Read its record,
[../results/2026-09-30-seed-hull-d.md](../results/2026-09-30-seed-hull-d.md), first.

## Why

The 128 px seed hull was adopted over its own precision guard: −0.0106 on the three smallest
GT-area quintiles against a limit of −0.01. Its visible failure is a triangle whose seeds sit
on two *separate* predicted components. That triangle fills the background between them, as
in `429` and the upper right of `nv (17)`.

A `P`-based guard cannot catch it, because `P` ≈ 0 inside inter-frond gaps too. The
pipeline's own claims can. The numbers below are GT-free, on the cached 5-fold CV `P` of the
68-image domain:

- Of the 869 triangles that pass 128 px, **51 (6 %) have seeds that claimed different
  components**. They contribute **32 % of the hull area** (55 728 of 176 097 px), on 13
  images.
- Of those 51, 28 join a component under 2 000 px to another: 13 distinct **fragments**, 5
  of them exactly disk-sized (709 px). The other 23 join two large components.
- 867 of 874 seeds stand on the component they claim. 7 stand on a `P > 0.5` blob of 65–499
  px, below `MinArea`, and so claim the nearest component, 31–672 px away. Every seed claims
  something.

**The fallback disk.** This is GT-based, on the current configuration. Disk pixels have
pixel precision 0.655 against 0.730 for the whole mask. But they are 0.5 % of it, 17 394 px
from 40 seeds on 9 images, and removing them raises mean per-image precision by only
+0.0014. Dropping the disk *and* its seed's claim was measured before at −0.0006 IoU
([../results/2026-09-18-relax-no-disk.md](../results/2026-09-18-relax-no-disk.md)).

## The rule

After claiming, label the pre-hull mask. Each seed's component id is the label of the
component it claimed (`perSeed`), or 0 if it claimed nothing. A Delaunay triangle whose
sides are all ≤ `hullLink` (128 px, unchanged) is filled only if its three seeds' ids are
**equal and non-zero**.

- "Claimed", not "stands on". The two differ on 2 triangles only, both near-miss seeds on
  strong `P`. Keeping one definition of "which lesion a seed belongs to" matters more.
- Strict: fragment joins are blocked too. An area exemption would add a threshold that needs
  cross-fitting, and a "fallback-only" exemption covers just 5 of the 13 fragments.
- `hullLink` stays at 128. Re-choosing it under the rule would be a second selection on
  these images.

## Arms

| arm | seed hull | seed that relax cannot serve |
|---|---|---|
| current (comparator) | all 128 px triangles | 15 px disk |
| **R** | claim rule | 15 px disk |
| **RN** | claim rule | nothing: no disk, no claim; it stays in the triangulation and blocks its triangles |

RN is the 2026-09-18 variant plus the rule. With the disk gone, no join through a disk can
happen either.

## Prediction (written before running)

- **R, main:** removes about 32 % of hull pixels, on 13 images. Precision up, recall down.
  ΔIoU vs current between −0.001 and +0.003: non-inferior.
- **R, guard:** back within −0.01. About 60 % confident.
- **RN − R**, two bets:
  - *Claude:* precision +0.001 to +0.002, recall about −0.002, ΔIoU within ±0.001 with a
    CI including 0, so R is adopted.
  - *User:* RN beats R on precision and IoU.
- **Open hypothesis, not tested for adoption:** fragments from fallback disks lower
  precision, and more so when a triangle joins them.

## Protocol

68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` via `run_cache`, whole-image scoring,
`tau` 0.5, relax 0.20. Paired per image. Percentile bootstrap on the mean Δ, 10 000
resamples, `rng(20260917)`; exact two-sided sign test, ties excluded.

## Gate, per arm

CIs at 97.5 % (Bonferroni over the two arms).

1. **Non-inferiority:** ΔIoU vs the current configuration, CI lower bound > −0.002.
2. **Precision guard:** mean precision on the three smallest GT-area quintiles, against **no
   seed hull** (0.5402), must be ≥ 0.5302. It is measured against no hull because against
   the current configuration it could only rise: the rule only removes pixels.

## Decision

| R | RN | action |
|---|---|---|
| pass | fail | adopt R |
| fail | pass | adopt RN |
| fail | fail | reject both; the current configuration stays |
| pass | pass | paired RN − R, 95 % CI: excludes 0 → adopt the winner; includes 0 → adopt R |

**Headline if adopted.** A cross-fitted estimate for the adopted arm: `hullLink` chosen per
fold over {64, 96, 128, 192, 256} with that arm's settings, using the relax grid's rule
(ties within 0.001 go to the smallest). For reporting only. `hullLink` stays at 128.

**Reported alongside, as description:** GT-area quintiles and per-dataset breakdowns, the
images that change, the 23 lesion-to-lesion vs 28 fragment-join triangles, `429` and
`nv (17)`, and how many seeds RN excludes (40 expected).

## Build

- Temporary switches `cfg.Post.hullSameClaim` (rule on) and `cfg.Post.relaxDisk` (disk on).
  Adopted: the switch goes and its value becomes the code. Rejected: tag
  `archive/seed-hull-claim` and delete them.
- One `test_cnngvf` check: two separate blobs do not fill; a C-shaped blob with seeds across
  its mouth fills; a seed that claimed nothing blocks its triangle.
- With both switches at their current values, `run_cache` must reproduce
  `2026-09-30-hull-128.csv` (`max |dIoU|` ~1e-15).
