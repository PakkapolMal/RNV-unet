---
title: Excluding Seeds Relax Cannot Serve — Not Adopted
description: Dropping a seed instead of planting the disk when relax finds nothing claimable.
kind: measurement-record
status: rejected
dataset: DRAC22 + TUH (68)
updated: 2026-09-18
---

# Excluding Seeds Relax Cannot Serve — Not Adopted

**Question.** When the relax fallback finds nothing claimable around a seed (P at the seed
≤ the floor, or a region under `MinArea`), should that seed be excluded — no disk and no
claim — instead of getting the 15 px disk?

**Setup.** Flag `Post.relaxDiskFallback` (commit `67559ce`), threshold mode, relax floor
0.20, scored off the cross-validated P cache: 68 images, out-of-sample.

| | mean IoU |
|---|---|
| disk kept (reference) | **0.5097** |
| seed excluded | 0.5091 |

−0.0006. It excludes 40 seeds on 9 images: 3 images better, 6 worse. The in-sample batch
(`result/DRACTUH_threshold_relax_nodisk/`) agrees, at 0.5574 against 0.5575.

**Decision.** Not adopted; the disk stays as relax's floor. The flag was deleted on
2026-09-25. Code at tag `archive/pre-cleanup`.
