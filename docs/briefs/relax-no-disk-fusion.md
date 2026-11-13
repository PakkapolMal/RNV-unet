---
title: Relax Without the Disk, Under Fusion
description: Pre-registered post-P arm. A seed relax cannot serve gets nothing instead of the 15 px disk, under the current pipeline (fusion at stage 4a, stage 9). One cell, nothing tuned. Both predictions and the gate written before the run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-09-18-relax-no-disk.md, results/2026-10-01-seed-hull-claim.md, results/2026-11-12-fusion-confirm.md
status: executed 2026-11-13
---

# Relax Without the Disk, Under Fusion

> **Executed brief.** Kept as written. The arm as specified (no disk, no claim) failed both
> gates: ΔIoU +0.0026, CI [−0.0059, +0.0126]; Δrecall −0.0205.
>
> **Amendment, after scoring.** The first implementation skipped only the disk, so the seed
> still claimed the nearest component. That variant scored ΔIoU +0.0120, CI [+0.0009, +0.0258],
> Δrecall −0.0024. The user adopted it. It was not pre-registered, and +0.0104 of its gain is
> three images: [../results/2026-11-13-relax-no-disk-fusion.md](../results/2026-11-13-relax-no-disk-fusion.md).

## Why

**This is the third test of the idea.** Both earlier ones rejected it:
- [2026-09-18](../results/2026-09-18-relax-no-disk.md): −0.0006 IoU; it excluded 40 seeds on 9
  images.
- [2026-10-01](../results/2026-10-01-seed-hull-claim.md), arm RN: −0.0045 IoU, CI
  [−0.0098, −0.0002]; recall fell on 16 images and rose on none.

**Both ran before stage 4a and the PyTorch `P`.** Fusion brings the seedless `P` in before
relax, so it may serve some of the seeds that used to reach the disk. The hypothesis is that
fusion has changed which seeds reach the disk, and that the disks still reached are mostly
false positive.

**This arm was proposed after looking at images.** The user saw disks landing on background.
It is therefore judged on all 69 images with one fixed cell, not on the images inspected.

**Against the hypothesis:** in 2026-10-01, 82 % of the pixels fallback-disk fragments joined
through the hull were lesion. That was before fusion, and about joins, not the disks alone.

## Arm

In `relaxFallback`, when the component under the seed is empty or under `MinArea`, the seed
gets nothing: no disk, no claim. Everything else is the current default: `tau` 0.5, relax
floor 0.20 in a 128 px window, fusion at `fuseTauS` 0.9, stage 9 at `noPointTau` 0.40, seed
hull at 128 px.

**Implementation:** a temporary `cfg.Post.relaxDisk` flag, default `true`, on
`exp/relax-no-disk-fusion`. `false` skips the `seedDisks` call in `relaxFallback`. The
`"disk"` fallback mode is untouched; the frozen arms use it. The scoring script is a scratch
`.m` file, not committed.

## Protocol

- 69 images (DRAC22 42, TUH 27), 5-fold CV caches, whole-image scoring by `run_cache`.
- The reference run must reproduce `reference/2026-11-12-fusion-69.csv` before the arm is
  scored.
- One cell, paired per image against that reference.
- Percentile bootstrap on the mean Δ: 10 000 resamples, `rng(20260917)`, 95 % CI. Exact
  two-sided sign test on up / down.
- Dice per image as 2·IoU / (1 + IoU), reported as a paired Δ; not gated.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.
- Nothing is selected, so the score is out-of-sample as it stands.

## Predictions (written before running)

**Claude:**
- Fusion serves at most a few of the seeds that reached the disk; about 40 seeds on about 9
  images still do.
- ΔIoU −0.006 to −0.001; ΔDice negative; Δprecision about +0.005; Δrecall about −0.008.
- The gate passes with about 10 % odds.

**The user:**
- IoU, Dice and precision all rise. Most fallback disks are false positive.

## Gate

Adopt only if both hold:

1. **Non-inferiority:** the paired ΔIoU 95 % CI lower bound is above −0.002.
2. **Recall guard:** Δrecall is at least −0.005.

**Otherwise** the disk stays as relax's floor. Write the result record, tag the branch
`archive/relax-no-disk-fusion`, delete the flag, and merge the records only.

**On adoption:**
- delete the disk branch from `relaxFallback`; `seedDisks` stays for `"disk"` mode;
- a new reference CSV and `cfg.CV.reference`, the README table, SPEC, CONTEXT, and the
  `defaults.m` comment on relax.

## Report alongside, as description

- **Seeds reaching the disk:** count and images, under the reference configuration, and how
  many of the 40 from 2026-09-18 fusion now serves. This is the direct test of the hypothesis.
- **What the disk adds:** in the reference configuration, the pixels each disk contributes to
  the final mask, and the fraction inside GT. This is the direct test of the user's reason.
  Computed together with the scores, not before.
- per dataset and by quintile; the largest per-image gains and losses.

**Valid until the seeded or the seedless `P` changes.**
