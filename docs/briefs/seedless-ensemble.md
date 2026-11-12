---
title: Averaging the Two Seedless Networks
description: Pre-registered post-P arm. The seedless P becomes the mean of the two seedless runs (rng 0 and rng 1), in both places the pipeline reads it (stage 4a fusion, stage 9 no-point mask). One cell, thresholds unchanged, nothing tuned. Prediction and gate written before the run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-11-12-fusion-confirm.md, results/2026-11-12-seedless-fallback.md
status: pre-registered 2026-11-12, not yet run
---

# Averaging the Two Seedless Networks

## Why

**Two runs of the same seedless recipe disagree on single images**
([2026-11-12](../results/2026-11-12-fusion-confirm.md)):
- on the three point-less images, they score 0.550 and 0.680 mean IoU;
- `445` scores 0 with the adopted run and 0.363 with the second;
- stage 4a's gain is +0.0120 with one and +0.0104 with the other.

**Averaging the two is the standard remedy.** Two members already exist, out-of-fold with the
same folds, so the arm needs no training.

**This arm was proposed after looking at `445`.** It is therefore judged on all 69 images with
one fixed cell, not on whether `445` improves.

## Arm

Seedless `P` = (rng 0 seedless `P` + rng 1 seedless `P`) / 2, per image, for both uses:
- **stage 4a:** max(seeded `P`, seedless `P` > `fuseTauS`), with `fuseTauS` unchanged at 0.9;
- **stage 9:** `noPointMask` at `noPointTau`, unchanged at 0.40.

**Both maps are near-binary, so the mean is mostly 0, 0.5 or 1.** Where only one run is
confident, the mean is about 0.5. That has opposite effects in the two stages:
- **stage 4a at 0.9 admits only pixels both runs agree on.** It becomes more conservative.
- **stage 9 at 0.40 admits pixels either run sees.** It becomes more permissive.

That asymmetry is part of what is tested. The thresholds were chosen for single-run maps and
are **not** re-tuned here.

**Implementation:** a cache of the averaged maps, in the `probCache` format, with each image's
fold stamp copied after checking that both runs agree on it. Then `run_cache` with
`Post.seedlessCache` pointed at it. No pipeline code changes.

## Protocol

- 69 images (DRAC22 42, TUH 27), 5-fold CV caches, whole-image scoring by `run_cache`.
- One cell, compared paired per image with the adopted reference
  (`reference/2026-11-12-fusion-69.csv`).
- Percentile bootstrap on the mean Δ: 10 000 resamples, `rng(20260917)`, 95 % CI.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.
- Nothing is selected, so the score is out-of-sample as it stands.

## Prediction (written before running)

- **Stage 4a loses a little.** Agreement-only fusion adds less: recall falls, precision rises
  slightly. The images with points move by −0.004 to +0.002 in total.
- **Stage 9 gains on `445`,** to about 0.2–0.4, at some precision cost from run 1's 701 px
  elsewhere. `082` and `217` stay within ±0.03.
- **Headline ΔIoU −0.003 to +0.006.** The CI includes 0.
- I give the gate about 30 %.

## Gate

Adopt only if both hold:

1. The paired ΔIoU against the reference has a 95 % CI lower bound above 0.
2. **Precision guard:** Q1–Q3 precision falls by no more than 0.01.

**Otherwise** the adopted seedless cache stays the rng 0 run alone. Record the result; there
is no code to archive.

**On adoption:**
- the averaged cache becomes `cv/probcache_seedless`, with a note on how it was built;
- a new reference CSV, `cfg.CV.reference`, the README table, and SPEC (stages 4a and 9: two
  seedless runs, averaged);
- `run_batch` and `run_steps` are unaffected; they apply neither stage.

## Report alongside, as description

- stage 4a and stage 9 separately: the images with points, and the three point-less images;
- per dataset and by quintile;
- `445`, `082` and `217`;
- the largest per-image gains and losses.

**Valid until a seedless `P` changes.**
