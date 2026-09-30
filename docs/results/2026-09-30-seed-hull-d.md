---
title: Seed hull, cross-fitted link distance
kind: measurement-record
dataset: DRAC22 + TUH (68)
updated: 2026-09-30
---

# Seed hull, cross-fitted link distance

Brief, prediction and gate: [../briefs/seed-hull-d.md](../briefs/seed-hull-d.md), committed
before any run. It follows [2026-09-30-seed-hull](2026-09-30-seed-hull.md), where the same
triangle fill at 64 px was inert.

> **Adopted against its own gate.** Gates 1 and 3 passed. The pre-registered precision guard
> measured −0.0106 against a limit of −0.01 and failed. `hullLink = 128` was adopted anyway,
> by explicit decision after the result was seen. Every number that includes the seed hull
> carries this caveat.

**Question.** Does the seed triangle fill help once the pairwise bound `d` is large enough
to act, with `d` chosen by cross-fitting?

**Prediction (from the brief).** Most folds choose `d` in 128–192. Cross-fitted ΔIoU between
+0.003 and +0.010, with the 95 % CI touching 0. The precision guard holds.

**Protocol.** 68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` through `run_cache`,
whole-image scoring, `tau` 0.5, relax 0.20. Grid `d` ∈ {64, 96, 128, 192, 256}. For each fold,
choose the `d` with the best mean IoU on the other four folds (ties within 0.001 go to the
smallest `d`) and score it on the held-out fold. Compared paired per image against no seed
hull, which reproduces `2026-09-17-relax-0.20.csv` to 5e-16. Percentile bootstrap on the mean
Δ, 10 000 resamples, `rng(20260917)`, 95 % CI; exact two-sided sign test, ties excluded.
Script: a scratch `.m` file, not committed. The code restored from `archive/seed-hull` also
marks each kept triangle's rounded vertex pixels, so a sub-pixel seed still claims its fill.
That changes the 64 px arm by a few pixels relative to the previous record.

## Result

### Cross-fitted

| | IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| no seed hull | 0.5097 | 0.5299 | 0.6489 | 0.6311 | 0.7579 |
| seed hull, `d` cross-fitted per fold | **0.5184** | 0.5375 | 0.6554 | 0.6228 | 0.7771 |

| metric | Δ | 95 % CI | up / down / tie | sign *p* |
|---|---|---|---|---|
| IoU | +0.0087 | [+0.0017, +0.0155] | 23 / 8 / 37 | 0.011 |
| precision | −0.0083 | [−0.0167, −0.0019] | 13 / 18 / 37 | 0.47 |
| recall | +0.0192 | [+0.0121, +0.0271] | 30 / 0 / 38 | <0.001 |

Folds choose 128, 128, **192**, 128, 128.

| gate | result | |
|---|---|---|
| 1. ΔIoU 95 % CI lower bound > 0 | +0.0017 | pass |
| 2. precision guard, three smallest GT-area quintiles, limit −0.01 | 0.5402 → 0.5295, **−0.0106** | **fail** |
| 3. ≥ 4 of 5 folds choose the same `d` | 4 of 5 on 128 | pass |

**Against the prediction:** the fold choices and ΔIoU fall where predicted. The CI clears 0,
which is more certain than predicted. The guard, predicted to hold, failed by 0.0006.

### Where it acts (cross-fitted)

| GT-area quintile | n | ΔIoU | Δprecision | Δrecall |
|---|---|---|---|---|
| Q1 (601–4 176 px) | 14 | +0.0000 | +0.0000 | +0.0000 |
| Q2 | 13 | −0.0040 | −0.0149 | +0.0070 |
| Q3 | 14 | +0.0000 | −0.0173 | +0.0215 |
| Q4 | 13 | +0.0070 | −0.0030 | +0.0109 |
| Q5 (50 314–432 533 px) | 14 | **+0.0393** | −0.0064 | +0.0551 |

- **Devices:** TUH +0.0169, 95 % CI [+0.0010, +0.0310], 14 up / 3 down. DRAC22 +0.0032,
  95 % CI [−0.0017, +0.0086], 9 up / 5 down.
- **Folds:** +0.0148, +0.0138, −0.0045, +0.0068, +0.0126.
- **Largest gains:** `nv (27)` +0.082, `nv (3)` +0.077, `nv (21)` +0.071, `nv (17)` +0.067,
  `nv (13)` +0.060.
- **Largest losses:** `nv (14)` −0.124, `429` −0.046, `125` −0.027, `458` −0.014, `194` −0.011.
  `nv (14)` is in fold 3, which chose 192. At the adopted 128 it loses 0.012.
- **Failure mode, seen in the error maps:** a triangle whose seeds sit on two *separate*
  lesions spans the background between them. `429` loses this way, and so does the upper
  right of `nv (17)`. The fill cannot tell a gap inside a lesion from a gap between lesions.
- **The five motivating images**, description only: `1115` 0.786 → 0.806, `nv (13)`
  0.749 → 0.809, `nv (17)` 0.584 → 0.651, `nv (21)` 0.666 → 0.737, `nv (27)` 0.538 → 0.620.

### Fixed-`d` surface

Tuned on the images it scores, so this is description only.

| `d` | IoU | precision | recall |
|---|---|---|---|
| none | 0.5097 | 0.6311 | 0.7579 |
| 64 | 0.5099 | 0.6306 | 0.7586 |
| 96 | 0.5154 | 0.6308 | 0.7660 |
| **128** | **0.5207** | 0.6268 | 0.7762 |
| 192 | 0.5199 | 0.6118 | 0.7913 |
| 256 | 0.5111 | 0.5980 | 0.7948 |

## Decision

**Adopted, `cfg.Post.hullLink = 128`, over the failed precision guard.** The IoU gain clears
its cross-fitted CI, the folds agree on 128, and the guard missed by 0.0006. The guard's
limit was set before the run and this adoption does not move it: the record says the gate
failed. The cost it measured is real. Precision on the smaller three quintiles falls by
about 0.011 for a recall gain there, while Q1 is untouched. The benefit is concentrated in
the largest lesions and in TUH.

New reference: [reference/2026-09-30-hull-128.csv](reference/2026-09-30-hull-128.csv), the
fixed `d` = 128 per image, which `run_cache` reproduces. Its mean, 0.5207, sits on a `d`
chosen by these images. The cross-fitted 0.5184 is the honest estimate.

**What would confirm it.** Images the pipeline has never scored. Any further tuning of `d`
or of a gate on these 68 images would be selection, not confirmation. The bridging failure
suggests a follow-up: a triangle should not span background that `P` is confident about.
That rule would be new and would need its own brief.
