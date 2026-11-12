---
title: Fusing the Seeded and Seedless P
description: Pre-registered post-P arm. On the 66 images with points, the seedless network's P is fused into the seeded P before the unchanged pipeline (threshold, relax, claiming, hull). A GT-free agreement score between the two networks' masks decides which images get the fusion. Operators, thresholds, cross-fit, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-11-12-seedless-fallback.md, results/2026-11-11-paper-recipe.md, results/2026-11-11-tau-floor-refit.md
status: pre-registered 2026-11-12, not yet run
---

# Fusing the Seeded and Seedless P

## Why

Two out-of-fold probability maps now exist for every image, from networks that fail
differently ([2026-11-12](../results/2026-11-12-seedless-fallback.md)):
- the **seeded** `P` (`cv/probcache`): seed-conditioned and claimed by seeds. It is strong on
  large TUH lesions, which seeded patches tile;
- the **seedless** `P` (`cv/probcache_seedless`): the whole frame with no seed input. Alone it
  scores lower (0.520 DRAC22, 0.473 TUH), but on the 66 images with points **6.4 % of GT
  pixels are above 0.5 only in the seedless `P`**, at a cost of 0.21 % of background pixels.

That 6.4 % is lesion the seeded pipeline misses. The deep misses, where seeded `P` < 0.2 inside
the GT, are the one failure the last two adoptions did not reach
([2026-11-10](../results/2026-11-10-train-budget-pmaps.md)).

**Seeds still decide what survives.** The fused `P` goes through the unchanged pipeline, so a
seedless blob that no seed claims is dropped by claiming, as any other is. This is what keeps
the fusion from importing the seedless network's whole-frame false positives.

**No training.** Both caches exist and are out-of-fold with the same folds, so every cell scores
off the caches in seconds and carries no training noise.

## The trigger: why not "seeded Dice < 0.5"

The obvious rule is to fall back to the seedless network where the seeded result is poor: for
example, where the seeded Dice is below 0.5. **That Dice is computed against GT, so it is
unknown at inference.** A rule using it would be leakage: it would pick, per image, whichever
map scores better, and the score it reports is an oracle.

**The GT-free stand-in is agreement between the two networks.** Let `M_k` be the seeded
pipeline's mask (the current configuration) and `M_s` the seedless mask at the adopted cell
(`noPointMask`, `tau` 0.40). Then

    A = Dice(M_k, M_s) = 2 |M_k ∩ M_s| / (|M_k| + |M_s|)

is known at inference, uses no GT, and is high where the two networks see the same lesion.
Where they disagree, at least one is wrong, and fusing lets claiming arbitrate. **"Agreement
< 0.5" is the legal form of the suggested rule.**

0.5 is one value in the grid below, not a fixed choice: the threshold is cross-fitted.
Whether low agreement actually marks poor seeded images is reported as a description, not
assumed.

## Arm

For each image **with points** whose agreement `A` is below `t`, replace the seeded `P` with a
fused `P_f`, then run `cnngvf.maskFromProb` unchanged (`tau` 0.5, relax 0.20 in 128 px,
`MinArea` 500, boundary claiming, hull 128). Images with `A ≥ t`, and the 3 images with no
points (stage 9), are unchanged.

**Operators:**

| operator | `P_f` | what it does |
|---|---|---|
| `max` | max(`P_k`, `P_s`) | adds what either network sees; the recall-heavy choice |
| `mean` | (`P_k` + `P_s`) / 2 | the networks vote; outside the seeded patches (`P_k` = 0) it cannot cross 0.5 unless `P_s` = 1, so it mostly reshapes lesions the seeds already reach |
| `replace` | `P_s` | the suggested fallback: the seedless map, still claimed by the seeds |

**Grid:** operator × `t` ∈ {0.3, 0.5, 0.7, **always**} (always means every image with
points), plus the default **off**: 13 cells.

## Selection (cross-fitting)

The rule of the earlier cross-fits:
- for each fold, choose the cell with the best mean IoU over the other four folds' images;
- score it on the held-out fold;
- ties within 0.001 go to **off**, then to the more conservative cell: lower `t` first, then
  `mean`, `max`, `replace`.

The fixed-cell surface is description only.

## Protocol

- 69 images (DRAC22 42, TUH 27), the two adopted 5-fold CV caches, whole-image scoring.
- Paired per image against **off**, which must reproduce
  `reference/2026-11-12-seedless-fallback-69.csv` to ~1e-15 (sanity check).
- Percentile bootstrap on the mean Δ: 10 000 resamples, `rng(20260917)`, 95 % CI.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.
- **Runner:** `run_fusion_grid` (experiment code). It reads both caches through
  `cnngvf.probCache` (the leakage guard on both), computes `A` per image, and scores every cell.

## Prediction (written before running)

- **`max` on every image** raises Q4–Q5 recall and costs Q1–Q3 precision. Net ΔIoU is
  −0.005 to +0.010.
- **Gating by agreement helps.** The best cells are `max` or `mean` at `t` 0.5–0.7.
- **`replace` loses** at every `t`. The seedless network alone is 0.25 IoU below the seeded
  pipeline on TUH.
- **Cross-fitted ΔIoU +0.000 to +0.012,** with the CI likely touching 0. Folds split between
  two neighbouring cells.
- **Agreement and seeded IoU correlate** (Spearman 0.3–0.6 over the 66 images), so the
  stand-in partly works.
- I give the gate about 30 %.

## Gate

Adopt a cell only if all three hold:

1. The cross-fitted ΔIoU against off has a 95 % CI lower bound above 0.
2. At least 4 of 5 folds choose the same cell. That cell ships.
3. **Precision guard:** on the cross-fitted masks, Q1–Q3 precision falls by no more than 0.01.
   Fusion adds pixels, so this is the binding guard. Q4–Q5 recall is reported alongside.

**Otherwise** nothing changes. Tag the runner `archive/fusion`, delete it, and record the
result.

**On adoption:**
- `cnngvf.fuseProb` becomes a stage before `maskFromProb`, with `cfg.Post.fusion` (operator)
  and `cfg.Post.fusionAgreement` (`t`);
- `run_cache` reads both caches;
- a new reference CSV, `cfg.CV.reference`, the README table, and SPEC (stage, fields, the
  pipeline diagram).

## Report alongside, as description

- the 13-cell surface (IoU, precision, recall), per dataset;
- each fold's choice;
- the distribution of `A` over the 66 images, and its rank correlation with the seeded IoU
  (that correlation uses GT, so it is a diagnostic of the stand-in, never an input);
- **an oracle bound, labelled as one:** per image, the better of off and the best operator,
  chosen with GT. It caps what any agreement rule could gain;
- precision and recall by quintile, cross-fitted against off.

**Valid until either `P` changes.**
