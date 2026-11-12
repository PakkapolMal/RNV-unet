---
title: Fusing the Seedless P at Weak Seeds
description: Pre-registered post-P arm. At the seeds the seeded network left empty (weak seeds), the seedless network's P either gives the outline first or is fused with the seeded P, with the relax fallback kept as the backstop. Variants, thresholds, cross-fit, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: results/2026-11-12-seedless-fallback.md, results/2026-11-11-tau-floor-refit.md, results/2026-09-16-reduce.md
status: executed 2026-11-12
---

# Fusing the Seedless P at Weak Seeds

> **Executed brief.** Kept as written. Outcome: gate 1 failed. Cross-fitted ΔIoU is +0.0010,
> CI [−0.0007, +0.0028]; four folds chose "max then relax" at 0.9. Relax stays:
> [../results/2026-11-12-fusion-weak-seeds.md](../results/2026-11-12-fusion-weak-seeds.md).

> **Amended 2026-11-12, before any run.** The first version triggered fusion per image on the
> agreement A = Dice(seeded mask, seedless mask) and fused the whole image (operators `max`,
> `mean`, `replace` × `t` ∈ {0.3, 0.5, 0.7, always}).
>
> **Why it was replaced:**
> - A compares two predictions, so it says nothing about which one is right.
> - It also falls when the seedless network marks blobs far from any seed, which claiming
>   removes anyway.
>
> **What replaces it:** the trigger is now the **weak seed**, a direct, GT-free sign that the
> seeded network did not see the lesion at that seed. Fusion acts only in the window of a weak
> seed. An image-level trigger on seeded uncertainty (U) is deferred to a later brief.

## Why

**Two out-of-fold probability maps exist for every image** ([2026-11-12](../results/2026-11-12-seedless-fallback.md)):
- the seeded `P` (`cv/probcache`);
- the seedless `P` (`cv/probcache_seedless`), almost binary. For the median image, only 17.5 %
  of its non-zero pixels lie between 0.1 and 0.9.

On the 66 images with points, 6.4 % of GT pixels are above 0.5 only in the seedless `P`.

**A weak seed is a seed the threshold left empty:** the seeded `P` at the seed pixel is at most
`tau` (0.5). These are exactly the seeds the relax fallback serves today: it thresholds the
seeded `P` again at 0.20 within ±128 px and takes the component under the seed, or a 15 px disk
if that finds nothing of `MinArea`. In the current caches, **156 of 871 seeds (17.9 %) are
weak, on 47 of the 66 images with points.**

Their outline comes from a map that was unsure at the seed. The seedless network saw the same
place without the seed, and its map is decisive. It may outline the lesion where relax gives
only part of it.

**Seeds still decide what survives:** the variants change `P` only near weak seeds, and
claiming and the hull run unchanged.

**No training:** both caches exist and are out-of-fold with the same folds. Every cell scores
off the caches with no training noise.

**Scope:**
- all 69 images (DRAC22 42, TUH 27) are scored;
- the variants can change only the 47 images with a weak seed;
- the 3 images with no points keep their stage 9 mask in every cell.

## Arm

Weak seeds are found on the seeded `P` alone, as the pipeline finds them today. Each variant
turns (seeded `P`, seedless `P`) into one map `P_f`, and `cnngvf.maskFromProb` then runs
unchanged on `P_f`: `tau` 0.5, relax 0.20 within 128 px, `MinArea` 500, boundary claiming,
hull 128.

| variant | `P_f` | relax 0.20 |
|---|---|---|
| **relax only** (default: the current pipeline) | the seeded `P` | the fallback for every weak seed |
| **seedless first** | the seeded `P`, plus `P_f` = 1 on the seedless component under each weak seed: seedless `P` > τ_s within ±128 px, 8-connected through the seed pixel, kept if at least `MinArea` | the backstop, for weak seeds with no such component |
| **max, then relax** | within ±128 px of each weak seed, max(seeded `P`, seedless `P` where it exceeds τ_s, else 0); the seeded `P` elsewhere | runs on `P_f`, for weak seeds still empty |

**Seedless entry threshold** τ_s ∈ {0.5, 0.9}. Below τ_s the seedless `P` is not used. 0.9
admits only confident seedless pixels.

**Grid:** the default plus {seedless first, max then relax} × {0.5, 0.9}: 5 cells.

The window, rounding, connectivity and `MinArea` test are relax's own, so "seedless first"
differs from relax only in the map it reads and its threshold.

## Selection (cross-fitting)

- For each fold, choose the cell with the best mean IoU over the other four folds' images, and
  score it on the held-out fold.
- Ties within 0.001 go to the default, then to the more conservative cell (τ_s 0.9 before 0.5,
  then "seedless first" before "max, then relax").
- The fixed-cell surface is description only.

## Protocol

- 69 images, the two adopted 5-fold CV caches, whole-image scoring by `run_cache`.
- **Sanity check:** the default cell reproduces
  `reference/2026-11-12-seedless-fallback-69.csv` to ~1e-15.
- Paired per image against the default. Percentile bootstrap on the mean Δ: 10 000 resamples,
  `rng(20260917)`, 95 % CI.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.
- **Code:**
  - `cnngvf.fuseWeakSeeds(Pk, Ps, seeds, cfg)` builds `P_f`, with experiment fields
    `cfg.Post.weakSeed` (`"relax"`, `"seedlessFirst"`, `"maxRelax"`) and
    `cfg.Post.weakSeedTau` (τ_s);
  - `run_cache` reads the seedless cache for images with points only when `weakSeed` is not
    `"relax"`;
  - the runner is `run_fusion_grid` (experiment code).

## Prediction (written before running)

- **Both variants raise recall on the lesions behind weak seeds,** which are the under-covered
  ones.
- **"Seedless first" at τ_s 0.9 is the best cell.** A near-binary seedless component gives a
  fuller outline than relax at 0.20, and 0.9 limits spill. "Max, then relax" is close behind.
- **Cross-fitted ΔIoU +0.003 to +0.015,** with the CI likely above 0 at the low end. At least 4
  of 5 folds choose "seedless first".
- **Q1–Q3 precision falls by 0.000–0.008,** inside the guard. The risk is a seedless component
  that runs into a neighbouring non-lesion area.
- **Relax stays as the backstop.** I expect a minority of weak seeds (20–40 %) to find no
  seedless component and still take the relax outline.
- I give the gate about 40 %.

## Gate

Adopt a cell only if all three hold:

1. The cross-fitted ΔIoU against the default has a 95 % CI lower bound above 0.
2. At least 4 of 5 folds choose the same cell. That cell ships.
3. **Precision guard:** on the cross-fitted masks, Q1–Q3 precision falls by no more than 0.01.
   Q4–Q5 recall is reported alongside.

**Otherwise** relax stays as it is. Tag the code `archive/fusion`, delete it, and record the
result.

**On adoption:**
- `cnngvf.fuseWeakSeeds` becomes a stage before `maskFromProb`, with the chosen variant and
  τ_s as defaults; the switch for the losing variant goes;
- `run_cache` reads both caches;
- a new reference CSV, `cfg.CV.reference`, the README table, and SPEC (stage 5, the pipeline
  diagram, the fields).

## Report alongside, as description

- the 5-cell surface (IoU, precision, recall), per dataset;
- each fold's choice;
- for each cell: how many weak seeds took a seedless component, and how many fell to relax and
  to the disk;
- precision and recall by quintile, cross-fitted against the default;
- per-image movement: the largest gains and losses.

**Valid until either `P` changes.**
