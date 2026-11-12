# Fusing the seedless P at weak seeds gains +0.001 IoU: relax stays (2026-11-12)

Brief: [../briefs/fusion.md](../briefs/fusion.md), as amended before the run.

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- the two adopted 5-fold CV caches: seeded `cv/probcache` and seedless
  `cv/probcache_seedless` ([2026-11-12](2026-11-12-seedless-fallback.md));
- whole-image scoring by `run_cache`, one run per cell;
- 5 cells: the default "relax only", and {seedless first, max then relax} × τ_s {0.9, 0.5};
- each fold's cell chosen on the other four folds, ties to the default, then the conservative
  order;
- paired against the default on the same caches, bootstrap 10 000 resamples,
  `rng(20260917)`.

**Sanity check: passes.** The default cell reproduces
[reference/2026-11-12-seedless-fallback-69.csv](reference/2026-11-12-seedless-fallback-69.csv),
max |dIoU| 5.6e-16.

Runner and stage: `run_fusion_grid`, `cnngvf.fuseWeakSeeds`, at tag `archive/fusion`.

## Result: the gate fails, and relax stays

| | IoU | F1 | precision | recall |
|---|---|---|---|---|
| default (relax only) | 0.6364 | 0.7535 | 0.7667 | 0.7625 |
| cross-fitted | 0.6374 | 0.7543 | 0.7663 | 0.7646 |

**Cross-fitted ΔIoU +0.0010, 95 % CI [−0.0007, +0.0028];** 21 images up, 10 down. 31 images
moved, and 8 by more than 0.01.

| fold | chosen cell | IoU, other 4 folds | IoU, held-out fold | default, held-out fold |
|---|---|---|---|---|
| 1 | max then relax, 0.9 | 0.6323 | 0.6688 | 0.6646 |
| 2 | max then relax, 0.9 | 0.6481 | 0.6067 | 0.6061 |
| 3 | max then relax, 0.9 | 0.6405 | 0.6367 | 0.6357 |
| 4 | seedless first, 0.9 | 0.6534 | 0.5709 | 0.5726 |
| 5 | max then relax, 0.9 | 0.6236 | 0.7090 | 0.7081 |

**Gate:**
1. **CI lower bound above 0: fails** (−0.0007).
2. **4 of 5 folds on one cell: holds.** Four chose (max then relax, 0.9).
3. **Precision guard: holds.** Q1–Q3 precision +0.0003; Q4–Q5 recall +0.0023.

## What the weak seeds got

The 66 images with points carry 156 weak seeds. Per cell, where each one's outline came from:

| cell | IoU | DRAC22 | TUH | seedless evidence | relax | disk |
|---|---|---|---|---|---|---|
| relax only | 0.6364 | 0.5797 | 0.7245 | 0 | 69 | 87 |
| seedless first, 0.9 | 0.6367 | 0.5799 | 0.7250 | 12 | 59 | 85 |
| max then relax, 0.9 | 0.6397 | 0.5849 | 0.7251 | 12 | 59 | 85 |
| seedless first, 0.5 | 0.6365 | 0.5795 | 0.7252 | 15 | 56 | 85 |
| max then relax, 0.5 | 0.6399 | 0.5854 | 0.7248 | 15 | 56 | 85 |

(The surface is description; the cross-fitted row above is the result.)

- **The seedless network sees nothing under most weak seeds.** At τ_s 0.5, only 15 of 156
  weak seeds (10 %) sit on a seedless component of at least `MinArea`. Where the seeded
  network was unsure, the seedless one mostly was too.
- **Most weak seeds end on the disk, not on relax.** Under the current pipeline, 87 of 156
  (56 %) find no relax component of `MinArea` at 0.20, and take the 15 px disk.
- **"Max then relax" gains more than "seedless first" for the same 12–15 seeds.** It takes the
  maximum over the whole ±128 px window, so it also extends components near a weak seed that
  are not under it. Its in-sample edge, +0.0035 at most, comes from those extensions, not from
  the weak seeds' own outlines. The cross-fit keeps under a third of it.

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| both variants raise recall on lesions behind weak seeds | held weakly: recall +0.002 cross-fitted |
| "seedless first" at τ_s 0.9 is the best cell, 4 of 5 folds on it | **wrong:** "max then relax" at 0.9, on 4 of 5 folds; "seedless first" gains almost nothing (+0.0003 in-sample) |
| cross-fitted ΔIoU +0.003 to +0.015, CI above 0 at the low end | **wrong, below:** +0.0010, CI including 0 |
| Q1–Q3 precision falls by 0.000–0.008 | held: +0.0003 |
| 20–40 % of weak seeds still take relax | **wrong:** 90 % find no seedless component; 85 take the disk, 56 relax |
| about a 40 % chance of passing | failed |

## Reading

- **Relax stays as it is.** Bringing the seedless `P` in at weak seeds is worth +0.0010 IoU
  (69 images, 5-fold CV, CI [−0.0007, +0.0028]).
- **Weak seeds are not where the two networks differ.** Under 90 % of them the seedless network
  also sees nothing. The 6.4 % of GT pixels that only the seedless `P` covers
  ([2026-11-12](2026-11-12-seedless-fallback.md)) must lie mostly elsewhere. "Max then relax"
  gained only where its window reached past the weak seed. That is a pointer, not a measurement,
  for the deferred trigger on seeded uncertainty (U), which acts on lesions seeds already reach.
- **The disk serves more weak seeds than relax does** (87 against 69 of 156). Measured on the
  replica network, the disk radius was near-inert ([2026-10-02](2026-10-02-minarea-disk.md)).
  The disk seeds have not been looked at on this `P`.
- **Valid until either `P` changes.**
