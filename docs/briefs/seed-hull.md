---
title: Seed Hull — Fill Inter-frond Gaps Between Linked Seeds
description: Pre-registered plan for two post-P arms that fill the space between seeds whose heat-map bumps overlap. Prediction, gate and decision rule written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/cnn, MATLAB R2026a
depends-on: SPEC.md, results/2026-09-30-relax-floor-015-paired.md
status: pre-registered 2026-09-30, not yet run
---

# Seed Hull — Fill Inter-frond Gaps Between Linked Seeds

Read [SPEC.md](../SPEC.md) first. Where this brief disagrees with it, stop and ask.

## Motivation

On large lesions with many detections — `1115.png`, `nv (13)`, `nv (17)`, `nv (21)`,
`nv (27)` — the mask misses a lot of the lesion, and the misses are mostly *not* between
fragments. The diagnostic below is GT-based, on the current configuration's cached 5-fold CV
`P` (68-image domain, relax 0.20). It is motivation, not a score: these five images were
picked by eye because they looked bad.

| image | precision | recall | FN in a fragment's own concavity | FN between fragments | FN at the rim |
|---|---|---|---|---|---|
| `1115.png` | 0.914 | 0.849 | 0.82 | 0.00 | 0.18 |
| `nv (13).png` | 0.917 | 0.804 | 0.95 | 0.00 | 0.05 |
| `nv (17).png` | 0.843 | 0.655 | 0.80 | 0.17 | 0.03 |
| `nv (21).png` | 0.829 | 0.772 | 1.00 | 0.00 | 0.00 |
| `nv (27).png` | 0.970 | 0.547 | 0.78 | 0.05 | 0.17 |

The failure is the **inter-frond gap** ([CONTEXT.md](../CONTEXT.md)): `P` follows the
vessels, FilledGT marks the space between them, and the gap opens onto the lesion's edge so
`imfill` cannot reach it. **Fragments** exist (isolated pieces of 709–850 px, about a
fallback disk) but carry little of the missed area. These images have high precision; their
deficit is recall. Across the 68-image reference, recall is lowest in the largest GT-area
quintile (0.675) and precision lowest in the smallest (0.432).

## Hypothesis

Seeds whose Gaussian heat-map bumps overlap sit on one lesion. Filling the space between them
recovers inter-frond gaps: recall rises in the largest-lesion quintile and IoU rises overall,
while small lesions — mostly sparse seeds — are left alone.

## Definitions

- **Link distance** `d = 2 * cfg.Patch.sigmaSeed` = 64 px. Two equal Gaussians of width σ
  sum to a single peak iff their centres are ≤ 2σ apart; beyond that they read as two
  blobs. So 2σ is where "overlapping bumps" begins. Derived, not tuned.
- **Linked seeds**: two seeds within `d`. **Seed group**: a connected set of linked seeds.
  Every inference seed is used; filtering by ground truth would be leakage.

On the 68 images, seed nearest-neighbour distance is median 42 px (q10 28, q25 33, q90 77).
At `d` = 64 there are 269 groups: 132 singletons, 89 of 2–4, 48 of ≥ 5.

## Arms

Both run **after claiming** (`cnngvf.seedMasks`): the added region is ORed into `mask` and
into the per-seed masks of the seeds that produced it, so `mask` stays the union of
`perSeed`. Nothing is fused before claiming, so no unclaimed blob can come back.

| arm | `cfg.Post.seedHull` | adds |
|---|---|---|
| reference | `"none"` | nothing |
| **B, primary** | `"triangles"` | every triangle of the seeds' Delaunay triangulation whose three seeds are pairwise within `d`, holes enclosed by those triangles filled |
| **C, mechanism** | `"close"` | `imclose(·, disk(sigmaSeed))` over the union of components claimed by linked seeds; bridges gaps up to 2 · 32 = 64 px |

B is the hypothesis stated as geometry. C asks whether a width-limited closing gets the same
gain; gaps wider than 64 px stay open under C and not under B.

### Why no seed-count threshold k

An earlier draft gated on "groups of ≥ k seeds". It was dropped. The gate itself matters —
ungated, a radius-32 closing adds +26 % mask area around singleton seeds and +16 % in groups
of ≥ 5, so about 60 % of what it adds would land around sparse seeds, i.e. small lesions,
where precision is already 0.43–0.61. But the link already *is* that gate: a seed with no
neighbour within `d` is excluded, which is k = 2 by definition, and arm B needs three mutually
linked seeds to fill anything, which is k = 3 by construction. Any larger k would be a
threshold chosen from these images, which needs cross-fitting. Seed count is a proxy for
lesion size, but it reads neither `P` nor ground truth.

### Rejected alternatives

- **Convex hull of the mask per group** — fills every concavity, and RNV is deeply concave
  by nature; crescent and vessel-following lesions would be filled to their envelope. A
  whole-image hull added pixels only 9–32 % GT on four of the five images.
- **`alphaShape` with α = 64** — α is a radius, so kept edges reach 2α = 128 px, looser than
  the link. **α = 32** drops fully linked triples (an equilateral triangle of side 64 has
  circumradius 37). The triangle rule says the intended thing with the one number `d`.
- **Scale from `Post.seedRadius` = 15** — that is the size assumed for a lesion `P` says
  nothing about, not a spacing. Link 30 px would leave over 75 % of seeds as singletons, so
  the experiment would be a no-op by construction.

## Prediction (written before running)

- **B**: ΔIoU between **+0.005 and +0.015**, from recall in the largest GT-area quintile
  (up ≥ 0.02); the precision guard holds.
- **C**: smaller ΔIoU than B.
- **Falsified** if B's 97.5 % CI includes 0 or its precision guard trips.

## Protocol

68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` via `run_cache`, whole-image scoring,
`tau` 0.5, relax 0.20, paired per image against `"none"`. Percentile bootstrap on the mean
Δ, 10 000 resamples, `rng(20260917)`; exact two-sided sign test, ties excluded.

## Gate, per arm

1. **ΔIoU 97.5 % CI lower bound > 0** — Bonferroni over the two arms.
2. **Precision guard**: mean precision over the three smallest GT-area quintiles
   (quintiles of `GtPx` over the 68 images) falls by no more than 0.01.

## Decision

| B | C | action |
|---|---|---|
| pass | fail | adopt B |
| fail | pass | adopt C |
| fail | fail | reject both |
| pass | pass | paired B − C, 95 % CI: excludes 0 → adopt the winner; includes 0 → adopt neither, write a follow-up brief |

Report alongside, as description rather than evidence: quintile and per-dataset breakdowns,
folds, and the five motivating images.

## Build

- `cfg.Post.seedHull`, default `"none"`, and `cnngvf.seedHull` called from
  `cnngvf.maskFromProb` after `seedMasks`. The switch is temporary: adopted, it becomes the
  default and the switch goes; rejected, tag `archive/seed-hull` and delete it.
- One `test_cnngvf` check: three mutually linked seeds fill their triangle, a seed beyond `d`
  adds nothing, and `"none"` is byte-identical to the reference.
- `run_cache` with `"none"` must reproduce the reference (`max |dIoU|` ~1e-15).
