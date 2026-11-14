---
title: Hard-Negative Seeds and Multi-Scale Seed Conditioning
description: In progress. The empty-mask handoff to stage 9, scored alone on the 2026-11-13 cache, moves no image. The arm's bundle is exported; the 5-fold CV has not run.
kind: measurement-record
status: in progress — cache-only check done, CV pending
dataset: DRAC22 + TUH (69)
updated: 2026-11-14
---

# Hard-Negative Seeds and Multi-Scale Seed Conditioning

Brief, predictions and gate: [../briefs/2026-11-14-hard-neg-seeds.md](../briefs/2026-11-14-hard-neg-seeds.md),
committed before any code (`b43de6a`).

## Cache-only check: the empty mask → stage 9, alone

This was run before any training, on the 2026-11-13 caches (`cv/probcache`,
`cv/probcache_seedless`), with `run_cache`. The scoring was the default except for
`Post.emptyToSeedless`.

| `emptyToSeedless` | mean IoU | max \|dIoU\| vs reference | images changed |
|---|---|---|---|
| false | 0.660423 | 5e-16 | (the reference reproduces) |
| true | 0.660423 | 5e-16 | **0** |

`573` is the only image with points and an empty mask. Its seedless `P` peaks at 0.011 (0.002
inside the GT), so stage 9 also returns 0 px, and it stays at IoU 0. The handoff can only act on
the arm's own cache, where a different seeded `P` may leave other images empty.

## The arm's bundle

`run_export(cnngvf.defaults(), <Output>/colab_bundle_hardneg)` was run on `exp/hard-neg-seeds`,
with `negSeedFrac` 0.20 and `multiScaleSeed` true.
- **Source A:** 30 points on 22 images, as the brief counted.
- **Source B:** 5.5 % to 25.1 % of each frame, median 23.6 %.
- **Epoch at `--rng 0`:**

| fold | positive | negative | share | negatives from Source A |
|---|---|---|---|---|
| 1 | 2 450 | 613 | 0.200 | 91 |
| 2 | 2 550 | 638 | 0.200 | 106 |
| 3 | 2 805 | 701 | 0.200 | 96 |
| 4 | 2 805 | 701 | 0.200 | 93 |
| 5 | 2 850 | 713 | 0.200 | 123 |

The positive lists are the reference run's: they are drawn first, from the same rng.

**Checks run:**
- `test_cnngvf`: 13 checks, all pass.
- `test_parity.py`, stages 1–2, on the reference bundle and on this one.
- `test_parity.py selftest`. Its multi-scale check skips locally, because there is no
  `segmentation_models_pytorch` here; stage 3 on Colab runs it.

## CV result

Pending: the 5-fold run on Colab (`runs/hardneg_rng0`), the stall check, then scoring against
the gate.
