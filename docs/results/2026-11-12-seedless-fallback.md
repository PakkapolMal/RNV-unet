# A seedless network on the three point-less images is worth +0.024 IoU and passes its gate (2026-11-12)

Brief: [../briefs/seedless-fallback.md](../briefs/seedless-fallback.md).

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- 5-fold CV with the reference's folds;
- the seedless network trained once (`--rng 0`) on a Colab L4. Its cache is in
  `result/DRACTUH_unclustered/colab_seedless/`;
- the no-point rule cross-fitted by `run_seedless_fit` on the 18 images with 1–3 points
  (the proxy), then scored on the 3 images with none (the targets);
- whole-image scoring; the other 66 images keep the reference's `P` and pipeline.

**Per-image scores**, rule on at the cell that ships:
[2026-11-12-seedless-fallback.csv](2026-11-12-seedless-fallback.csv).

## Checks before scoring

- **Stall rule, relative form:** no fold stalled. The last-five-epoch mean training losses are
  0.052, 0.054, 0.052, 0.051 and 0.052. The median is 0.052 and the threshold is 0.065.
- **No deviation from the brief:** minibatch 4 fit; 60 epochs; 550 frames per epoch (560 in
  fold 5, which trains on 56 images); about 68 s per epoch.
- **Provenance:** all 69 maps come from `rng 0 epoch 60`, seedless, one per held-out image,
  each stamped with its own fold. Every read went through `cnngvf.probCache`'s leakage guard.

## Result: all three gate conditions hold

| image | fold | GT px | chosen cell | predicted px | IoU | precision | recall |
|---|---|---|---|---|---|---|---|
| `082` | 3 | 2 408 | (0.40, all) | 2 683 | **0.788** | 0.836 | 0.931 |
| `217` | 4 | 601 | (0.40, all) | 668 | **0.863** | 0.880 | 0.978 |
| `445` | 2 | 4 052 | (0.30, all) | 0 | 0 | 0 | 0 |

**Gate:**
1. **Gain: holds.** The three-target mean IoU is **0.550** against the 0.115 floor: +0.0239 on
   the headline.
2. **It generalises: holds.** The proxy's cross-fitted mean IoU is **0.467** against the 0.20
   floor (precision 0.550, recall 0.554, 18 images).
3. **Isolation: holds.** On the 66 images with points, max |dIoU| against the reference is
   5.6e-16.

**This is a floor, not a significance test** (n = 3). The rule cannot lower an image, and the
other 66 images carry no training noise, because their `P` is the reference's.

| rule on, cell (0.40, all) | n | mean IoU | F1 | precision | recall |
|---|---|---|---|---|---|
| **all images** | 69 | 0.6125 → **0.6364** | 0.7273 → 0.7535 | 0.7418 → 0.7667 | 0.7348 → 0.7625 |
| DRAC22 | 42 | 0.5404 → 0.5797 | 0.6585 → 0.7015 | 0.6960 → 0.7368 | 0.6471 → 0.6926 |
| TUH | 27 | 0.7245 (unchanged) | 0.8343 | 0.8131 | 0.8713 |

The median IoU moves from 0.6494 to 0.6694. The per-dataset rows break out the pooled folds
and are not a separate protocol.

**What `445` shows:** the seedless network gives `P` = 0.000 everywhere inside its lesion. Its
only component above 0.3 is a 63 px blob elsewhere, which `MinArea` removes. The miss is the
network's, not the threshold's.

## The proxy surface (description)

Mean IoU of each fixed cell on the 18 proxy images, and on the 3 targets:

| `tau` | proxy, all | proxy, top-1 | targets, all |
|---|---|---|---|
| 0.30 | 0.470 | 0.452 | 0.550 |
| 0.40 | **0.480** | 0.450 | 0.550 |
| 0.50 | 0.478 | 0.448 | 0.553 |
| 0.60 | 0.476 | 0.445 | 0.555 |
| 0.70 | 0.473 | 0.442 | 0.558 |
| 0.80 | 0.470 | 0.438 | 0.560 |
| 0.90 | 0.464 | 0.431 | 0.561 |

- **The threshold barely matters.** The proxy spans 0.016 over 0.30–0.90, and the targets
  0.011. A seedless `P` is close to binary.
- **Top-1 is worse at every threshold,** by about 0.03 on the proxy. It never won a fold, so
  the domain-wide RNV-positive assumption it needs is not used. On the targets top-1 equals
  all, because each has a single component.
- **Fold choices:** (0.50, all), (0.30, all), (0.40, all), (0.40, all), (0.40, all). The cell
  chosen on all 18 proxy images, which ships, is (0.40, all).

**Diagnostic, the rule on every image** (the seedless network alone, cell (0.40, all)): DRAC22
0.520 IoU (precision 0.703, recall 0.589), TUH 0.473 (precision 0.761, recall 0.530). That is
below the seeded pipeline on both datasets (0.540 and 0.725), and far below on TUH, whose large
lesions the seeded patches tile.

**Seedless against seeded `P`** (both above 0.5, the 66 images with points, mean fraction per
image):
- **inside GT:** 49.2 % of pixels are found by both; **6.4 %** by the seedless network only;
- **outside GT:** 0.27 % by both; 0.21 % by the seedless network only.

That 6.4 % is lesion the seeded network misses and the seedless one finds. A later fusion arm
would aim at it.

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| no fold stalls | held |
| proxy cross-fitted IoU 0.15–0.35 | **wrong, above:** 0.467 |
| folds choose `tau` ≥ 0.6 and top-1 | **wrong:** 0.30–0.50, and all on every fold |
| three-target mean IoU 0.00–0.25; `217` stays at 0 | **wrong, above:** 0.550; `217` scores 0.863, the best of the three |
| diagnostic over all 69: IoU 0.35–0.48, below seeded on both | held for TUH (0.473); DRAC22 above the range (0.520); below seeded on both |
| about a 30 % chance of passing | passed |

I expected whole-frame false positives to dominate small-lesion images. They do not: the
seedless network's output on the point-less and point-poor images is mostly a single
confident component on the lesion.

## Reading

- **The rule passes its gate.** On the three images with no localization points it scores 0.550
  mean IoU, against 0 before. That lifts the 69-image headline from 0.6125 to **0.6364**
  (5-fold CV), with the 66 other images untouched.
- **The estimate that matters is the proxy's, 0.467 over 18 images.** The targets' 0.550
  rests on 3 images, two of which landed well and one missed entirely.
- **The rule needs no tuning to work.** Every threshold from 0.3 to 0.9 gives the targets
  0.55–0.56. The cross-fit chose 0.40 on images that never include the targets.
- **What adoption requires**, per the brief:
  - `cnngvf.noPointMask` becomes a pipeline stage, with `noPointTau` 0.40. The keep switch
    goes: top-1 lost on every fold, so `"all"` is the rule;
  - the seedless cache path becomes a default;
  - the trainer's `SEEDLESS` constants move to `cfg.Net` and the bundle;
  - a new reference CSV, `cfg.CV.reference`, the README table, and SPEC (a stage, the cfg
    fields);
  - `run_seedless_fit` is experiment code: archive and delete it;
  - `run_batch` and `run_steps` still have no seedless network: a separate change.
