# Whole-image fusion replicates on a second seedless network: +0.010 IoU, CI above 0 (2026-11-12)

Brief: [../briefs/fusion-confirm.md](../briefs/fusion-confirm.md). This record settles the
decision left open by [2026-11-12-fusion-uncertainty.md](2026-11-12-fusion-uncertainty.md),
whose gate 2 failed on a 3–2 τ_s split.

**Domain:** the standardized 69 images (DRAC22 42 + TUH 27).

**Protocol:**
- the adopted seeded 5-fold CV cache, and a **second** seedless network: same recipe,
  `--rng 1`, trained on a Colab L4, cache in `result/DRACTUH_unclustered/colab_seedless/rng1/`;
- two cells fixed before training:
  - **(a)** max(seeded `P`, seedless `P` > 0.9) over the whole image, then the unchanged
    pipeline;
  - **(b)** (a), keeping only the pieces it adds of at least `MinArea`;
- the 3 images with no points keep stage 9 on the adopted seedless cache in every cell;
- nothing selected, so every score is out-of-sample as it stands;
- paired per image, bootstrap 10 000 resamples, `rng(20260917)`.

**Per-image scores:** [2026-11-12-fusion-confirm.csv](2026-11-12-fusion-confirm.csv).

Scorer: `run_fusion_confirm`.

## Checks before scoring

- **Stall rule, relative form:** no fold stalled. The last-five-epoch means are 0.053, 0.050,
  0.050, 0.050 and 0.051. The median is 0.050 and the threshold 0.063.
- **Provenance:** all 69 maps come from `rng 1 epoch 60`, seedless, each stamped with its own
  fold.
- **Sanity:** off reproduces
  [reference/2026-11-12-seedless-fallback-69.csv](reference/2026-11-12-seedless-fallback-69.csv),
  max |dIoU| 5.6e-16.

## Result: (a) passes its gate; (b) does not replace it

On the rng 1 seedless map:

| cell | IoU | precision | recall | DRAC22 | TUH |
|---|---|---|---|---|---|
| off (current) | 0.6364 | 0.7667 | 0.7625 | 0.5797 | 0.7245 |
| **(a) fuse all** | **0.6468** | 0.7619 | 0.7803 | 0.5939 | 0.7291 |
| (b) fuse all, no slivers | 0.6417 | 0.7650 | 0.7707 | 0.5882 | 0.7249 |

| comparison | ΔIoU | 95 % CI | up / down | Δ Q1–Q3 precision |
|---|---|---|---|---|
| **(a) − off** | **+0.0104** | **[+0.0040, +0.0192]** | 41 / 17 | −0.0019 |
| (b) − off | +0.0053 | [−0.0004, +0.0140] | 16 / 6 | +0.0015 |
| (b) − (a) | −0.0051 | [−0.0077, −0.0027] | 15 / 43 | +0.0034 |

**Gate:**
- **(a): adopted.** Its CI lower bound is above 0 (+0.0040), and Q1–Q3 precision falls by
  0.0019, inside the 0.01 guard.
- **(b): not adopted.** Its own CI includes 0, and it scores below (a) with a CI below 0. The
  pieces under `MinArea` that it drops carry more lesion gain than loss.

## Replication

Both cells, fixed, on both seedless runs:

| | rng 0 (adopted map) | **rng 1 (this test)** |
|---|---|---|
| (a) − off | +0.0120 [+0.0046, +0.0225] | **+0.0104 [+0.0040, +0.0192]** |
| (b) − off | +0.0075 [+0.0007, +0.0177] | +0.0053 [−0.0004, +0.0140] |
| (a) − off without `795` and `461` | +0.0061 | +0.0059 |
| median per-image (a) − off | +0.0017 | +0.0008 |
| `795` | +0.284 | +0.248 |
| `461` | +0.139 | +0.075 |

The rng 0 figures are in-sample for the selection made on that map in
[2026-11-12-fusion-uncertainty.md](2026-11-12-fusion-uncertainty.md); its cross-fitted figure
there was +0.0116.

- **The gain replicates in size and in place.** The per-image (a) − off correlates at 0.947
  between the two runs, and 42 of the 59 images that moved under either run moved the same way.
- **The concentration replicates too.** Without `795` and `461` the gain is +0.006 on both
  runs. Two very large DRAC22 lesions, whose extent the seeded patches do not reach, carry
  about half of it on both.

By quintile, (a) against off on rng 1:

| quintile | ΔIoU | Δ precision | Δ recall |
|---|---|---|---|
| Q1 | +0.004 | −0.006 | +0.009 |
| Q2 | +0.006 | −0.004 | +0.014 |
| Q3 | +0.008 | +0.005 | +0.008 |
| Q4 | +0.009 | −0.009 | +0.020 |
| Q5 | +0.026 | −0.010 | +0.039 |

**Largest losses under (a) on rng 1:** `458` (−0.039), `nv (5)` (−0.026), `109` (−0.019), `312`
(−0.011), `123` (−0.010). `458` loses on both runs: its seedless extension is not lesion.

**The seedless network's training noise, on the point-less targets** (stage 9; description
only, nothing changes): `082` 0.788 → 0.822, `217` 0.863 → 0.854, `445` 0 → 0.363. The mean
moves from 0.550 to 0.680 between two runs of one recipe. Single images swing widely; the stage
9 figure rests on one run.

## Prediction, checked

| prediction (from the brief) | outcome |
|---|---|
| no fold stalls | held |
| (a) − off +0.004 to +0.012, CI above 0; `795` and `461` gain again | held: +0.0104, CI [+0.004, +0.019]; +0.248 and +0.075 |
| (b) − off within ±0.003 of (a) − off | **wrong:** 0.0051 lower |
| (b) − (a) CI includes 0 | **wrong:** CI [−0.0077, −0.0027], below 0 |
| the point-less targets within ±0.15 of 0.550 | held: 0.680 (+0.130) |
| (a) about 55 %, (b) replacing it about 20 % | (a) passed; (b) did not replace it |

## Reading

- **Whole-image fusion is a real gain, and the U-gated record's open decision is settled.** On
  a second, independent seedless network it gives +0.0104 IoU (69 images, 5-fold CV, CI
  [+0.004, +0.019]). The first network gave +0.0120 for the same fixed cell, and +0.0116
  cross-fitted.
- **The gain is a trade:** recall +0.018, precision −0.005. It is largest on the largest
  lesions (Q5 +0.026 IoU).
- **About half of it is two images on both runs.** Reported, not hidden: the other 67 images
  gain +0.006.
- **The sliver filter, designed from GT on these images, did not survive the independent map.**
  It is not adopted. It is recorded here, so it is not re-tried without new data.
- **What adoption requires**, per the brief:
  - a fixed max fusion at τ_s 0.9 on every image with points becomes a pipeline stage, and U
    and its cut-off go;
  - the adopted seedless cache stays the rng 0 run, so the new reference is the rng 0 figure of
    the fixed cell (a), 0.6484. The rng 1 run is the evidence;
  - a new reference CSV, `cfg.CV.reference`, the README table and SPEC.
