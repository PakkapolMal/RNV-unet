---
title: Confirming Whole-image Fusion on a Retrained Seedless Network
description: Pre-registered confirmation. Two fixed fusion cells, chosen before training, are scored on the P of a second seedless training run (--rng 1). Nothing is tuned on the scored map. Prediction and gate written before the run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/colab (user runs), RNV-Claude/cnn (scoring)
depends-on: results/2026-11-12-fusion-uncertainty.md, results/2026-11-12-seedless-fallback.md
status: executed 2026-11-12
---

# Confirming Whole-image Fusion on a Retrained Seedless Network

> **Executed brief.** Kept as written. Outcome: (a) passes, +0.0104, CI [+0.0040, +0.0192].
> (b) does not replace it: (b) − (a) is −0.0051, CI below 0.
> [../results/2026-11-12-fusion-confirm.md](../results/2026-11-12-fusion-confirm.md).

## Why

**Fusing the seedless `P` into every image** gave +0.0116 IoU cross-fitted, CI [+0.004,
+0.022] (69 images, 5-fold CV; [2026-11-12](../results/2026-11-12-fusion-uncertainty.md)). It
is not adopted, for three reasons:
- **gate 2 failed:** all five folds chose to fuse every image, but they split 3–2 on τ_s;
- **the gain is concentrated:** `795` and `461` carry 53 % of it;
- **one network:** the seedless network was trained once, so its training noise is unmeasured.

**The cell no longer needs to be chosen.** The selection question is settled by that run:
fuse every image, and τ_s 0.9 wins the tie. What is still open is whether the gain belongs to
fusion or to one seedless network.

**So this brief replicates on an independent seedless map.** A second training run, same
recipe, `--rng 1`, gives a new out-of-fold seedless `P` for all 69 images. Two cells fixed now
are scored on it, with nothing tuned.

**The second cell tests one mitigation.** The losing images got mostly thin seedless rims of
about 100–150 px, little of which is lesion. The winners got large connected extensions,
80 % lesion. The rule below was designed from that GT diagnostic on the same 69 images, so it
is judged only on the new map, and it is pre-registered here.

## Cells

Both apply to the 66 images with points. The 3 images with no points keep stage 9 on the
adopted seedless cache, as do all images under off.

| cell | mask |
|---|---|
| **off** | the current pipeline: the adopted reference |
| **(a) fuse all** | `P_f` = max(seeded `P`, seedless `P` where it is > 0.9, else 0), over the whole image; then `cnngvf.maskFromProb` unchanged |
| **(b) fuse all, no slivers** | as (a), then keep only the pixels (a) adds in connected pieces of at least `MinArea` (500 px). Final mask = off mask ∪ those pieces. Pixels (a) removes stay removed. |

In (b), "the pixels (a) adds" means mask(a) minus the off mask, taken as 8-connected
components. `MinArea` is the pipeline's existing value. No new parameter is introduced.

## The training run (Colab)

- **Command:** `rnvtorch.py train-seedless`, folds 1–5, `--rng 1`, into `runs/seedless_rng1/`.
- **Recipe:** the adopted one from the bundle: 60 epochs, minibatch 4, 10 copies per image.
- **The bundle must be the current one** (`run_export` of 2026-11-12), which carries the
  `seedless*` settings. The trainer refuses an older one.
- **Stall rule, relative form, before scoring:** a fold whose last-five-epoch mean loss exceeds
  1.25 × the median of the five is retrained once with `--rng 2`, and that retrain is used.
- **Cost:** about 68 s per epoch on an L4: about 1.1 h per fold, 5.7 h per CV.

## Protocol

- 69 images (DRAC22 42, TUH 27), the adopted seeded cache, and the new seedless cache.
- **Sanity check:** off reproduces `reference/2026-11-12-seedless-fallback-69.csv` to ~1e-15.
- **Comparisons, paired per image:** (a) − off, (b) − off and (b) − (a). Percentile bootstrap
  on the mean, 10 000 resamples, `rng(20260917)`, 95 % CI.
- **No selection:** both cells are fixed above, so nothing is cross-fitted, and the scores are
  out-of-sample as they stand.
- Quintiles by GT area, Q1 smallest, 14/14/14/14/13.

## Prediction (written before running)

- **No fold stalls.**
- **(a) − off: +0.004 to +0.012,** with a CI lower bound above 0. `795` and `461` gain again,
  because the seedless network sees beyond the seeded patches whatever its seed.
- **(b) − off ≈ (a) − off within ±0.003.** (b) recovers most of (a)'s small-lesion losses, and
  costs a little on the winners' smaller pieces.
- **(b) − (a) CI includes 0.**
- **The seedless network replicates:** the three point-less targets score within ±0.15 mean IoU
  of the rng 0 run's 0.550 (description only; stage 9 is not changed here).
- I give (a) about a 55 % chance of passing, and (b) replacing it about 20 %.

## Gate

**Adopt (a)** if both hold:
1. (a) − off has a 95 % CI lower bound above 0.
2. **Precision guard:** Q1–Q3 precision under (a) falls by no more than 0.01 against off.

**Adopt (b) instead** only if (b) passes the same two tests **and** (b) − (a) has a 95 % CI lower
bound above 0. Otherwise the simpler rule, (a), is the one adopted.

**If (a) fails,** fusion is rejected. Tag `archive/fusion-uncertainty`, delete the code, and
record the result, with both seedless runs' numbers.

**On adoption:**
- **Fusion becomes a stage without U:** `cnngvf.fuseUncertain` is replaced by a fixed max
  fusion at τ_s 0.9 (plus the sliver filter if (b) wins). `seededUncertainty` and the cut-off
  field go.
- **The adopted seedless cache stays the rng 0 run.** The reference figure is the one measured
  on it. The rng 1 run is the evidence, and its cache is kept outside git.
- **This replication is the documented reason gate 2 of the U-gated brief was set aside.**
- **Updates:** a new reference CSV, `cfg.CV.reference`, the README table, and SPEC (the stage,
  the diagram, the fields).

## Report alongside, as description

- (a) and (b) on the rng 0 seedless map as well (both fixed cells; in-sample for the selection
  already made there);
- per dataset and by quintile;
- the images that lose under (a), and what (b) does to each;
- `795` and `461` under both runs;
- the stage 9 targets on the rng 1 map.

**Valid until either `P` changes.**
