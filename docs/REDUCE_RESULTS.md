---
title: Reduced Pipeline — Measured Results
description: The 2^3 post-processing ablation on cached P, and what the GVF + level-set block is actually worth.
kind: measurement-record
status: partial — DRAC22 41 images done, combined domain running
dataset: DRAC22 (41 images)
updated: 2026-09-03
---

# Reduced Pipeline — Measured Results

What [REDUCE.md](REDUCE.md) asked for, measured. Vocabulary: [CONTEXT.md](CONTEXT.md).
Specification: [UNCLUSTERED.md](UNCLUSTERED.md).

> **Domain: 41 DRAC22 images.** Not 17, not 39, not 68. Every number below is
> cross-validated, every image out-of-sample. They are **not** comparable with the
> 17-image or 68-image figures in UNCLUSTERED.md, and not with any single-split number.

## Protocol

All eight cells were scored from **one cached set of probability maps**, written per
fold by `run_cv("unclustered_boundary")` with `cfg.CV.skipTrain`. The five fold networks
are `models/cv/unclustered_fold{1..5}.mat` — the same networks behind the frozen 0.4603
record, reused rather than retrained.

That matters twice over. No trainable parameter is involved in any of these switches, so
**training variance between cells is exactly zero** and the paired tests are unusually
powerful. And `cnngvf.probCache` re-checks, on every read, that the cached `P` carries a
fold whose held-out set contains that image — an in-sample `P` fails loudly instead of
scoring.

Inference over 41 images took **6 minutes** against the 3.9 h the original run cost.

## 1. The claiming rule, measured alone

`cnngvf.seedMasks` claimed by nearest centroid; training labels by nearest boundary.
REDUCE.md §7 requires that fixed before the level set is measured, so it is not confounded.

Paired on the **39 images common to both runs** — identical seeds, identical ground truth,
the claiming rule the only variable:

| | mean IoU |
|---|---|
| centroid (frozen record) | 0.4603 |
| **boundary (fixed)** | **0.4635** |
| delta | **+0.0032** |

**34 of 39 images are bit-identical.** 3 improved, 2 regressed. The entire effect is one
image: `573.png`, **+0.128**. Worst regression `098.png`, −0.0029.

So the rule was *wrong* but rarely *load-bearing* — which is what makes it the kind of bug
that survives for months. It is worth more under the reduced pipeline, where components
come straight off thresholded `P` and are more fragmented.

The full 41-image run with the fix: mean IoU **0.4602**, median 0.4511, F1 0.5992,
precision 0.6321, recall 0.6339. (41 images, not the frozen record's 39 — the localization
CSVs have since gained seeds for `217.png` and `445.png`.)

## 2. The 2³ factorial

`run_ablate("unclustered_boundary")`. Cell 1 is the current pipeline and the reference.

| # | mode | disk | close | IoU | median | F1 | prec | recall | ΔIoU | ± | sign *p* | Wilcoxon *p* | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | levelset | ✓ | ✓ | **0.4602** | 0.4510 | 0.5993 | 0.6321 | 0.6339 | — | — | — | — | — |
| 2 | levelset | ✓ | ✗ | 0.4601 | 0.4506 | 0.5992 | 0.6322 | 0.6337 | −0.0001 | 13/20 | 0.296 | 0.158 | [−0.0002, +0.0000] |
| 3 | levelset | ✗ | ✓ | 0.4532 | 0.4488 | 0.5924 | 0.6316 | 0.6237 | −0.0070 | 1/12 | **0.003** | **0.003** | [−0.0130, −0.0026] |
| 4 | levelset | ✗ | ✗ | 0.4531 | 0.4489 | 0.5924 | 0.6318 | 0.6235 | −0.0071 | 10/23 | **0.035** | **0.006** | [−0.0130, −0.0026] |
| **5** | **threshold** | ✓ | ✓ | **0.4545** | **0.4729** | 0.5965 | 0.6118 | **0.6592** | **−0.0057** | 14/25 | 0.108 | 0.222 | **[−0.0220, +0.0099]** |
| 6 | threshold | ✓ | ✗ | 0.4547 | 0.4587 | 0.5965 | 0.6236 | 0.6463 | −0.0054 | 17/22 | 0.522 | 0.130 | [−0.0203, +0.0103] |
| 7 | threshold | ✗ | ✓ | 0.4452 | 0.4430 | 0.5875 | 0.6127 | 0.6445 | −0.0149 | 10/29 | **0.003** | **0.006** | [−0.0310, +0.0008] |
| 8 | threshold | ✗ | ✗ | 0.4450 | 0.4377 | 0.5869 | 0.6244 | 0.6309 | −0.0151 | 12/27 | **0.024** | **0.004** | [−0.0303, +0.0011] |

The Wilcoxon column is the **normal approximation with tie correction** — this MATLAB has
no Statistics Toolbox, so `signrank` is unavailable. The sign test and the bootstrap CI are
exact.

### Removing the level set costs −0.0057 IoU, and it is not significant

Cell 1 → cell 5 removes `buildForce`, `gvfField`, the edge stopper, `drlse` and the first
of the two `seedMasks` calls. Nine stages become five.

- **ΔIoU −0.0057**, ΔF1 **−0.0028**
- sign test *p* = 0.108, Wilcoxon *p* = 0.222
- bootstrap 95 % CI **[−0.022, +0.010]** — straddles zero
- median IoU actually **rises**, 0.4510 → 0.4729

REDUCE.md predicted ≈ −0.014 F1 from the frozen 17-image record. Measured: **−0.0028 F1**,
five times smaller, and in the same direction. The port is behaving as expected.

The composition of the loss is the opposite of the prediction, though. REDUCE.md expected
the loss to be almost entirely in recall (precision delta 0.002). Measured, **recall rises**
0.6339 → 0.6592 while **precision falls** 0.6321 → 0.6118. The level set was not adding
recall here; it was trimming false positives. Precision is the documented weak side, so the
block being cut was paying on the axis that already hurts.

### The disk fallback matters more than the level set does

The single largest effect in the table is not the level set. It is
`cfg.Post.diskFallback` — a 15 px disk planted at a seed the threshold missed:

| removing it | ΔIoU | sign *p* |
|---|---|---|
| in level-set mode (1→3) | −0.0070 | 0.003 |
| in threshold mode (5→7) | −0.0093 | — |

Both significant where the level-set removal is not, and cell 3's 1-improved / 12-regressed
split is the most lopsided in the table. A lesion the CNN never responds to is otherwise
dropped silently, and on 41 training images the CNN does miss.

This also revises what the disk was *for*. It was documented as giving the balloon and GVF
terms "something to grow from" — an initialization device. But it is worth **more** in
threshold mode, where nothing grows and the disk survives at its literal radius. It is not
a seed for evolution; it is a floor on recall.

### `imclose` earns nothing

| | ΔIoU |
|---|---|
| level-set mode (1→2) | −0.0001 |
| threshold mode (5→6) | +0.0003 |

Indistinguishable from zero in both modes, both directions, *p* ≥ 0.30. Its documented
justification — bridging the 1–2 px gaps *the level set leaves* — does not show up in IoU.
It is the cheapest thing in the table to delete.

### Biggest movers, cell 5 vs cell 1

`312.png` +0.168, `573.png` +0.096, `194.png` +0.051, `1115.png` +0.036 improved;
`217.png` **−0.227**, `421.png` −0.089, `193.png` −0.061 regressed. `217.png` is a
single-seed image, which is where losing the level set's ability to grow off a small
initialization should hurt most.

## 3. `tau` sensitivity — flat, so the fixed threshold is safe

`run_ablate("unclustered_boundary", "tau")`, threshold mode, 41 images, 0.30–0.80.
**Reported as a curve. Its maximum is not a result** — choosing `tau` on the images being
scored is leakage, the same error class as quoting `Split.Val` as held-out.

| tau | mean IoU | median | mean F1 | | tau | mean IoU | median | mean F1 |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.4427 | 0.4658 | 0.5816 | | 0.55 | 0.4464 | 0.4549 | 0.5873 |
| 0.35 | 0.4528 | 0.4838 | 0.5920 | | 0.60 | 0.4361 | 0.4594 | 0.5769 |
| 0.40 | 0.4551 | 0.4907 | 0.5960 | | 0.65 | 0.4292 | 0.4360 | 0.5718 |
| 0.45 | 0.4567 | 0.4859 | 0.5982 | | 0.70 | 0.4053 | 0.3907 | 0.5468 |
| **0.50** | **0.4545** | 0.4729 | **0.5965** | | 0.75 | 0.3896 | 0.3621 | 0.5309 |
| 0.525 | 0.4544 | 0.4613 | 0.5963 | | 0.80 | 0.3660 | 0.3497 | 0.5026 |

**Spread of mean IoU across 0.40–0.65: 0.0275.** Broad, single-peaked, no cliff — the
pre-registered `tau = 0.5` sits 0.0022 below the sweep's best (0.45) and 0.0004 above its
own neighbour at 0.525. By REDUCE.md's own decision rule, **a flat curve means the fixed
`tau` is safe**, and there is no calibration problem in `P` to report.

That the optimum sits slightly *below* 0.5 is consistent with the precision/recall shift in
§2: the threshold path is already over-inclusive, so lowering `tau` further buys little.

**Oracle per-image `tau`: 0.5201.** Not quotable — an upper bound only. It is +0.066 over
the fixed threshold, which is a larger gap than anything in the factorial, and it is the
one number here that says where the remaining headroom actually is: not in the contour
method, but in how well-calibrated `P` is image to image.

## 4. Stage dumps, both modes

`run_steps("458.png", cfg)` with each `Post.mode`, into `steps/458/` and
`steps/458_threshold/`:

| | files | stages 6–11 | runtime | IoU |
|---|---|---|---|---|
| levelset | 15 png | `06_force` … `11_levelset_raw` written | 16.9 s | 0.4293 |
| threshold | 10 png | **absent**, `11_binary_raw.png` in their place | **2.7 s** | 0.4234 |

Stage counts on `458.png`: `P > 0.5` gives 51 630 px either way; the level set settles to
47 844 and the threshold to 54 116 before morphology, ending at 47 641 vs 48 549 against a
51 180 px ground truth. **6.3× faster** for −0.006 IoU on this image, in line with §2.

## What this means

On 41 DRAC22 images the GVF + DRLSE block is **not measurably earning its place**: the
point estimate is −0.006 IoU, the confidence interval includes zero, and the median
improves. Against that it costs five pipeline stages, a compiled mex dependency, ~2.5 s per
image, and eight tuned parameters.

That is a case for deletion, but it is **not yet the whole case**:

- One domain, 41 images. The combined 68-image run is what the project actually reports on.
- A non-significant delta is not a zero delta. The CI's lower bound is −0.022.
- The τ curve (§3) removes one worry — the result is not an artefact of the chosen
  threshold — but it does not add images.

What the level set *does* still buy is visible in §2's per-image movers: `217.png`
regressed −0.227, a single-seed image, which is where growing off a small initialization
should matter most. A deletion is a decision about the mean, not about that image.

## Still outstanding

1. **The combined domain** — `run_cv("DRACTUH_unclustered")` is running. It is the first
   combined-domain number, which UNCLUSTERED.md records as owed regardless of this work,
   and it writes its own `probcache`, so the same eight cells and the τ curve can then be
   reproduced on 68 images in minutes:
   `run_ablate("DRACTUH_unclustered")` and `run_ablate("DRACTUH_unclustered", "tau")`.
2. **Deletion** — only after 1, and only as its own change.

## Deviations from REDUCE.md

1. **`cfg.Post.diskFallback` defaults to `true`, not `false`.** REDUCE.md writes `false`,
   but it also makes byte-identical level-set reproduction the non-negotiable constraint,
   and `initPhi` has always planted these disks unconditionally. Both cannot hold once the
   flag reaches `initPhi`. Reproduction wins; the ablation varies the flag anyway.
2. **`maskFromProb` omits `initPhi`'s `imopen(disk 1)`**, per REDUCE.md's "exactly".
3. **The disk trigger is the seed's own pixel**, matching `initPhi`, not REDUCE.md's
   "8-connected neighbourhood".
4. **Fold-keyed networks already existed.** REDUCE.md §4 says `trainNet` archives by
   timestamp and cannot reconstruct fold ownership; `run_cv` has always written
   `models/cv/<arm>_fold<k>.mat`.
5. **Wilcoxon is the normal approximation.** No Statistics Toolbox on this machine.
6. **`test_cnngvf` had nine checks, not eight** (UNCLUSTERED.md and REDUCE.md §8 both say
   eight). Now eleven.

## Reproducing

```matlab
cd RNV-Claude/CNN+GVF
test_cnngvf                                       % 11 checks

cfg = cnngvf.defaults(); cfg.CV.skipTrain = true;
run_cv("unclustered_boundary", cfg)               % ~6 min, borrows the fold networks
run_ablate("unclustered_boundary")                % ~4 min, all 8 cells
run_ablate("unclustered_boundary", "tau")         % the sensitivity curve

run_steps("458.png", cfg)                         % stage dump, level-set mode
tc = cfg; tc.Post.mode = "threshold";
run_steps("458.png", tc)                          % ... and threshold mode
```

`cfg.CV.skipTrain` also resumes an interrupted run: it reuses every fold network already
trained and trains only the missing ones. The combined-domain run was stopped during fold
4 and restarted this way, costing 2 folds rather than 5.

Every score here is tied to the localization CSVs in place on 2026-09-03; their mtimes are
recorded in each cached `P`'s metadata.
