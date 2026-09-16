---
title: Reduced Pipeline — Measured Results
description: The 2^3 post-processing ablation on cached P, and what the GVF + level-set block is actually worth.
kind: measurement-record
status: current — both domains measured
dataset: DRAC22 + TUH (68) and DRAC22 (41)
updated: 2026-09-16
---

# Reduced Pipeline — Measured Results

What [REDUCE.md](REDUCE.md) asked for, measured. Vocabulary: [CONTEXT.md](CONTEXT.md).
Specification: [UNCLUSTERED.md](UNCLUSTERED.md).

> **Two domains, stated on every number.** §0–§4 are **41 DRAC22 images**
> (`unclustered_boundary`); §5–§7 are the **68-image combined domain**
> (`DRACTUH_unclustered`). Both are 5-fold cross-validated with every image scored
> out-of-sample. They answer different questions and **must not be compared with each
> other**, nor with the 17- or 39-image figures in UNCLUSTERED.md, nor with any
> single-split number.
>
> **The 68-image results in §5–§7 are the ones to report.** The 41-image work came
> first because its fold networks already existed; it is kept because it is the paired
> evidence for the claiming-rule fix, and because the two domains agree.

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

---

# The combined domain — 68 images

## 5. The first combined-domain number

`run_cv("DRACTUH_unclustered")`. UNCLUSTERED.md records this as owed regardless of the
reduced-pipeline work: **no cross-validated figure over both datasets had ever existed.**

| 5-fold CV, 68 images, all out-of-sample | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| **DRACTUH_unclustered** | **0.4708** | 0.5026 | 0.6137 | 0.6981 | 0.6432 |

Folds 0.3623 / 0.5289 / 0.5361 / 0.4496 / 0.4763 — sd ≈ 0.070, comparable to the DRAC22
run's 0.082, so read fold differences as noise. Runtime 4.3 h.

| breakout | n | IoU | F1 |
|---|---|---|---|
| TUH | 27 | 0.4788 | 0.6281 |
| DRAC22 | 41 | 0.4656 | 0.6041 |

The two devices land within **0.013** of each other. That is worth noting given the folds
are pooled rather than device-stratified, and given TUH still carries its burned-in
acquisition overlay — which was the standing suspect for straight-line false positives.
Nothing here suggests it is costing much.

Integrity: 68 rows, 68 unique, 68 cached `P`, 5 fold networks, exit 0. The only two images
scoring 0 are `299.png` and `445.png`, exactly the pair UNCLUSTERED.md documents as having
ground truth but no usable seed — a localization gap charged to the segmenter, as intended.

**This supersedes the `perSeed = 1` single-split 0.3852**, which was diagnosed as an
underfit. It does **not** compare with the DRAC22-only 0.4603 (41 images ≠ 68), but this
run's own DRAC22 subset, 0.4656, is a legitimate same-domain check and sits slightly above.

## 6. The factorial on 68 images — the level set all but disappears

| # | mode | disk | close | IoU | median | F1 | prec | recall | ΔIoU | ± | sign *p* | Wilcoxon *p* | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | levelset | ✓ | ✓ | **0.4708** | 0.5026 | 0.6137 | 0.6981 | 0.6432 | — | — | — | — | — |
| 2 | levelset | ✓ | ✗ | 0.4708 | 0.5025 | 0.6136 | 0.6982 | 0.6430 | −0.00009 | 18/41 | 0.004 | 0.001 | [−0.00014, −0.00004] |
| 3 | levelset | ✗ | ✓ | 0.4548 | 0.4876 | 0.5962 | 0.6975 | 0.6223 | −0.0161 | 2/31 | **1.3e−7** | **8.2e−7** | [−0.0225, −0.0103] |
| 4 | levelset | ✗ | ✗ | 0.4547 | 0.4874 | 0.5961 | 0.6976 | 0.6221 | −0.0161 | 15/45 | 1.3e−4 | 6.9e−7 | [−0.0226, −0.0104] |
| **5** | **threshold** | ✓ | ✓ | **0.4691** | 0.4727 | 0.6131 | 0.6686 | **0.6653** | **−0.0017** | 29/37 | **0.389** | **0.406** | **[−0.0113, +0.0081]** |
| 6 | threshold | ✓ | ✗ | 0.4656 | 0.4797 | 0.6102 | 0.6826 | 0.6455 | −0.0052 | 21/45 | 0.004 | 0.019 | [−0.0119, +0.0018] |
| 7 | threshold | ✗ | ✓ | 0.4519 | 0.4641 | 0.5942 | 0.6733 | 0.6408 | −0.0189 | 17/49 | 1.0e−4 | 1.5e−5 | [−0.0275, −0.0113] |
| 8 | threshold | ✗ | ✗ | 0.4468 | 0.4695 | 0.5889 | 0.6886 | 0.6193 | −0.0241 | 13/53 | 7.2e−7 | 2.8e−8 | [−0.0319, −0.0170] |

### Removing the level set costs 0.0017 IoU — and the estimate shrank with more data

- **ΔIoU −0.0017**, ΔF1 **−0.0005** (0.6137 → 0.6131)
- sign *p* = 0.389, Wilcoxon *p* = 0.406 — nowhere near significance
- 95 % CI **[−0.0113, +0.0081]**, comfortably straddling zero
- 29 images better, 37 worse — near a coin flip

On 41 images the point estimate was −0.0057; on 68 it is **−0.0017**. More data made the
effect *smaller*, not clearer. REDUCE.md predicted ≈ −0.014 F1 from the frozen 17-image
record; measured on the full domain it is **−0.0005 F1**, roughly thirty times smaller.

The precision/recall trade repeats and is sharper here: **precision falls 0.698 → 0.669,
recall rises 0.643 → 0.665**. The level set buys precision at recall's expense — and
precision is the side UNCLUSTERED.md documents as already the stronger of the two on this
arm, so the block is paying into the wrong account.

**On TUH the reduced pipeline is better outright**: cell 5 scores 0.4864 against cell 1's
0.4788. On DRAC22 it is worse, 0.4577 against 0.4656. Fold 1 also prefers the reduced
path (0.3736 vs 0.3623). Whatever the level set contributes, it is not uniform across
devices.

### The disk fallback is the real load-bearing stage

| removing it | ΔIoU | sign *p* |
|---|---|---|
| level-set mode (1→3) | **−0.0161** | 1.3e−7 |
| threshold mode (5→7) | **−0.0172** | 1.0e−4 |

Nine times the level set's effect, with *p*-values seven orders of magnitude smaller, and
cell 3's **2 better / 31 worse** is the most one-sided split in the table. This confirms
the 41-image finding on twice the data, and confirms the reinterpretation: the disk is
worth slightly **more** in threshold mode, where nothing grows — so it is a floor on
recall, not the initialization device it was documented as.

### `imclose`: significant and negligible, and the interaction is real

Cell 1 → 2 gives *p* = 0.004 on a delta of **−0.00009**. With `P` identical across cells
the paired variance is near zero, so the sign test resolves an effect four decimal places
down. **Statistically significant, practically nothing** — a clean illustration that the
two are different questions, and a caution for reading the rest of this table.

But the factorial earns its keep here:

| `imclose` worth | ΔIoU |
|---|---|
| in level-set mode (2→1) | +0.00009 |
| in threshold mode (6→5) | **+0.0035** |

**~40× more in threshold mode.** This is exactly the interaction REDUCE.md predicted when
it insisted on all eight cells rather than a sequential ablation: `imclose` was justified
as bridging gaps *the level set leaves*, and the truth is the opposite — it matters when
there is **no** level set, because thresholded components are more fragmented. A
sequential ablation would have measured it in level-set mode, found ≈ 0, and deleted it
just as it became useful.

### Biggest movers, cell 5 vs cell 1

Worse: `641.png` −0.172, `nv (1).png` −0.080, `429.png` −0.057, `193.png` −0.053.
Better: `nv (21).png` **+0.123**, `319.png` **+0.123**, `nv (3).png` +0.093,
`nv (2).png` +0.058.

`nv (21)` is the 432k px lesion at 41 % of the frame — the reduced pipeline more than
doubles it, 0.1065 → 0.2298. The level set appears to *hurt* on the largest lesions, where
the contour has furthest to travel.

## 7. `tau` on 68 images — flat where it matters, then a cliff

| tau | mean IoU | | tau | mean IoU |
|---|---|---|---|---|
| 0.375 | 0.4734 | | 0.525 | 0.4630 |
| 0.400 | 0.4730 | | 0.550 | 0.4593 |
| 0.425 | 0.4721 | | 0.575 | 0.4504 |
| 0.450 | 0.4698 | | 0.600 | 0.4421 |
| 0.475 | 0.4724 | | 0.625 | 0.4326 |
| **0.500** | **0.4691** | | 0.650 | 0.4181 |

Flat across **0.375–0.525** (0.4691–0.4734, a 0.004 band), then falling steadily. The
quoted spread over 0.40–0.65 is 0.0549, but that is dominated by the decline above 0.55 —
the plateau containing the pre-registered value is genuinely flat, and `tau = 0.5` sits
0.0043 below the sweep's best at 0.375.

So the fixed threshold is safe and there is no calibration problem to report. The optimum
again sits *below* 0.5, consistent with §6's precision/recall shift.

**Oracle per-image `tau`: 0.5468** — not quotable, an upper bound only. At **+0.078** over
the fixed threshold it is once more far larger than anything in the factorial. Both domains
now say the same thing: the remaining headroom is in how well-calibrated `P` is image to
image, not in the contour method.

## 8. The seed fallback — the largest effect measured on this pipeline

§6 found the disk fallback worth −0.016 at *p* = 1.3e−7, an order of magnitude more than
the level set. That prompted the obvious question nobody had asked: **how often does it
fire, and is a 15 px disk the right answer?**

**258 of 874 seeds — 29.5 %, across 34 of the 68 images — never produce `P > tau` at all.**
For every one of them the fallback *is* the output. A disk at `seedRadius = 15` is ~707 px,
barely over `minArea = 500`, against ground-truth lesions of 1 106–432 533 px. And that
radius was never measured: it was chosen when the disk was a level-set *initialization*
that the evolution would grow. In the reduced pipeline nothing grows.

`run_ablate("DRACTUH_unclustered", "fallback")`, 68 images, threshold mode, `tau` fixed at
0.5. Reference is the current default, disk r = 15.

| variant | IoU | median | F1 | prec | recall | ΔIoU | ± | sign *p* | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.4519 | 0.4641 | 0.5942 | 0.6733 | 0.6408 | −0.0172 | 4/28 | 1e−6 | [−0.025, −0.010] |
| disk r=5 | 0.4546 | 0.4726 | 0.5971 | 0.6737 | 0.6445 | −0.0145 | 4/28 | 1e−6 | [−0.022, −0.008] |
| disk r=10 | 0.4591 | 0.4726 | 0.6020 | 0.6738 | 0.6506 | −0.0100 | 4/28 | 1e−6 | [−0.015, −0.006] |
| **disk r=15** *(default)* | 0.4691 | 0.4727 | 0.6131 | 0.6686 | 0.6653 | — | — | — | — |
| disk r=20 | 0.4750 | 0.4812 | 0.6194 | 0.6593 | 0.6779 | +0.0059 | 24/10 | 0.024 | [+0.002, +0.011] |
| disk r=30 | 0.4809 | 0.4916 | 0.6250 | 0.6348 | 0.7029 | +0.0118 | 21/13 | 0.230 | [−0.002, +0.027] |
| disk r=40 | 0.4826 | 0.4869 | 0.6253 | 0.6098 | 0.7294 | +0.0135 | 17/17 | 1.000 | [−0.009, +0.036] |
| disk r=60 | 0.4683 | 0.4812 | 0.6090 | 0.5626 | 0.7687 | −0.0008 | 16/18 | 0.864 | [−0.030, +0.029] |
| relax floor=0.05 | 0.4863 | 0.5047 | 0.6232 | 0.5597 | **0.8172** | +0.0172 | 18/15 | 0.728 | [−0.021, +0.058] |
| **relax floor=0.10** | **0.5101** | 0.5276 | 0.6454 | 0.5951 | 0.8006 | **+0.0410** | 19/14 | 0.487 | [+0.004, +0.080] |
| **relax floor=0.20** | 0.5097 | **0.5299** | **0.6489** | 0.6311 | 0.7579 | **+0.0406** | 22/11 | 0.080 | [+0.014, +0.069] |
| relax floor=0.30 | 0.4909 | 0.4956 | 0.6339 | 0.6521 | 0.7094 | +0.0218 | 22/9 | 0.029 | [+0.008, +0.037] |

### +0.041 IoU — twenty-four times the level-set effect

The relaxation fallback at floor 0.10–0.20 is worth **+0.041 IoU / +0.035 F1** over the
current default. For scale, every other effect measured in this document:

| | ΔIoU |
|---|---|
| **seed fallback: disk r=15 → relax 0.20** | **+0.041** |
| disk fallback vs none | +0.017 |
| claiming rule, centroid → boundary | +0.003 |
| removing the level set | −0.002 |
| `imclose`, in threshold mode | +0.004 |

It is also **not a level-set question at all** — it applies to both modes, and the reduced
pipeline is simply where it became visible, because there the fallback is the output rather
than a starting guess.

### Two independent reasons to trust the direction

**A broad plateau, not a spike.** Floors 0.10 and 0.20 give 0.5101 and 0.5097 — nearly
identical — with 0.05 and 0.30 falling to 0.486 and 0.491 on either side. A single lucky
value would not behave like that.

**Both devices agree.** DRAC22 0.4577 → 0.4792 (+0.022, best at floor 0.20); TUH 0.4864 →
0.5634 (+0.077, best at floor 0.10). The gain is much larger on TUH, but it is present and
positive on both.

### What it actually does, and the honest caveat

It is a **recall** intervention: recall 0.665 → 0.758 at floor 0.20, precision 0.669 →
0.631. Recall is the weaker side on this arm, so unlike the level set this pays into the
right account. The disk-radius rows say the same thing more crudely — r = 15 is simply too
small, and r = 30–40 already buys +0.012 — but relaxation beats every disk because it
recovers the lesion's *shape* rather than a circle.

**The sign tests do not confirm the mean.** Floor 0.10 is 19 better / 14 worse, *p* = 0.49,
despite a +0.041 mean and a CI excluding zero. Only 33 images change at all (the 34 where
the fallback fires), and the distribution is heavy-tailed — large wins, small losses. So
this is a real mean improvement, **not** a uniform one. Floor 0.20 is the better-behaved
choice on that axis: 22/11, *p* = 0.08, the tightest CI, the best F1, and better precision.

**`relaxFloor` is a decision variable swept on the images it is scored on.** The value is
therefore subject to the same leakage caution as `tau`: report the curve, not its maximum.
The plateau is the defence, but a confirmatory run at a pre-registered floor is owed before
the exact figure is quoted. **Pre-register 0.20.**

### It captures half the oracle-`tau` gap

§7 put the oracle per-image `tau` at 0.5468, **+0.078** over the fixed threshold, and called
it the largest headroom in the project. The relaxation fallback is a *legitimate* per-seed
adaptive threshold — it reads only `P`, never ground truth — and it recovers **+0.041, some
53 % of that gap**, at inference, with no retraining.

That is direct evidence for §7's diagnosis: the problem is that one global `tau` is wrong
for images `P` is miscalibrated on, and letting the threshold move locally recovers much of
what a per-image oracle would.

## 9. Confirming `relaxFloor = 0.20` — the selection costs 0.004

§8 left the floor as a sweep maximum. The 68 images cannot be replaced, so re-scoring 0.20
alone would only reproduce 0.5097 bit for bit — which it does, through the production
`maskFromProb` path rather than `run_ablate`'s, and that is all it shows. What can be tested
is whether *choosing* the floor on the scored images inflated the number.
`run_confirm_relax`, off cached `P`, 68 s:

| check | how the floor is chosen | IoU | ΔIoU vs disk r=15 | ± | sign *p* | 95 % CI |
|---|---|---|---|---|---|---|
| pre-registered | fixed 0.20 | 0.5097 | +0.0406 | 22/11 | 0.080 | [+0.014, +0.069] |
| **cross-fitted** | **on 4 folds, scored on the 5th** | **0.5061** | **+0.0370** | 20/13 | 0.296 | **[+0.003, +0.073]** |
| cross-device | on DRAC22, scored on TUH (picks 0.20) | 0.5561 | +0.0698 | | | |
| cross-device | on TUH, scored on DRAC22 (picks 0.15) | 0.4759 | +0.0182 | | | |
| replication | fixed 0.20, 41-image arm, DRAC22-only networks | 0.4727 | +0.0182 | 8/4 | 0.388 | [+0.001, +0.040] |

**The leakage-free estimate is 0.5061, +0.037 over the disk.** Selection optimism is
0.0036. The inner selection picked 0.10 or 0.15 on every fold, never 0.20 — the plateau
again, so the choice among 0.10–0.20 does not matter and the mechanism does. Every fold
gains at the fixed floor (+0.017 to +0.070), both devices gain when the floor is chosen on
the other, and the gain survives on networks that never saw TUH, at the DRAC22 size (+0.018
there, +0.021 on the 68-image networks' DRAC22 subset).

Against the level-set pipeline, reduced + relax is **+0.039 IoU, CI [+0.008, +0.073]** — the
reduced pipeline is now better, not merely no worse. But 31/35 images: the mean is carried
by large wins on the 33 images where the fallback fires, and no sign test reaches
significance. Report it as a mean improvement, not a uniform one.

Still the same 68 images throughout; §9 removes the selection leak, not the need for data
the pipeline has never touched.

## What this means

**The GVF + DRLSE block does not earn its place.** Both domains agree, and the larger one
agrees more strongly:

| | ΔIoU | ΔF1 | sign *p* | 95 % CI |
|---|---|---|---|---|
| 41 DRAC22 images | −0.0057 | −0.0028 | 0.108 | [−0.022, +0.010] |
| **68 combined images** | **−0.0017** | **−0.0005** | **0.389** | **[−0.011, +0.008]** |

Doubling the domain made the estimate *smaller* and the *p*-value *larger*. Against
−0.0005 F1 the block costs five pipeline stages, a compiled mex dependency, eight tuned
parameters, and ~6× the runtime per image (16.9 s → 2.7 s on `458.png`).

Three things sharpen the case beyond the headline:

1. **It pays into the wrong account.** Precision 0.698 → 0.669, recall 0.643 → 0.665. The
   level set buys precision by giving up recall, and recall is the weaker side here.
2. **It is not uniform.** On TUH the reduced pipeline is *better* (0.4864 vs 0.4788), and
   it more than doubles `nv (21)`, the largest lesion in the domain. The mean hides a
   device-dependent split.
3. **Two stages that survive matter far more than the one being cut.** The disk fallback
   is worth −0.016 at *p* = 1.3e−7, an order of magnitude more than the level set, and
   `imclose` becomes ~40× more valuable once the level set is gone.

### What would argue against deleting

- A non-significant delta is not a zero delta; the CI's lower bound is −0.011.
- `641.png` loses −0.172 and `nv (1).png` −0.080 without it. Deletion is a decision about
  the mean, and specific images pay for it.
- Both `phi0` and the seed-claiming logic are entangled with `initPhi`, which also owns the
  disk fallback — the stage that matters most. **Deleting the level set must not delete the
  disks with it.** `cnngvf.seedDisks` was extracted for exactly this reason.

### Where the remaining headroom actually is

The oracle per-image `tau` is **+0.078** over the fixed threshold on 68 images — larger
than every effect in the factorial combined. Both domains point the same way, and it agrees
with UNCLUSTERED.md's standing conclusion that **the probability map is the bottleneck**
and post-processing is exhausted. Effort belongs in `P`: calibration, more data, longer
training — not in the contour method.

### The finding that outranks the one we set out to make

REDUCE.md asked whether the level set earns its place. The answer is no — but the more
useful result is §8: **the stage nobody was looking at is worth twenty-four times the stage
under investigation.** The seed fallback fires for 29.5 % of seeds, its parameter had never
been measured, and replacing a blind 15 px disk with a locally relaxed threshold is worth
+0.041 IoU — while the level set, eight tuned parameters and a mex dependency, is worth
−0.002.

Both belong in the same story: the pipeline's remaining error is concentrated in what
happens when `P` is weak, not in how a contour is refined once `P` is strong.

## Still outstanding

1. **Make `fallbackMode = "relax"`, `relaxFloor = 0.20` the default** — confirmed in §9
   (cross-fitted +0.037). Not yet done. It must not move a pixel in `Post.mode = "levelset"`
   for the frozen arms.
2. **The deletion change itself** — its own commit, after this report is read. Remove
   `buildForce`, `gvfField`, `drlse`, the `Force`/`Gvf`/`Drlse` config blocks and the
   `lib/Fast-Gradient-Vector-Flow` dependency; **keep** `cnngvf.seedDisks`,
   `cfg.Drlse.seedRadius` (rename it), `seedMasks` and the morphology. The frozen arms must
   keep working, so `Post.mode = "levelset"` may need to survive as a compatibility path
   rather than being removed outright.
3. **Update UNCLUSTERED.md** — it still says "No combined-domain number exists yet" and
   lists the centroid claiming rule as an open bug. Both are now resolved.

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

For the 68-image domain, substitute `"DRACTUH_unclustered"` throughout (its `run_cv` is the
full ~15 h; the rest is unchanged).

`cfg.CV.skipTrain` also resumes an interrupted run: it reuses every fold network already
trained and trains only the missing ones. The combined-domain run was interrupted twice and
restarted this way, and every reused fold reproduced its earlier mean **exactly**
(0.3623 / 0.5289 / 0.5361 / 0.4496 across three separate runs) — so a resumed run is not a
different experiment.

**Launch long runs as `matlab -logfile <path> -batch "..."`, never with shell redirection.**
With redirection the log handle belongs to the wrapper shell; when that dies the detached
MATLAB keeps running blind and is killed at its next write — which cost one fold's scoring
here, four hours after the log had frozen. When a run's progress is in doubt, trust the
filesystem (`models/cv/*_fold*.mat`, per-fold `per_image.csv`, `probcache` count), not the
log.

Every score here is tied to the localization CSVs in place on 2026-09-03; their mtimes are
recorded in each cached `P`'s metadata.
