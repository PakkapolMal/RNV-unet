---
title: Refinement Plan — Measured Results
description: Step-by-step record of REFINE.md on the 68-image combined domain, cached P, reduced pipeline + relax.
kind: measurement-record
status: in progress — steps 0–2 done
dataset: DRAC22 + TUH (68)
updated: 2026-09-17
---

# Refinement Plan — Measured Results

What [REFINE.md](REFINE.md) asked for, one section per step. Every number is 68 images
(DRAC22 41, TUH 27), `EvalMode = "image"`, cross-validated through the fold network that held
each image out. Localization CSV mtimes in the cache metadata: DRAC22 `2026-08-31 14:46:10`,
TUH `2026-08-31 23:01:37`.

Machine-readable outputs: `CNN+GVF/result/DRACTUH_unclustered_threshold/refine/`.

## Decisions taken before step 0

REFINE.md disagreed with the code in two places. Both went to the human, who decided:

1. **Relax is per-arm, not a global default.** `initPhi` calls `seedFallback`, so making
   `fallbackMode = "relax"` the default in `defaults.m` would have moved the level-set arm.
   `defaults.m` keeps `"disk"`; `cnngvf.arm` sets `relax` / `0.20` for `DRACTUH_unclustered`
   only. Every runner applies the arm *before* setting `Post.mode`, so the arm cannot test the
   mode itself: `initPhi` now always builds disks, which is what every level-set number on
   record was measured with.
   *Side effect:* a rerun of `run_ablate("DRACTUH_unclustered")` would now score its threshold
   cells with relax rather than the disk recorded in REDUCE_RESULTS §6. Its level-set cells do
   not move.
2. **Paths.** The arm's outputs live in `result/DRACTUH_unclustered_threshold/`, not
   `result/DRACTUH_unclustered/`. Every new script takes `outputDir` and defaults to that folder.

## Deviations from REFINE.md

1. **One runner, `run_refine(step, outputDir)`,** instead of a script per step. Every CPU step
   loads the same 68 `P`, GT masks and seeds.
2. **The leakage test points every GT path at a folder that does not exist** instead of making
   the GT directory unreadable (Windows ACLs), and asserts the output is unchanged.
3. **The trace has no `image` column inside `maskFromProb`,** which does not know the name;
   `run_refine` prepends it.
4. **Size quintiles are rank-based** (`ceil(rank·5/n)`, giving 13/14/13/14/14): `quantile` and
   `prctile` are Statistics Toolbox. `PredGt_Q<k>` is the pooled ratio `ΣPredPx / ΣGtPx` in the
   quintile, not a mean of per-image ratios.
5. **Grid order encodes "most conservative".** `cnngvf.crossFit` takes the first row within
   `tieTol` of the best, so each step lists its grid starting from the feature-off end.

---

## Step 0 — Lock the reference

**Decision:** ADOPT (all verifications pass) → step 1
**Config delta:** `DRACTUH_unclustered` arm: `Post.fallbackMode = "relax"`, `Post.relaxFloor = 0.20`
(threshold mode only; `defaults.m` unchanged)
**Code:** `+cnngvf/arm.m`, `initPhi.m`, `seedFallback.m`, `seedMasks.m`, `maskFromProb.m`,
`writeSummary.m`; new `+cnngvf/crossFit.m`, `pairedStats.m`, `refineScore.m`, `run_refine.m`;
`test_cnngvf.m`

### How relax sets the threshold

**(a) fixed floor, straight to it, in a window.** `+cnngvf/seedFallback.m`, `relaxFallback`,
word for word:

```matlab
    comp = false(size(sub));
    if sub(sr, sc) > floor_
        comp = bwselect(sub > floor_, sc, sr, 8);
    end

    if nnz(comp) >= cfg.Post.MinArea
        D(r0:r1, c0:c1) = D(r0:r1, c0:c1) | comp;
    else
        % P never gives a claimable component near this seed, even at the floor.
        % Keep the recall floor rather than dropping the lesion.
        D = D | cnngvf.seedDisks([h w], seeds(s, :), BW, cfg);
    end
```

`sub` is `P` inside ±`relaxWindow` = 128 px of the seed. The seed's 8-connected component of
`P > relaxFloor` becomes its region; a region under `MinArea` falls back to the disk. So step 4
runs.

### Verification

| check | expected | reproduced | max per-image \|ΔIoU\| |
|---|---|---|---|
| relax 0.20, production `maskFromProb` | 0.5097 / 0.5299 / 0.6489 / 0.6311 / 0.7579 | 0.5097 / 0.5299 / 0.6489 / 0.6311 / 0.7579 | 5e-16 vs `confirm_per_image.csv` |
| cross-fitted floor, `cnngvf.crossFit`, plain max | 0.5061 | 0.5061, chose 0.10 / 0.15 / 0.10 / 0.15 / 0.15 | 5e-16 |
| level set, arm cfg with relax set | 0.4708 | 0.4708 | 5e-16 vs `ablate/per_image.csv` cell 1 |
| `test_cnngvf` | 11 + new | 13 checks pass | |

With the 0.001 tie-break (lowest floor first) the cross-fit chooses the same floors: 0.5061.
Cross-device: choose on DRAC22 → 0.20, TUH 0.5561; choose on TUH → 0.15, DRAC22 0.4759
(matches REDUCE_RESULTS §9).

### REF0

| arm | IoU | median | F1 | prec | rec | pooled IoU | pooled F1 | pooled prec | pooled rec |
|---|---|---|---|---|---|---|---|---|---|
| REF0: reduced + relax 0.20, fixed | 0.5097 | 0.5299 | 0.6489 | 0.6311 | 0.7579 | 0.5896 | 0.7418 | 0.7271 | 0.7570 |
| REF0 cross-fitted floor | 0.5061 | | | | | | | | |

Dataset breakout: DRAC22 0.4792, TUH 0.5561.
Per fold: 0.4124, 0.6004, 0.5490, 0.4832, 0.5011.

| quintile | n | GT px | IoU | Pred/GT (pooled) |
|---|---|---|---|---|
| Q1 | 13 | 601 – 4 052 | 0.4272 | 2.129 |
| Q2 | 14 | 4 176 – 10 332 | 0.4582 | 1.691 |
| Q3 | 13 | 10 645 – 22 281 | 0.4900 | 1.467 |
| Q4 | 14 | 23 655 – 48 105 | 0.5897 | 1.323 |
| Q5 | 14 | 50 314 – 432 533 | 0.5762 | 0.890 |

Frozen to `refine/quintiles.mat` (and `.csv`).

Pooled precision (0.727) is well above mean precision (0.631), and Q1–Q3 over-predict by
1.5–2.1×. The precision problem is concentrated on small lesions, where a few hundred FP pixels
cost a large share of per-image IoU. Under relax, Q5 Pred/GT is 0.89 (pooled), not the ~0.51
REFINE.md step 5 quotes from earlier diagnostics. Those did not record the pipeline or the ratio
definition, so the two are not directly comparable. Step 5 measures it again.

### Trace (874 seeds)

| | seeds |
|---|---|
| fired (`P > tau` at the seed) | 616 |
| relaxed | 218 |
| disk (not fired, relax found < `MinArea`) | 40 |
| claimed nothing | 0 |
| `claimDist > 0` | 7 |
| `0 < claimDist ≤ 20` | 0 |
| `20 < claimDist < Inf` | **7, on 5 images** |

218 + 40 = 258 fallback seeds on 34 images, matching REDUCE_RESULTS §8.

All seven far claimers are **fired** seeds whose own component is below `MinArea`. The fallback
skips fired seeds, so they claim the nearest qualifying component at any distance. Relaxed and
disk seeds always stand inside their own region:

| image | seed | claimDist | claimed area |
|---|---|---|---|
| 1236.png | 8 | 31.4 | 14 226 |
| 125.png | 5 | 98.5 | 6 762 |
| 299.png | 2 | 110.6 | 1 301 |
| 299.png | 3 | 585.7 | 1 301 |
| 299.png | 4 | 672.1 | 1 301 |
| 429.png | 5 | 38.0 | 4 031 |
| 498.png | 2 | 42.5 | 11 894 |

The mechanism is present, so step 1 is not a SKIP. Its ceiling is small: 7 seeds, 5 images.

### Run summaries

`writeSummary` recorded **no** `Post` settings before this step: not `Post.mode`, `tau`,
`fallbackMode`, `relaxFloor` or `imclose`. It now writes the whole `Post` block, flattening
nested structs and naming the strel, so every later flag is recorded automatically.

**Next step:** 1.

---

## Step 1 — Cap the claim distance

**Prediction written before running:** ΔIoU 0 to +0.010, all precision. Relax may already
absorb most of it, because a missed seed builds its own relaxed region instead of claiming a
far component. *(Added after step 0, before any step-1 scoring: the ceiling is small, since
only 7 seeds on 5 images claim beyond 20 px, three of them on `299.png`, which has no seed
near its lesion.)*

**Decision:** REJECT (the effect is exactly zero; the flag stays in the code, default `Inf`)
**Config delta:** `Post.claimMaxDist` ∈ {Inf, 128, 64, 32, 20, 10, 0}; CUR stays `Inf`
**Code:** `+cnngvf/defaults.m`, `seedMasks.m` (5th argument `maxDist`), `maskFromProb.m`,
`run_refine.m` (step 1, plus the shared report / gate helpers), `test_cnngvf.m`

Flag off (`Inf`) reproduces step-0 REF0: max per-image |ΔIoU| 5e-16. The level set never
receives the flag (`initPhi`, `segmentImage` call `seedMasks` without it); a test asserts it.

| arm | IoU | median | F1 | prec | rec | pooled IoU | pooled rec | ΔIoU vs CUR | ± | sign p | Wilcoxon p | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUR = REF0 (`Inf`) | 0.5097 | 0.5299 | 0.6489 | 0.6311 | 0.7579 | 0.5896 | 0.7570 | — | | | | |
| pre-registered 20 | 0.5097 | 0.5299 | 0.6489 | 0.6311 | 0.7579 | 0.5896 | 0.7570 | +0.0000 | 0/0 | 1 | 1 | [0, 0] |
| **cross-fitted** | **0.5097** | 0.5299 | 0.6489 | 0.6311 | 0.7579 | 0.5896 | 0.7570 | **+0.0000** | 0/0 | 1 | 1 | [0, 0] |

The comparison is against REF0 and CUR together, since they are the same configuration here.

Cross-fitted chosen parameters per fold: `Inf` ×5. Every grid row ties exactly, and the tie-break
picks the feature-off end.
Cross-device: DRAC22→TUH `Inf`, +0.0000; TUH→DRAC22 `Inf`, +0.0000.
Surface: every cap from 0 to `Inf` scores 0.5097 / prec 0.6311 / rec 0.7579 / pooled IoU 0.5896.
Size quintiles and dataset breakout: identical to REF0.
Movers: none.

**Why it is exactly zero.** Each of the 7 far claims lands on a component that at least one
other seed claims from inside it, so capping removes a duplicate claim and leaves the union
unchanged:

| image | far seed | claimDist | component also claimed from inside by seeds |
|---|---|---|---|
| 1236.png | 8 | 31.4 | 9–14 |
| 125.png | 5 | 98.5 | 1, 3 |
| 299.png | 2, 3, 4 | 110.6, 585.7, 672.1 | 1 |
| 429.png | 5 | 38.0 | 1, 2 |
| 498.png | 2 | 42.5 | 1, 5, 6 |

Diagnostic (GT, `diag/step1_disconnected.csv`): of the 7 seeds a cap of 20 disconnects, 4 are
stray (`125` seed 5, `299` seeds 2–4, all > 100 px from `FilledGT`) and 3 are true seeds within
0 px of a lesion whose own `P` component fell under `MinArea`. Caps of 64 and 128 disconnect
only stray seeds (4 and 2), but it makes no difference to the output.

**What it means for step 3.** Stray seeds do not add false positives by claiming far
components. The FP a stray seed causes must come from the region it builds itself: a fired
component, a relaxed region or a disk. That is what a seed gate removes and a claim cap cannot.

Gate evaluation: rule 1 ✗ (+0.0000 < +0.003), rule 2 ✓ (CI lower bound 0.0000), rule 3 ✓
(pooled IoU unchanged), rule 4 ✓ (no image moves), rule 5 ✓ (both directions +0.0000). Not
ambiguous: the delta is below the +0.001…+0.003 band. → **REJECT**.

**Next step:** 2.

---

## Step 2 — Seed features and the separability check

**Decision:** SKIP step 3 → step 4 (best in-budget catch rate 25%, below the 30% floor)
**Prediction written before running:** `support` and `peakP` separate stray seeds from true ones
with AUC ≥ 0.80. Stray seeds are more than 100 px from any lesion, and true lesions attract many
detections.
**Config delta:** none (diagnostic only)
**Code:** new `+cnngvf/seedFeatures.m` (P, seeds, cfg only; a test asserts it ignores GT paths);
`run_refine.m` step 2; `test_cnngvf.m`

Counts check against REDUCE_RESULTS §10: **874 seeds, 842 near, 32 stray, on 23 images**. They match.

### AUC per feature (stray vs true, oriented ≥ 0.5)

| feature | stray seeds have | all 68 | DRAC22 | TUH |
|---|---|---|---|---|
| fired | higher | 0.588 | 0.581 | 0.459 |
| peakP_r16 | higher | 0.610 | 0.585 | 0.584 |
| peakP_r32 | higher | 0.571 | 0.547 | 0.580 |
| meanP_r16 | higher | 0.592 | 0.549 | 0.551 |
| support_R64 | lower | **0.914** | 0.880 | 0.875 |
| support_R100 | lower | **0.939** | 0.935 | 0.864 |
| support_R150 | lower | **0.910** | 0.930 | 0.735 |
| nnDist | higher | **0.929** | 0.943 | 0.789 |
| regionArea | lower | 0.834 | 0.849 | 0.548 |
| regionMeanP | higher | 0.734 | 0.665 | 0.564 |
| distToFiredComp | higher | 0.563 | 0.580 | 0.486 |
| imageSeedCount | lower | 0.757 | 0.738 | 0.456 |

**The prediction is half wrong.** Support does separate (AUC 0.91–0.94). `peakP` does not
(0.57–0.61), and its direction is the opposite of what the plan assumed: **stray seeds sit on
high P**. 29 of 32 stray seeds fired, most with `peakP_r32` = 1.00. A stray seed is a confident
false detection with its own confident CNN response, not a weak one.

### Candidate rules

Budget: ≤ 16 true seeds dropped (2% of 842) and no single-seed image orphaned. Best in-budget
rows (full grid in `step2/rules.csv`):

| rule | k | p0 | d0 | stray caught | true dropped | images losing every seed | oracle-weighted ΔIoU |
|---|---|---|---|---|---|---|---|
| `support_R100<=k & peakP_r32<p0` | 0 | 0.9 | | **8/32** | 10 | 1 | +0.0029 |
| `support_R100<=k & peakP_r32<p0` | 1 | 0.7 | | 5/32 | 10 | 0 | +0.0004 |
| `support_R100<=k & peakP_r32<p0` | 0 | 0.7 | | 3/32 | 3 | 0 | +0.0004 |
| `~fired & support_R100<=k & peakP_r32<p0` | 1 | 0.7 | | 2/32 | 10 | 0 | +0.0002 |
| `~fired & distToFiredComp>d0` | | | any | ≤ 1 | | | ≤ +0.0002 |

Oracle ceiling (every stray seed dropped, cached `P`, relax 0.20): **+0.0312**.
Every rule containing `~fired` or `peakP < p0` works against the signal, because stray seeds
are fired and high-P.

### Diagnostic beyond the pre-registered forms (not a result, not used for the gate)

Support alone, from `diag/step2_seed_features.csv`:

| rule | stray caught | true dropped (images) | images losing every seed | single-seed images orphaned |
|---|---|---|---|---|
| `support_R100<=0` (= `nnDist>100`) | 27/32 | 38 (28) | 14 | 7 |
| `support_R150<=0` (= `nnDist>150`) | 22/32 | 27 (22) | 12 | 7 |
| `nnDist>200` | 19/32 | 23 (20) | 11 | 7 |
| `nnDist>250` | 16/32 | 22 (19) | 10 | 7 |
| `nnDist>300` | 10/32 | 21 (18) | 10 | 7 |

Even support alone does not fit the budget. An isolated seed is either a stray detection or the
only detection on a small lesion, and support cannot tell those apart. The ~20 isolated true
seeds are exactly what a gate would delete, and each is the only seed on its lesion, so dropping
it removes that lesion. `keepLastSeed` would protect the 7 single-seed images, not the rest.

### The stray seeds

Sorted by the per-image oracle gain (full table in `diag/step2_stray_seeds.csv`). The five images
carrying most of the gain:

| image | seed | gtDist | fired | peakP_r32 | support_R100 | nnDist | regionArea | seeds in image | oracle gain |
|---|---|---|---|---|---|---|---|---|---|
| 641.png | 3 | 843 | 1 | 1.00 | 0 | 859 | 9 590 | 3 | +0.441 |
| 428.png | 3 | 300 | 1 | 1.00 | 0 | 309 | 6 942 | 3 | +0.408 |
| nv (25).png | 6 | 529 | 1 | 1.00 | 0 | 536 | 25 733 | 6 | +0.212 |
| 419.png | 1 | 603 | 1 | 1.00 | 0 | 611 | 3 948 | 4 | +0.186 |
| 419.png | 4 | 116 | 1 | 0.72 | 0 | 156 | 954 | 4 | +0.186 |
| 576.png | 1 | 457 | 1 | 1.00 | 0 | 507 | 6 878 | 5 | +0.161 |

Each is a lone, confident detection hundreds of pixels from any lesion, and the CNN responds
fully to it. The 3 stray seeds on `nv (21).png` are the exception: 30–46 px off a 432k px lesion,
with support 1–5.

### Gate

Best in-budget catch rate **25%** (8/32), below the 30% skip threshold → **SKIP step 3, go to
step 4.** Not ambiguous under the plan's own bands.

**Flagged for the human:**
- The separating feature exists (support / `nnDist`, AUC 0.93), but none of the pre-registered
  rule forms uses it the right way, and even support alone costs ~20 true seeds at 50% catch.
- Whether a support-based gate would still raise IoU is an **IoU** question the plan's
  **count** gate does not answer. Dropping a true seed costs nothing when another seed covers
  its lesion, but an isolated true seed covers its lesion alone.
- Running that would mean choosing a new rule form after seeing these numbers, so it was not
  run.

**Next step:** 4.
