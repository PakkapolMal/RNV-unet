---
title: Refinement Plan — Measured Results
description: Step-by-step record of REFINE.md on the 68-image combined domain, cached P, reduced pipeline + relax.
kind: measurement-record
status: in progress — step 0 done
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
