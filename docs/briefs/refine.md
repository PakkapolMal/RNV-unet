---
title: Refinement Plan — Inference-Only Gains on Reduced + Relax 0.20
description: Staged build specs for post-processing and inference changes on cached P. Each step is built, evaluated, recorded, and gated before the next one starts.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/CNN+GVF, MATLAB R2026a
depends-on: REDUCE.md, REDUCE_RESULTS.md, UNCLUSTERED.md
status: done — executed 2026-09-17
---

# Refinement Plan — Inference-Only Gains on Reduced + Relax 0.20

> **Executed brief.** Kept as written. Outcome:
> [../results/2026-09-17-refine.md](../results/2026-09-17-refine.md); code at tag
> `archive/reduce-refine`.

Read [UNCLUSTERED.md](../SPEC.md) and [REDUCE_RESULTS.md](../results/2026-09-16-reduce.md) first.
Where this document disagrees with either, stop and say so. Do not resolve it yourself.

## Objective

Raise cross-validated IoU and F1 on the 68-image combined domain (`DRACTUH_unclustered`)
without retraining. This plan starts from the current best configuration:

| reference (68 images, 5-fold CV) | IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| reduced + relax floor 0.20, fixed | 0.5097 | 0.5299 | 0.6489 | 0.6311 | 0.7579 |
| same, floor cross-fitted | 0.5061 | — | — | — | — |

Precision is now the weak side, 0.13 below recall. The largest known headroom that needs
no retraining is the oracle seed filter at +0.032 IoU, all of it precision. Re-inferring
`P` added nothing there, so most of this plan runs on cached `P` and takes seconds per
variant. Only steps 6 and 7 need the GPU.

## How to work through this plan

Every step has the same shape. Do them in order, and do not start a step until the
previous step's record is written.

```
BUILD     implement the spec, behind a flag whose default reproduces the current reference
VERIFY    flag off → current reference reproduced exactly; tests pass
EVALUATE  run the step's evaluation on cached P
RECORD    append the step's section to REFINE_RESULTS.md (template below)
DECIDE    apply the step's gate → ADOPT / REJECT / SKIP, and set the next step
COMMIT    one commit per step: code + results + the decision
```

A step that is rejected still gets recorded, and its flag stays in the code, off.
Rejected steps are findings.

### Stop and ask the human when

- The flag-off path does not reproduce the current reference exactly.
- The code does not match what a step assumes, for example relax is implemented differently
  from both variants named in step 0.
- A leakage guard fires.
- A step seems to require retraining, a new input channel, or a loss change.
- A gate result lands inside its "ambiguous" band.
- Any number moves by more than 0.05 IoU in either direction. That is more likely a bug
  than a finding.

## Invariants — binding for every step

### Untouched

- **Training:** network, loss, optimizer, epochs, patches, channels, fold networks
  `models/cv/DRACTUH_unclustered_fold{1..5}.mat`. No retraining anywhere in this plan.
- **Evaluation:** `EvalMode = "image"`. 68 rows, 68 unique. Every fold scores exactly its
  held-out set. `299.png` and `445.png` still count.
- **Frozen arms:** `Post.mode = "levelset"` must stay byte-identical. Every new flag is
  inert in level-set mode. Assert it.
- **Seeds:** `[cx cy]` = (column, row), read positionally. A transposed seed fails silently.

### Leakage rules

1. **Production code never reads ground truth.** `maskFromProb`, `seedMasks` and every
   new function added in this plan take `P`, seeds and `cfg` only. Add a test that runs
   them with the GT directory made unreadable.
2. **Diagnostic GT labels live in `result/<arm>/refine/diag/` only.** They are for
   go/no-go checks and are never read by production code.
3. **Every tuned parameter is cross-fitted** (protocol below). The headline number for a
   step is the cross-fitted one. A grid maximum is reported as a surface, never as a result.
4. **Every cached `P` passes the existing fold/held-out check on read.** Pseudo-seed and
   TTA caches carry the same metadata as the original cache.

### Reporting rules

State the domain (68 images) with every number. Cross-validated numbers only. Tie every
score to the localization CSV mtimes in the cache metadata.

## Shared infrastructure — build once, in step 0

### Metrics

Extend the scorer so every evaluation reports all of these:

| metric | definition |
|---|---|
| mean / median IoU, mean F1, precision, recall | per image, then averaged (existing) |
| **pooled IoU, pooled F1, pooled precision, pooled recall** | TP, FP and FN summed over all 68 images, then computed once |
| **size-quintile breakout** | per-image IoU and Pred/GT ratio by GT-area quintile |
| dataset breakout | DRAC22 (41) / TUH (27) |
| per-fold | mean IoU per fold |

Compute the size-quintile edges once in step 0 from GT areas and freeze them to
`result/<arm>/refine/quintiles.mat`. Every later step reuses that file.

### Paired statistics

Against a named comparator, on per-image IoU deltas:

- exact two-sided sign test (ties excluded; report better/worse counts)
- Wilcoxon signed-rank, normal approximation with tie correction (no Statistics Toolbox)
- bootstrap 95% CI on the mean delta: 10 000 resamples, fixed RNG seed `rng(20260917)`
- the five largest gains and five largest regressions, by image name

Every step reports against two comparators: **REF0**, the step-0 reference, and **CUR**,
the configuration current at the start of the step. They are the same until something is
adopted.

### Cross-fitting — `cnngvf.crossFit`

Generalize the logic in `run_confirm_relax.m`:

```
function out = crossFit(grid, scoreFn, folds)
% grid     : table, one row per parameter setting
% scoreFn  : @(row) -> per-image IoU vector over all 68 images, from cached P
% folds    : the existing cvFolds assignment
% for k = 1..5: choose the grid row maximizing mean IoU over images NOT in fold k,
%               then score that row on fold k
% out.perImage  : 68 cross-fitted IoUs (and F1, prec, rec)
% out.chosen    : 5 chosen rows, one per fold
% out.surface   : every grid row's full-domain metrics (reported as a surface only)
```

**Tie-break:** within 0.001 IoU, pick the most conservative row, meaning the one closest
to the feature being off. Also run the **cross-device check**: choose on DRAC22, score TUH,
and the reverse.

### The adoption rule

Unless a step overrides it, adopt a change when all of these hold:

1. cross-fitted mean ΔIoU vs CUR **≥ +0.003**
2. bootstrap CI lower bound **> −0.005**
3. pooled IoU does not fall
4. no single image regresses by more than **0.10**
5. the cross-device check is non-negative in both directions

**Ambiguous band:** rules 1–2 pass but 4 or 5 fails, or the delta sits in +0.001…+0.003.
Record it and stop for a human decision.

With 68 images and gains concentrated on a few of them, sign tests often will not reach
significance. The rule deliberately does not require it. Report the tests anyway.

## Results file

Create `docs/REFINE_RESULTS.md` in step 0. Append one section per step, always with this
skeleton:

```markdown
## Step N — <name>

**Decision:** ADOPT | REJECT | SKIP | AMBIGUOUS (awaiting human)
**Prediction written before running:** <copied from this plan>
**Config delta:** <flag = value>
**Code:** <files touched>, commit <hash>

| arm | IoU | median | F1 | prec | rec | pooled IoU | pooled rec | ΔIoU vs CUR | ± | sign p | Wilcoxon p | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Cross-fitted chosen parameters per fold: ...
Cross-device: DRAC22→TUH ..., TUH→DRAC22 ...
Size quintiles (IoU, Pred/GT): Q1 ... Q5
Dataset breakout: DRAC22 ..., TUH ...
Movers: better ..., worse ...
Step-specific diagnostics: ...
Gate evaluation: rule 1 ✓/✗ (value), rule 2 ..., ...
Next step: N+1 | skip to M, because ...
```

Machine-readable outputs go to `result/DRACTUH_unclustered/refine/stepN/`: `per_image.csv`
for every arm, `summary.csv`, `stats.csv`, and step-specific tables.

---

# Step 0 — Lock the reference and instrument it

### Build

1. **Read `+cnngvf/maskFromProb.m` and record how relax sets the threshold for a seed that
   never reaches `P > tau`.** Record it word for word in the results file. It is either:
   - **(a) fixed floor:** the threshold drops from `tau` toward `relaxFloor`, stepwise or
     straight to it, or
   - **(b) peak-relative:** the threshold depends on local peak `P`, floored at `relaxFloor`.
   - If it is neither, stop and ask.
2. **Make relax the default** (outstanding item 1 in REDUCE_RESULTS.md):
   `cfg.Post.fallbackMode = "relax"`, `cfg.Post.relaxFloor = 0.20`, threshold mode only.
   Level-set mode must not move a pixel. Assert it.
3. Build the shared infrastructure: pooled metrics, the quintiles file, paired stats,
   `cnngvf.crossFit`, and the results file.
4. **Per-seed trace.** `maskFromProb` optionally returns a `trace` table, one row per seed:
   `image, seedIdx, cx, cy, fired, relaxThrUsed, claimedCompId, claimedCompArea,
   claimDist`. `claimDist` is 0 if the seed is inside its component, otherwise the distance
   to the component boundary, and `Inf` if it claimed nothing. This table feeds steps 1–3.
5. Record whether each run summary writes `Post.mode`, `tau`, `fallbackMode`, `relaxFloor`,
   `imclose` and every new flag. Add any that are missing. This was an open action item.

### Verify

- `run_ablate` / `maskFromProb` on the `DRACTUH_unclustered` cache reproduces **0.5097 /
  0.5299 / 0.6489 / 0.6311 / 0.7579** exactly.
- Recomputing the cross-fit through `cnngvf.crossFit` reproduces **0.5061**.
- Level-set mode reproduces **0.4708** exactly.
- `test_cnngvf` passes, 11 checks plus the new ones.

### Evaluate and record

No comparison. Record REF0: the full metric set, pooled metrics, quintile table, dataset
breakout, and the relax mechanism from build item 1. Also record trace summary counts:
seeds that fired, seeds that relaxed, seeds that claimed nothing, and seeds with
`claimDist > 0`.

### Gate

All verifications pass → step 1. Any mismatch → stop.

---

# Step 1 — Cap the claim distance

**Hypothesis.** A seed outside every component claims the nearest one at unlimited
distance, so a stray seed licenses a false-positive component. Training uses a 20 px
tolerance and inference does not.

**Prediction:** ΔIoU 0 to +0.010, all precision. Relax may already absorb most of it,
because a missed seed builds its own relaxed region instead of claiming a far component.

### Build

- `cfg.Post.claimMaxDist = Inf`. Default `Inf` reproduces REF0.
- In `seedMasks`, a seed may claim a component only if `claimDist <= claimMaxDist`.
  Otherwise it claims nothing.
- The cap must apply the same way to fired seeds and to relaxed regions. Confirm from the
  step-0 trace where relaxed seeds get their component.
- Tests: `Inf` is byte-identical to REF0; a synthetic image with one component and one
  seed 50 px away claims at `Inf` and does not claim at 20.

### Evaluate

- First, from the step-0 trace, count seeds with `20 < claimDist < Inf` and the images
  they sit on. **If the count is 0, record SKIP (mechanism absent) and go to step 2.**
- Pre-registered arm: `claimMaxDist = 20`.
- Surface: `{0, 10, 20, 32, 64, 128, Inf}`.
- Cross-fitted arm over the same grid.
- Diagnostic (GT, `diag/` only): of the seeds the cap disconnects, how many are stray
  (more than 20 px from `FilledGT`)?

### Gate

Adoption rule, applied to the cross-fitted arm. If adopted, CUR becomes the cross-fitted
setting. If the pre-registered 20 and the cross-fitted choice disagree, adopt 20 when it is
within 0.002 of the cross-fitted IoU, because it matches training. → step 2.

---

# Step 2 — Seed features and the separability check

This step builds no production behaviour. It decides whether step 3 is worth building.

**Prediction:** `support` and `peakP` separate stray seeds from true ones with AUC ≥ 0.80.
Stray seeds are more than 100 px from any lesion, and true lesions attract many detections.

### Build

`+cnngvf/seedFeatures.m`, a production-safe function that uses `P`, seeds and `cfg` only:

```
function F = seedFeatures(P, seeds, cfg)
% one row per seed
```

| column | definition |
|---|---|
| `fired` | `P(cy,cx) > tau` (same trigger as the fallback) |
| `peakP_r16`, `peakP_r32` | max `P` in a disk of radius 16 / 32 px |
| `meanP_r16` | mean `P` in a disk of radius 16 px |
| `support_R64`, `support_R100`, `support_R150` | number of *other* seeds within R px |
| `nnDist` | distance to the nearest other seed |
| `regionArea`, `regionMeanP` | area and mean `P` of the component this seed claims (after step 1) |
| `distToFiredComp` | distance to the nearest component claimed by a fired seed |
| `imageSeedCount` | seeds in this image |

Diagnostic script `run_seed_diag.m` (GT allowed; writes to `diag/` only):

- label `stray = distance(seed, FilledGT) > seedTolerance (20)`
- check the counts against REDUCE_RESULTS §10: **874 seeds, 842 near, 32 stray, on 23
  images.** A mismatch means stop.
- AUC per feature, overall and per dataset. Use a rank-based AUC written by hand; there is
  no Statistics Toolbox. Orient each feature so AUC ≥ 0.5 and record the direction.
- For every candidate rule below, over a grid: stray seeds caught, true seeds dropped, and
  images whose *only* seed would be dropped.
  - `~fired & support_R100 <= k & peakP_r32 < p0`
  - `support_R100 <= k & peakP_r32 < p0`
  - `~fired & distToFiredComp > d0`
- Estimate the oracle-weighted capture: sum the per-image IoU gain the §10 oracle got on
  each image, weighted by the fraction of that image's stray seeds each rule catches. Treat
  it as a rough ceiling for step 3.

### Record

AUC table, rule table, and the list of stray seeds with feature values, including the five
images carrying most of the oracle gain (`641`, `428`, `nv (25)`, `419`, `576`).

### Gate

- **Go to step 3** if some rule catches ≥ 50% of stray seeds while dropping ≤ 2% of true
  seeds (≤ 17) and orphans no single-seed image.
- **Skip step 3 and go to step 4** if no rule catches ≥ 30% of stray seeds within that
  budget.
- **Ambiguous** (30–50%): record and ask.

---

# Step 3 — P-based seed gate

**Prediction:** cross-fitted ΔIoU +0.010 to +0.020. Precision up about 0.02–0.04, recall
flat or down ≤ 0.005. Gains land mostly on the five oracle images, and the sign test
probably misses significance.

### Build

```matlab
cfg.Post.seedGate.enabled      = false;   % default reproduces CUR
cfg.Post.seedGate.rule         = "";      % the rule form chosen in step 2
cfg.Post.seedGate.k            = 0;
cfg.Post.seedGate.p0           = 0.10;
cfg.Post.seedGate.supportR     = 100;
cfg.Post.seedGate.peakR        = 32;
cfg.Post.seedGate.keepLastSeed = true;    % never drop an image's last surviving seed
```

- The gate runs **inside `maskFromProb`, before fallback and claiming**, using
  `seedFeatures`. A dropped seed neither relaxes nor claims.
- Compute `support` over the **original** seed set, not the survivors, so the result does
  not depend on drop order.
- Dropped seeds are logged in the trace with the reason. They are never deleted from the
  seed list that `run_eval` recovers from filenames: write an empty per-seed mask for them,
  so the `perSeed` contract still holds with one entry per seed.
- Tests: disabled is byte-identical to CUR; `keepLastSeed` holds on a synthetic
  single-seed image; the gate never reads GT.

### Evaluate

- Grid: only the rule form chosen in step 2, `k ∈ {0,1,2}`,
  `p0 ∈ {0.05, 0.10, 0.20, 0.30, 0.50}`.
- Cross-fitted arm (headline), grid surface, cross-device check.
- Diagnostic (`diag/`): stray and true seeds dropped per fold at the chosen setting.
- Precision/recall movement, pooled metrics, and the size-quintile table. The gate should
  not touch Q5; check it.

### Gate

Adoption rule, plus one step-specific condition: **recall must not fall by more than
0.010.** A precision-only intervention that costs recall is dropping true seeds. → step 4.

---

# Step 4 — Relax guard

**Run only if step 0 recorded relax as (a) fixed floor.** If it is (b) peak-relative,
record SKIP with the reason and go to step 5.

**Hypothesis.** A weak seed with a fixed floor can flood a large low-`P` region. Relax cost
precision 0.669 → 0.631, and at floor 0.05 precision fell to 0.560.

**Prediction:** ΔIoU 0 to +0.010, precision up, and a smaller Pred/GT in Q1–Q2.

### Build

```matlab
cfg.Post.relaxAlpha = 0;     % 0 reproduces CUR exactly
```

For a relaxed seed: `thr = max(relaxFloor, relaxAlpha * peakP_r32)`. Everything else in
relax is unchanged. Test that `relaxAlpha = 0` is byte-identical to CUR.

### Evaluate

- Grid: `relaxAlpha ∈ {0, 0.3, 0.5, 0.7, 0.9}` × `relaxFloor ∈ {0.10, 0.15, 0.20}`.
  The floor is re-crossed because the guard shifts where the plateau sits.
- Cross-fitted arm (headline), surface, cross-device check.
- For relaxed seeds only: median relaxed-region area and mean region `P`, `relaxAlpha = 0`
  vs the chosen setting.

### Gate

Adoption rule. → step 5.

---

# Step 5 — Where the large-lesion pixels are lost

A diagnostic, CPU only. It decides whether step 7 is worth building.

**Background.** Earlier diagnostics put largest-quintile Pred/GT at about 0.51, and the ten
largest lesions hold about 65% of GT pixels. Seedless tiling failed. Before extending
patches, find out whether the missing pixels were ever inside a patch.

**Prediction:** more than 40% of Q5 false negatives lie outside every patch window.

### Build

`run_coverage_diag.m` (GT allowed; writes to `diag/` only), on the CUR configuration:

- Reconstruct the patch windows from the inference seeds and `Patch.size = 512`, with the
  same centring and clipping that `inferProbMap` uses. Read that code; do not re-derive it.
- Also reconstruct the effective window mask: where the raised-cosine weight is above 0.
- Classify every false-negative pixel into exactly one bucket:

| bucket | condition |
|---|---|
| `outside` | outside every patch window |
| `lowP` | inside a window, `P < relaxFloor` |
| `midP` | inside, `relaxFloor ≤ P < tau` |
| `highP_dropped` | `P ≥ tau`, lost to claiming, `minArea`, or the seed gate |

- Report the bucket fractions by size quintile and for the ten largest GT images.
- Report the mean `P` inside GT, by quintile. This separates "the network responds weakly
  on big lesions" from "it never saw them".
- Write an overlay per Q5 image: GT, prediction, window union, and FN coloured by bucket,
  to `diag/coverage/`.

### Record

Bucket table, per-image table for Q5, and the overlays list.

### Gate

- `outside` ≥ 40% of Q5 FN pixels → step 7 enabled.
- `outside` < 20% → step 7 **skipped**. Record "large-lesion loss is model-side (low `P`
  inside the window); the lever is training" and flag it for the human.
- Between 20% and 40% → ambiguous. Record and ask.
- Either way, go to step 6 next.

---

# Step 6 — Test-time augmentation (GPU, no retraining)

**Prediction:** ΔIoU 0 to +0.015. `P` becomes smoother and less extreme, so the best
`tau` / `relaxFloor` may shift. That is why they are re-fitted here.

### Build

- `cfg.Infer.tta = "none"`, alternatively `"d4"`: the 8 dihedral transforms, meaning
  identity, 3 rotations, and 4 flipped variants.
- Inside `inferProbMap`, per patch: transform **both channels together** (intensity and the
  seed heat-map) → predict → inverse-transform the foreground probability → average the 8
  results → then the existing raised-cosine stitch. Average probabilities, not logits or
  binary masks.
- Write to a **new cache**, `result/<arm>/probcache_tta/`, with full metadata plus
  `tta = "d4"`. Never overwrite the original cache.
- Each image uses its held-out fold network. The leakage guard runs on read, as before.

### Tests (all must pass before any GPU run)

1. `"none"` reproduces the original cached `P` for `458.png` within `single` equality.
2. For each transform, the argmax of the transformed heat-map equals the transformed seed
   coordinate. This catches `[cx cy]` transposition, which is otherwise silent.
3. `inverse(transform(X)) == X` for all 8, on a non-symmetric test array.
4. Identity-only TTA (a 1-element group) reproduces `"none"` exactly.

### Evaluate

Score the **CUR post-processing chain** (all adopted steps) on both caches. **Re-fit every
tuned parameter on each cache independently by cross-fitting**: `relaxFloor`, and if
adopted `claimMaxDist`, the gate's `k`/`p0`, and `relaxAlpha`. Otherwise the comparison
pits a tuned arm against an untuned one.

- Headline: cross-fitted TTA vs cross-fitted CUR, paired.
- `tau` sensitivity curve on the TTA cache, 0.30–0.65, reported as a curve.
- Re-run the step-2 AUC table on TTA `P`, plus one new feature: per-seed TTA variance of
  `P` in a 16 px disk. That needs `inferProbMap` to optionally cache the per-transform
  variance, as `single`, in the same file.

### Gate

Adoption rule. If adopted, the TTA cache becomes the cache for step 7 and for the final
confirmation. If the new variance feature raises the best step-2 rule's catch rate by
≥ 15 points, record it and ask whether to re-run step 3 with it. Do not re-run step 3 on
your own. → step 7, or step 8 if step 5 skipped step 7.

---

# Step 7 — Pseudo-seed patch extension (GPU, only if step 5 enabled it)

**Hypothesis.** Large lesions are clipped where the predicted mask runs into a patch-window
edge that is not the frame edge. Centring a new patch inside the mask at that point extends
coverage while staying in distribution, because training seeds also lie inside lesions.

**Prediction:** pooled recall and Q5 IoU clearly up; mean IoU +0 to +0.02, since
per-image means weight the few large lesions lightly.

### Build

```matlab
cfg.Infer.pseudo.enabled     = false;
cfg.Infer.pseudo.maxIter     = 2;     % extension rounds
cfg.Infer.pseudo.edgeBand    = 16;    % px from a window edge that counts as "clipped"
cfg.Infer.pseudo.inset       = 32;    % pseudo-seed placed this far inside the mask
cfg.Infer.pseudo.minSpacing  = 256;   % px between pseudo-seeds and existing seeds
cfg.Infer.pseudo.maxPerImage = 20;
```

Per image, per round:

1. Take the current final mask (from the CUR chain).
2. **Clipped points:** mask pixels within `edgeBand` of the union-of-windows boundary, and
   more than `edgeBand` from the frame border.
3. Move each clipped point `inset` px inward along the inverse distance-transform gradient
   of the mask, and keep only points still inside the mask.
4. Farthest-point sample those points with `minSpacing` from all seeds, including earlier
   pseudo-seeds. Cap at `maxPerImage`.
5. Re-infer **only the new patches** with the image's held-out fold network, then
   re-stitch. Check in `inferProbMap` how per-patch heat-maps are built; each pseudo-seed's
   patch gets a heat-map exactly as a real seed's would.
6. Re-run the CUR post-processing chain on the new `P`.
7. Stop early if no new pseudo-seeds were placed.

**Rule: pseudo-seeds are used for patches only and never claim.** Claiming and the seed
gate use the original seeds, so pseudo-seeds cannot license new false-positive components.
Assert it in the trace.

Write to `result/<arm>/probcache_pseudo[_tta]/`, with the pseudo-seed coordinates and round
numbers in the metadata.

### Tests

- `enabled = false` is byte-identical to CUR.
- On a synthetic long bar lesion with one seed, round 1 places a pseudo-seed near the
  clipped end and recall rises.
- Pseudo-seeds never appear in claiming output.

### Evaluate

- Headline for this step: **pooled IoU and pooled recall**, together with cross-fitted mean
  IoU vs CUR, paired.
- Size-quintile table: Pred/GT and IoU by quintile, before and after.
- Re-run the step-5 buckets. The `outside` fraction for Q5 must fall; if it does not, the
  build is wrong.
- `maxIter` surface over `{1, 2, 3}`, reported as a surface. Cross-fit `maxIter` only.
  Keep the other pseudo parameters fixed at their defaults; do not tune them on these
  images.
- Runtime per image, added.

### Gate

Adopt when all hold:

1. pooled IoU up by ≥ +0.010
2. mean IoU delta CI lower bound > −0.005
3. Q1–Q3 IoU not down by more than 0.005 each
4. no image regresses by more than 0.10

→ step 8.

---

# Step 8 — Confirmation and handoff

### Build

Set `defaults.m` to the adopted configuration, threshold mode only. Level-set mode stays
byte-identical; assert it once more.

### Evaluate

One final run of the full adopted chain, using the cache chosen in step 6/7:

- cross-fitted headline vs REF0 and vs level-set 0.4708, paired
- per-fold, per-dataset, size quintiles, pooled metrics
- cross-device check on the whole chain
- an **attribution table**: remove each adopted step alone from the final chain
  (leave-one-out), cross-fitted. This catches stacking interactions the step order may
  have hidden, the same issue the 2³ factorial exposed for `imclose`.

### Record

A top-of-file summary in `REFINE_RESULTS.md`:

| step | decision | ΔIoU at adoption | leave-one-out ΔIoU in the final chain |
|---|---|---|---|

Final cross-fitted metrics. The list of flagged items for the human: ambiguous gates, a
step-5 "model-side" verdict, a variance-feature re-run request.

### Docs

Update UNCLUSTERED.md's "Known limits" to match what was measured here. Do not remove any
earlier record.

---

## Expected outcome — write down before step 1

| after | cross-fitted mean IoU | notes |
|---|---|---|
| REF0 | 0.5061 | |
| step 1 | 0.506 – 0.512 | may be SKIP |
| step 3 | 0.515 – 0.530 | largest expected post-processing gain |
| step 4 | +0 – 0.010 | only if relax is a fixed floor |
| step 6 | +0 – 0.015 | re-fitted parameters |
| step 7 | pooled IoU +0.01 – 0.04 | mean IoU moves less |

If the final chain exceeds 0.56 cross-fitted, audit for leakage before reporting it. That is
above the oracle-filter figure (0.5412) plus any plausible TTA gain.

## Explicitly out of scope

Retraining, loss changes, new channels, input normalization, TUH overlay masking (it
changes the network input), level-set changes, `imclose` / `minArea` / `seedRadius` sweeps,
a joint `tau × relaxFloor` sweep outside steps 4 and 6, and seedless tiling. If a result
seems to call for one of these, record the recommendation and stop.
