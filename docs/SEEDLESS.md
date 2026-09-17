---
title: Seedless Probability Map — Implementation Plan
description: Compute P from the image alone with overlapping tiles, keep the seeds only for claiming, and measure it paired against reduced + relax.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/CNN+GVF, MATLAB R2026a, branch seedless-prob
status: implemented; fold-1 pilot FAILED the gate — stopped, folds 2-5 not run
updated: 2026-09-17
---

# Seedless Probability Map — Implementation Plan

Vocabulary: [CONTEXT.md](CONTEXT.md). Baseline and protocol: [REDUCE_RESULTS.md](REDUCE_RESULTS.md).
Where this document and UNCLUSTERED.md disagree on protocol, UNCLUSTERED.md wins.

## The question

Today every patch is centred on a localization point and carries that point's heat-map, so
**P is 0 wherever no seed reaches**. Does a probability map computed from the image alone —
no seed in the network at all — recover lesion the seeded map never covers?

This is a **recall / coverage** bet, and deliberately only that. It is *not* a fix for false
seeds: REDUCE_RESULTS.md §10 re-inferred P with the stray seeds removed and it moved nothing
(0.5412 vs 0.5409 on cached P). A wrong seed costs precision at *claiming*, not through P. A
post-processing seed filter is the right tool for that and is listed under
[Later](#later-not-in-this-plan).

## Why it might help

| failure | today | what seedless P changes |
|---|---|---|
| lesions no seed reaches | P = 0 there by construction | P is defined everywhere |
| lesions larger than one 512 px patch | `1115`, `795`, `nv (21)`, `461`, `535` clipped | tiles overlap across the whole frame |
| seeds whose P never crosses tau | 258 / 874 (29.5 %), fallback *is* the output | a lesion the tile sees in context may cross tau |
| images with no usable seed | `299`, `445` score 0 | **unchanged** — claiming still needs a seed |

## Why it might not

- **The task gets harder.** With no heat-map the network must separate RNV from normal
  vasculature, IRMA and optic-disc noise on its own, from ~53 training images per fold. That
  is the regime where the localization literature found VNet over-segmenting. Expect
  precision to fall; the question is whether recall pays for it.
- **The heat-map may be doing real work.** A weak lesion the network only finds *because*
  it was told where to look will be lost.
- **Claiming bounds the upside.** Seeds still decide which components survive, so new P
  outside every seed's reach is discarded. The gain can only come from components some seed
  claims.

## Settled design

Decided in the grilling session of 2026-09-16.

| # | decision | choice |
|---|---|---|
| Q1 | target | recall / coverage, not false seeds |
| Q2 | role of seeds | **P seedless; seeds still claim** — `maskFromProb`, `seedMasks` and the relax fallback unchanged |
| Q3 | "whole image" | **overlapping 512 tiles**, not a literal 1024² pass (8 GB GPU; miniBatch 4 at 512 already OOMed) |
| Q4 | success | paired vs reduced + relax 0.20, mean IoU up **and** bootstrap CI of the paired delta excluding zero |
| Q5 | compute | fold-1 pilot first, then the remaining folds |
| Q6 | labels | the whole `FilledGT` cropped to the patch |
| Q7 | crop sampling | 50 % uniform random, 50 % containing a random GT pixel at a **random position** in the crop |
| Q8 | held fixed | everything except channels, labels and sampling |
| Q9 | inference tiles | stride 256 → 3 × 3 = 9 tiles, raised-cosine blend |
| Q10 | tau / floor | fixed at 0.5 / 0.20 for the headline; curves secondary |
| Q11 | pilot gate | fold-1 IoU ≥ 0.39 **and** GT coverage above seeded P |
| Q12 | code shape | new arm `DRACTUH_seedless` + `cfg.Patch.seeded` switch |

## Pipeline

```
seeded (today)                                   seedless (this plan)

image ─► loadChannels ─┐                         image ─► loadChannels
seed ─► heat-map ──────┤                                        │
                       ▼                                        ▼
       512² seed-centred patch, 2 ch             512² tiles, stride 256, 1 ch  (9 per image)
                       │                                        │
                    U-Net                                    U-Net
                       │                                        │
       raised-cosine stitch → P (0 off-seed)     raised-cosine stitch → P (defined everywhere)
                       │                                        │
                       └──────────────┬─────────────────────────┘
                                      ▼
            maskFromProb: P > 0.5 → relax fallback 0.20 → imclose → imfill → seedMasks
                                      ▲
                                    seeds  (unchanged: the only place they enter)
```

## What to change

### 1. `+cnngvf/arm.m` — the `DRACTUH_seedless` arm

A copy of `DRACTUH_unclustered` differing only in:

| field | value |
|---|---|
| `seeded` | `false` → sets `cfg.Patch.seeded` |
| model | `rnvNet_DRACTUH_seedless.mat`, folds `models/cv/DRACTUH_seedless_fold<k>.mat` |
| output | `result/DRACTUH_seedless` |
| post | `Post.mode = "threshold"`, `fallbackMode = "relax"`, `relaxFloor = 0.20` |

`cfg.Patch.seeded` defaults to `true` in `defaults.m`. **Every existing arm must stay
byte-identical** — verify with the existing `test_cnngvf` level-set check on `458.png`.

Folds come from `cnngvf.cvFolds`, which depends only on the manifest, so both arms hold out
exactly the same images and are paired image by image.

### 2. `+cnngvf/prepareData.m` — seedless training data

When `~cfg.Patch.seeded`:

- **Labels** — the whole `FilledGT` of the image, cropped with the patch (Q6). No per-seed
  segment, no `seedTolerance`, no dropped background seeds.
- **Sampling** (Q7) — per crop, a fair coin:
  - *uniform*: top-left uniform over valid positions;
  - *positive*: pick a random GT foreground pixel, then place it at a uniform random position
    inside the crop (not the centre), clipped to the frame.
- **Epoch size** — `patchesPerEpoch` computed exactly as the seeded path would for the same
  fold (`perSeed × trainSeeds`, balanced by image), then spent as crops balanced by image.
  Compute is matched; seed counts only set the budget, never a position.
- **Augmentation** — whatever the seeded path applies (flips), unchanged.
- **Validation** — deterministic: the 9-tile inference grid over each inner-validation image,
  so `best-validation` checkpoint selection sees the same geometry inference uses.
- `info.numChannels` = image channels only (no `+ 1`).

### 3. `+cnngvf/trainNet.m`

Only the channel description string. Network, loss (`1 - generalizedDice`), Adam 1e-3,
20 epochs (as `run_cv`), miniBatch 2, `best-validation` — unchanged (Q8).

**Check first:** `generalizedDice` on a crop with no foreground. Uniform crops will often be
all background; if the foreground weight goes to Inf/NaN the loss is undefined. Verify on a
synthetic all-zero label before the pilot; if it breaks, the fix is a smoothing term in the
loss, which is then a deviation to record — not something to discover 3 h into fold 1.

### 4. `+cnngvf/inferProbMap.m` — tiled inference

When `~cfg.Patch.seeded`, ignore `seeds` and predict the fixed 3 × 3 grid (origins 0, 256,
512 on each axis), all 9 tiles in one `predict` batch (`Net.inferBatch` permitting), blended
with the existing raised-cosine window. The seeded branch is untouched.

`segmentImage`, `maskFromProb`, `seedFallback`, `seedMasks` — **no changes.**

### 5. Caching and leakage

`run_cv` already writes `result/<arm>/cv/probcache/` with the fold stamped in, and
`probCache("read")` re-derives the folds and refuses in-sample P. Nothing new — but seedless P
must go to **its own** cache so it can never be read as seeded P.

### 6. `run_seedless_compare.m` — the paired report

Off both caches, no GPU. For the images present in both:

- headline: mean / median IoU, F1, precision, recall — seeded vs seedless, both through
  `maskFromProb` with relax 0.20; paired delta with sign test and bootstrap 95 % CI
- **GT coverage** (Q4 diagnostic): fraction of `FilledGT` pixels with P > 0.5, and fraction
  of GT components touched by P > 0.5, seeded vs seedless
- **P outside coverage**: fraction of P > 0.5 pixels outside GT (the precision risk, before
  claiming)
- **unclaimed gain**: GT pixels with seedless P > 0.5 that no seed claims — the upside the
  claiming rule throws away, which sizes the "seeds decide" constraint
- secondary, as curves only: tau 0.30–0.80 and relax floor 0.05–0.30 on seedless P, plus the
  cross-fitted floor
- per dataset (DRAC22 / TUH) and per fold

### 7. `test_cnngvf.m` — new checks

1. Every existing arm: `cfg.Patch.seeded` is `true` and `458.png` level-set output unchanged.
2. Seedless `inferProbMap` returns a `[1024 1024]` P with no zero band at tile seams and does
   not depend on the seeds passed (same P for two different seed sets).
3. Seedless labels are the full GT crop; a positive crop contains foreground; the positive
   pixel is not always at the centre.
4. `generalizedDice` loss is finite on an all-background label.

## Order of work

1. Branch `seedless-prob` (done). This plan.
2. `cfg.Patch.seeded` + the arm. `test_cnngvf` still passes, `458.png` byte-identical.
3. `generalizedDice` on an empty label — resolve before anything trains.
4. `prepareData` seedless path + tests 3–4. Dry run: one epoch on fold 1, inspect a batch of
   crops and labels visually.
5. `inferProbMap` tiled path + test 2.
6. **Pilot:** `run_cv("DRACTUH_seedless")` restricted to fold 1 (~3–4 h). Score with
   `run_seedless_compare` on the 14 fold-1 images.
7. **Gate** (below). Stop, or resume the remaining folds with `cfg.CV.skipTrain` (~12 h).
8. Full paired report → `docs/SEEDLESS_RESULTS.md`.

## Expected goals — written before running

### Baseline to beat

Reduced + relax 0.20, 68 images, 5-fold CV (REDUCE_RESULTS.md §9):

| | IoU | F1 | precision | recall |
|---|---|---|---|---|
| all folds | **0.5097** (cross-fitted floor 0.5061) | 0.6489 | 0.6311 | 0.7579 |

| fold | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| relax 0.20 IoU | **0.4124** | 0.6004 | 0.5490 | 0.4832 | 0.5011 |

### Pilot gate — fold 1, 14 images

Continue to all five folds only if **both** hold:

1. seedless + relax 0.20 fold-1 mean IoU **≥ 0.39** (within ~0.02 of 0.4124; with 14 images
   and fold-to-fold sd ≈ 0.07, a smaller gap is noise)
2. GT pixel coverage of seedless P > 0.5 is **higher** than seeded P's on the same images

If (2) fails the mechanism being bet on is absent, and the full run cannot pay off whatever (1)
says.

### Full-run decision (Q4)

**Success**: 68-image mean IoU above 0.5097 **and** the bootstrap 95 % CI of the paired
per-image delta excludes zero. Anything else is reported as a non-result, with the coverage
and precision diagnostics explaining which way it failed.

### Predictions, for later comparison

- **Recall up, precision down.** Coverage rises; P > 0.5 outside GT rises more, and some of
  it is claimed.
- **The gain, if any, concentrates on large and multi-component lesions** — `1115`, `795`,
  `nv (21)`, `nv (17)`, `1236`, `nv (27)` — where a 512 seed patch visibly under-covers.
- **Small single-seed lesions lose** — `217`, `421`, `123`, `nv (4)` — where the heat-map
  tells the network exactly where a faint lesion is.
- **Fewer fallback firings.** The 29.5 % of seeds with no P > tau should drop; if it does not,
  seedless P is not finding the weak lesions either.
- **`299` and `445` stay at 0.** Nothing here gives them a seed.
- **Calibration shifts.** The tau curve may peak away from 0.5. Recorded as a finding, not
  used to re-tune the headline.

A plausible honest outcome is a wash on the mean with a real device or size split. That is
worth knowing: it would say the heat-map and the tiling each carry information the other
lacks, and point at the hybrid below rather than a replacement.

## Pilot result — 2026-09-17: stopped at the gate

`run_cv("DRACTUH_seedless")` with `cfg.CV.folds = 1` (2 h 37 min, 2 448 patches/epoch — the
same budget as the seeded fold 1), then `run_seedless_compare`. **14 images; this gates a
decision and is not a result.**

| fold 1, 14 images | seeded | seedless | Δ |
|---|---|---|---|
| mean IoU | 0.4124 | **0.3273** | −0.085 |
| F1 | 0.5582 | 0.4572 | −0.101 |
| precision | 0.5102 | 0.6841 | +0.174 |
| recall | 0.7138 | 0.3776 | −0.336 |
| GT px covered by P > 0.5 | 0.4618 | **0.3285** | −0.133 |
| GT components touched by P > 0.5 | 0.857 | 0.929 | +0.071 |
| P > 0.5 px lying off GT | 0.423 | 0.623 | +0.200 |
| seeds where the fallback fires | 139 / 233 | 135 / 233 | −4 |

Paired ΔIoU −0.085, 4 better / 9 worse, CI [−0.214, +0.046].

| gate | needed | got | |
|---|---|---|---|
| 1. fold-1 IoU | ≥ 0.39 | 0.3273 | **FAIL** |
| 2. GT coverage above seeded | > 0.4618 | 0.3285 | **FAIL** |

**Both fail, so per the plan the remaining folds were not run.**

### What it says

- **The mechanism being bet on is absent.** Seedless P covers *less* ground truth, not more.
  It touches slightly more lesions (0.93 vs 0.86 of components) but covers each one thinly,
  and the fallback fires just as often (135 vs 139). The recall story this plan rested on
  did not appear.
- **It is the predicted failure, inverted.** The prediction was recall up, precision down.
  Measured: recall collapses (0.71 → 0.38) and precision rises. Before claiming, P is noisier
  (62 % of P > 0.5 lies off GT); claiming discards most of that, so what survives is precise
  but sparse.
- **The network barely learned.** Validation loss went 0.91 → 0.69 by epoch 1 and then sat at
  0.65–0.73 for 19 epochs; the seeded fold-1 network finished at 0.437. Training-loss prints
  of exactly 1 are all-background crops.
- **Likely confound: the loss on empty crops.** `1 − generalizedDice` scores any predicted
  foreground on an all-background crop as Dice ≈ 0 (checked before training; finite, but
  harsh). Uniform sampling makes those a large share of every epoch, which pushes P down
  everywhere. The tau curve agrees: seedless IoU is best at the *lowest* tau tried (0.39 at
  0.30) and the relax floor curve at the lowest floor — P is under-confident, not merely
  mislocated. Q8 fixed the loss deliberately, so this is a finding about the recipe as
  specified, not about seedless P in general.
- **The device split repeats.** TUH improved (0.443 → 0.491, n = 5; `nv (25)` +0.42, `nv (9)`
  +0.27); DRAC22 collapsed (0.395 → 0.237, n = 9; `123.png` 0.55 → 0). Small single-seed
  DRAC22 lesions lose exactly as predicted — the heat-map was doing real work there.

### Follow-up: route TUH to seedless, DRAC22 to seeded? No.

The pilot's device split suggested routing by dataset. Checked batch-style (NOT
cross-validated): the seeded and seedless **fold-1** networks, which trained on the same 52
images, run over all 68 through the same relax pipeline.

| mean IoU, 68 images | all seeded | all seedless | routed (TUH seedless, DRAC22 seeded) |
|---|---|---|---|
| all | 0.4575 | 0.4551 | **0.4578** (+0.0004, 13 better / 14 worse) |

| | held out (14) | trained on (54) |
|---|---|---|
| DRAC22 seeded → seedless | 0.395 → 0.237 | 0.437 → **0.476** |
| TUH seeded → seedless | 0.443 → **0.491** | 0.516 → 0.507 |

- **The split reverses off fold 1**, and seedless wins on 13 of 27 TUH images. The pilot's TUH
  advantage was 5 images of noise; the routing rule buys nothing.
- **Seedless overfits.** Best of the three on images it trained on (0.488), worst on unseen ones
  (0.327): a 0.16 train/held-out gap against seeded's 0.06.
- In-sample numbers; not comparable to any cross-validated figure. The held-out rows reproduce
  the pilot exactly.

Routing is dropped rather than tested on folds 2–5.

### If this is revisited

Not run, and each is a new pre-registration rather than a tweak to this one:

1. **The loss, first** — batch-level Dice or Dice + cross-entropy, so an empty crop is not a
   near-maximal penalty. Without this, nothing else here is interpretable.
2. **Hybrid P** — heat-map randomly blanked in training, so one network works with or without
   a seed. Suggested by the TUH/DRAC22 split: tiling helps large lesions, the heat-map helps
   small ones.

The cheaper open item is unchanged and unaffected by this result: the post-processing seed
filter (REDUCE_RESULTS.md §10).

Artefacts: `result/run_cv_seedless_fold1.log`, `models/cv/DRACTUH_seedless_fold1.mat`,
`result/DRACTUH_seedless/compare/`.

## What NOT to do

- **Do not retune tau or the relax floor** for the headline (Q10).
- **Do not change the network, loss, optimiser, epochs or miniBatch** — P's source must be the
  only variable (Q8).
- **Do not drop the seeds from claiming** — that is a second variable (Q2).
- **Do not touch any existing arm**, cache or fold network. The seeded record must remain
  reproducible and pairable.
- **Do not center positive crops on lesions** — that re-teaches the positional prior this
  experiment removes (Q7).
- **Do not quote fold-1 pilot numbers as a result.** 14 images gate a decision; they are not
  a finding.

## Later (not in this plan)

- **Post-processing seed filter** — REDUCE_RESULTS.md §10 bounds it at ~+0.03 IoU, all
  precision, re-inference unnecessary. A practical version scores each seed from P (e.g.
  local peak, claimed-component support) without ground truth. Independent of this plan;
  cheap, no GPU.
- **Hybrid P** — heat-map optional in one network (trained with the heat-map randomly
  blanked), or `max`/blend of seeded and seedless P. Only if this plan shows the two carry
  different information.
- **Seedless claiming** — dropping seeds from post-processing entirely. A different research
  question (full segmentation), and it would forfeit the localization stage's precision.
