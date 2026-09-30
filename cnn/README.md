# RNV-Claude/cnn

A seed-conditioned U-Net predicts a probability map `P`; the mask is `P > 0.5`, a relaxed
threshold around seeds `P` left empty, `imclose` + `imfill`, and seed claiming.

Specification: [../docs/SPEC.md](../docs/SPEC.md). Vocabulary:
[../docs/CONTEXT.md](../docs/CONTEXT.md). Measurements: [../docs/results/](../docs/results/).

The folder was `CNN+GVF/` until 2026-09-25, when the GVF + level-set stages were deleted
(they measured no better than the threshold). The package keeps its name, `+cnngvf`.

```
+cnngvf/                package - one stage per file, all take a cfg struct
  defaults.m            THE current configuration; dataset registry, toolbox guard
  arm.m                 the three frozen arms, as overrides on defaults
  manifest.m            the image domain: name -> image, mask, dataset
  loadPoints.m          localization points across every active dataset
  valSplit.m            derive the validation split across datasets + area range
  loadChannels.m        image -> grayscale (+ optional median filter / vesselness)
  seedPatch.m           crop geometry + Gaussian seed heat-map (shared)
  prepareData.m         seeds -> train/val patch datastores
  trainNet.m            unet + generalized Dice -> the model file
  inferProbMap.m        batched patch inference + windowed stitch -> P
  maskFromProb.m        P -> mask: threshold, fallback, morphology, claiming, seed hull
  seedFallback.m        relax (or disk) for seeds the threshold missed
  seedDisks.m           the disk, relax's floor
  seedMasks.m           split a mask into the component each seed claims
  seedHull.m            fill between nearby seeds after claiming (inter-frond gaps)
  segmentImage.m        all of the above, one image
  probCache.m           per-fold P cache with a leakage guard
  cvFolds.m             deterministic area-stratified folds over the domain
  evaluate.m            scoring, whole-image or per-segment (cfg.EvalMode)
  segGtFile.m           seed -> its SegmentFilledGT path (clustered arm only)
  loadMask.m metrics.m saveOverlay.m writeSummary.m    small I/O helpers
run_train.m             train
run_batch.m             segment + score every image (mostly in-sample)
run_cv.m                k-fold, every image out-of-sample - the reportable number
run_eval.m              re-score cached masks, no GPU
run_cache.m             score cached cross-validated P, check against the reference
run_steps.m             one image, every intermediate written out
test_cnngvf.m           self-check, 9 checks
```

## Requirements

**MATLAB R2026a** with Image Processing, Deep Learning, Computer Vision and (for the GPU)
Parallel Computing Toolbox. `cnngvf.defaults` fails immediately with a clear message if one
is missing.

The datasets live outside the repo, next to `RNV-Claude/`: `RNV-DRAC22NV/`,
`RNV-DRAC22GT/` and `TUH/`. `models/` and `result/` are git-ignored.

**GPU memory.** At patch 512 an 8 GB laptop GPU has little headroom, so `cfg.Net.miniBatch`
is 2. On a larger card, raising it is the first thing to try for speed.

## Usage

```matlab
cd RNV-Claude/cnn

test_cnngvf                            % verify the install first

run_train                              % -> models/rnvNet_DRACTUH_unclustered.mat
run_cv                                 % 5 folds, every image out-of-sample (~18 h)
run_batch                              % -> result/DRACTUH_unclustered/

run_steps("458.png")                   % one DRAC22 image, every stage written out
run_steps("nv (21).png")               % one TUH image

run_eval(fullfile(cnngvf.defaults().Paths.Output))   % re-score, ~1 s, no GPU
run_cache                              % cached CV P -> scores + diff vs reference, seconds

cfg = cnngvf.defaults();               % anything is overridable
cfg.CV.skipTrain = true;               % reuse the fold networks: inference only, minutes
run_cv(cfg);

run_batch("unclustered")               % a frozen arm, by name
```

Every runner takes nothing (the current configuration), a frozen arm's name, or a `cfg`.

Output goes to `result/<arm>/`:
- `combined/<stem>.png` — union mask per image
- `segment/<stem>-<x>-<y>.png` — one mask per seed, named like the ground truth
- `overlay/<stem>.png` — image + seeds + final contours
- `prob/<stem>.png` — the CNN probability map
- `per_image.csv` (or `per_segment.csv`), `run_summary.txt`
- `cv/` from `run_cv`: per-fold outputs, `probcache/`, and the aggregate `per_image.csv`
- `cache/` from `run_cache`: `per_image.csv`, `run_summary.txt`

## Evaluation

Metrics are `cnngvf.metrics` (IoU, F1, precision, recall). `cfg.EvalMode = "image"` scores
the union mask against the whole `FilledGT` mask, one row per image — the only valid unit
for unclustered points and across both datasets. `"segment"` scores each seed against its
own `SegmentFilledGT` file; it exists for the frozen `clustered` arm, does not charge for
lesions no seed claims (reported as `unclaimedGtPx` instead), and is not comparable with
`"image"`.

Quote cross-validated numbers only: `run_batch` scores images that trained its model. See
the reporting rules in [SPEC.md](../docs/SPEC.md#reporting-rules).
