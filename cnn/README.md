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
  arm.m                 the two frozen arms, as overrides on defaults
  manifest.m            the image domain: name -> image, mask, dataset
  loadPoints.m          localization points across every active dataset
  valSplit.m            derive the validation split across datasets + area range
  seedLabels.m          the GT component each training seed is labelled with (tolerance)
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
  evaluate.m            whole-image scoring
  loadMask.m metrics.m saveOverlay.m writeSummary.m    small I/O helpers
run_train.m             train
run_batch.m             segment + score every image (mostly in-sample)
run_cv.m                k-fold, every image out-of-sample - the reportable number
run_eval.m              re-score cached masks, no GPU
run_cache.m             score cached cross-validated P, check against the reference
run_steps.m             one image, every intermediate written out
run_export.m            bundle for the PyTorch trainer in ../colab/ (data, folds, weights, fixtures)
test_cnngvf.m           self-check, 9 checks
```

## Requirements

**MATLAB R2026a** with Image Processing, Deep Learning, Computer Vision and (for the GPU)
Parallel Computing Toolbox. `cnngvf.defaults` fails immediately with a clear message if one
is missing.

The datasets are the standardized set in [`../datasets/`](../datasets/) (`DRAC22_raw/`,
`DRAC22_filled_gt/`, `TUH_raw/`, `TUH_filled_gt/`, and the two unclustered point CSVs at its
top level). They are read-only; see CLAUDE.md. `models/` and `result/` are git-ignored.

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

run_export                             % -> result/DRACTUH_unclustered/colab_bundle/, minutes
```

### PyTorch trainer

`../colab/` ports patch sampling, the U-Net, the loss, training and `P` inference to PyTorch
so long training can run on Google Colab. New arms train there; everything else stays here. Plan and gates:
[../docs/briefs/colab-parity.md](../docs/briefs/colab-parity.md).

```
../colab/rnvtorch.py      bundle reader, seedPatch, sampler, UNet (+ MATLAB weight load), loss,
                          train_fold (checkpoint + resume), inferProbMap, P-cache writer
../colab/test_parity.py   stage 1 (fixtures), stage 2 (MATLAB weights -> P), stage 3 / budget (one-fold smoke)
../colab/train.ipynb      Colab driver: mount Drive, copy the bundle, smoke, train one fold
```

```bash
python colab/test_parity.py cnn/result/DRACTUH_unclustered/colab_bundle <cache_out>
python colab/test_parity.py stage3 cnn/result/DRACTUH_unclustered/colab_bundle <scratch_run_root>
python colab/rnvtorch.py train cnn/result/DRACTUH_unclustered/colab_bundle <run_root> --fold 1 --rng 0
```

Experiment options on `train` (docs/briefs/train-budget.md; defaults are the parity
configuration): `--lr-schedule cosine`, `--output last`, `--no-inner-val`. A checkpoint
refuses to resume under different options.

`train` checkpoints every epoch to `<run_root>/fold<k>/` and resumes there when rerun. When it
finishes, it writes the fold's held-out `P` to `<run_root>/probcache/`. Score a cache with
`run_cache(cnngvf.defaults(), "<dir>")`. Needs Python 3.11, PyTorch with CUDA, NumPy and
SciPy. On Windows keep `--workers 0`, because worker processes would each copy the bundle.

Every runner takes nothing (the current configuration), a frozen arm's name, or a `cfg`.

Output goes to `result/<arm>/`:
- `combined/<stem>.png` — union mask per image
- `segment/<stem>-<x>-<y>.png` — one mask per seed; `run_eval` recovers seeds from the names
- `overlay/<stem>.png` — image + seeds + final contours
- `prob/<stem>.png` — the CNN probability map
- `per_image.csv`, `run_summary.txt`
- `cv/` from `run_cv`: per-fold outputs, `probcache/`, and the aggregate `per_image.csv`
- `cache/` from `run_cache`: `per_image.csv`, `run_summary.txt`

## Evaluation

Metrics are `cnngvf.metrics` (IoU, F1, precision, recall), scoring the union mask against
the whole `FilledGT` mask, one row per image. Per-segment scoring is at tag
`archive/clustered-arm`.

Quote cross-validated numbers only: `run_batch` scores images that trained its model. See
the reporting rules in [SPEC.md](../docs/SPEC.md#reporting-rules).
