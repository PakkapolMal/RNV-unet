# RNV-Claude

Segments retinal neovascularization (RNV) lesions in OCTA en-face images, starting from the
localization points an upstream detector emits. A seed-conditioned U-Net (ImageNet-pretrained
EfficientNet-B3 encoder, trained in PyTorch) predicts a
probability map `P`; the mask is `P > 0.5`, a relaxed threshold around seeds `P` left
empty, seed claiming, and a seed hull that fills between nearby seeds. An image with no
points at all is segmented by a second, seedless network. MATLAB R2026a.

## Current result

Out-of-sample, 5-fold cross-validation, **69 images** (DRAC22 42 + TUH 27) from the
standardized `datasets/` (not included, see Data), every image scored, whole-image scoring, current
configuration (`cnngvf.defaults()`: a pretrained EfficientNet-B3 U-Net trained by the PyTorch
trainer with CE + Dice loss and affine + CLAHE augmentation, 40 epochs with cosine decay and
last-epoch output; seed hull link 128 px, relax floor 0.20; a seedless network for the
images with no points):

| | n | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|---|
| **all images** | 69 | **0.6364** | 0.6694 | 0.7535 | 0.7667 | 0.7625 |
| DRAC22 | 42 | 0.5797 | | 0.7015 | 0.7368 | 0.6926 |
| TUH | 27 | 0.7245 | | 0.8343 | 0.8131 | 0.8713 |

Post-processing settings were fixed on the earlier 68-image data and the replica U-Net's `P`;
re-fitting `tau` and `relaxFloor` on this `P` gained nothing (+0.0004 cross-fitted). The network recipe passed a pre-registered margin gate
against the previous configuration (+0.087 IoU, CI [+0.056, +0.118]); it was trained once,
so the figure carries the training noise floor of about ±0.015. Six DRAC22 images have no
usable localization point. The three with no points at all go to the seedless network, which
passed its own gate (+0.024 on this table; `082` 0.79, `217` 0.86, `445` 0); the three with
stray points score ≈ 0. The mean over the 63 with a usable seed (0.6668) is a labelled
secondary figure only. Previous: 0.6125 (no seedless network, 2026-11-11), 0.5256 (replica
U-Net, 2026-11-07), 0.5047 (MATLAB trainer). Sources:
[docs/results/2026-11-11-paper-recipe.md](docs/results/2026-11-11-paper-recipe.md) and
[docs/results/2026-11-12-seedless-fallback.md](docs/results/2026-11-12-seedless-fallback.md); per
image: [docs/results/reference/2026-11-12-seedless-fallback-69.csv](docs/results/reference/2026-11-12-seedless-fallback-69.csv),
which `run_cache` checks against. This table is the only place the current number is quoted;
update it here when it changes.

## Run it

```matlab
cd cnn
test_cnngvf          % self-check
run_cache            % score the cached CV P; reproduces the table above in seconds
run_batch            % segment every image with the MATLAB full-data network (two adoptions old)
run_steps("458.png") % every intermediate stage of one image, same network
run_export           % bundle for the PyTorch trainer; training runs on Colab (colab/train.ipynb)
```

Datasets: the standardized, read-only `datasets/` (69 images; see Data).
Details: [cnn/README.md](cnn/README.md).

## Data

The images, masks and localization points are **not in this repository**. The code expects
them in `datasets/` at the repository root, laid out as `cnn/README.md` describes:

- **DRAC22** (42 images): from the public DRAC 2022 challenge (Diabetic Retinopathy Analysis
  Challenge, MICCAI 2022). Obtain it from the challenge under its own terms.
- **TUH** (27 images): a clinical dataset that is not publicly available.

The localization points (the detector output this pipeline starts from) are not public
either. The records in `docs/results/` give per-image scores and image names only.

## License

MIT, see [LICENSE](LICENSE). The license covers the code and documents here, not the data.

## Layout

| path | what it is |
|---|---|
| [`cnn/`](cnn/) | the pipeline: package `+cnngvf`, runners, self-check |
| [`colab/`](colab/) | the PyTorch trainer the current configuration trains with, on Google Colab |
| [`docs/SPEC.md`](docs/SPEC.md) | how the current pipeline works and why — current state only |
| [`docs/CONTEXT.md`](docs/CONTEXT.md) | glossary: one term, one meaning |
| [`docs/results/`](docs/results/) | dated measurement records, append-only |
| [`docs/briefs/`](docs/briefs/) | plans written before the work, kept as written |
| [`docs/THEORY.md`](docs/THEORY.md) | literature and snake/GVF background |

### Results, oldest first

| record | question | outcome |
|---|---|---|
| [2026-08-19](docs/results/2026-08-19-clustered-vs-unclustered.md) | cluster centres or raw detections as seeds? (17 images) | raw detections, +0.086 IoU |
| [2026-09-01](docs/results/2026-09-01-filters-perseed-levelset.md) | median filter, vesselness, `perSeed`, the level set | filters off; `perSeed` 4; level set +0.017 at best |
| [2026-09-16](docs/results/2026-09-16-reduce.md) | does the GVF + level set earn its place? | no — deleted; relax fallback +0.037 |
| [2026-09-17 seedless](docs/results/2026-09-17-seedless.md) | `P` from whole-frame tiles, seeds only claim? | failed its pilot gate |
| [2026-09-17 refine](docs/results/2026-09-17-refine.md) | any inference-only gain on top? | none; relax 0.20 locked |
| [2026-09-18](docs/results/2026-09-18-relax-no-disk.md) | drop seeds relax cannot serve? | no, −0.0006 |
| [2026-09-30](docs/results/2026-09-30-relax-grid.md) | tune the relax window × floor? | no, −0.0098 cross-fitted; 128 px is the best window |
| [2026-09-30 paired](docs/results/2026-09-30-relax-floor-015-paired.md) | floor 0.15 vs 0.20, per image? | noise on IoU, +0.0027 CI [−0.003, +0.010]; a precision / recall trade |
| [2026-09-30 seed hull](docs/results/2026-09-30-seed-hull.md) | fill between seeds linked within 2σ? | no: triangles inert (+0.0002); closing +0.0056 CI [−0.010, +0.022], trips the precision guard |
| [2026-09-30 seed hull d](docs/results/2026-09-30-seed-hull-d.md) | triangles with a cross-fitted link distance? | adopted at 128 px: +0.0087 CI [+0.002, +0.016], over a failed precision guard (−0.0106) |
| [2026-10-01](docs/results/2026-10-01-seed-hull-claim.md) | fill only triangles on one claimed component, with or without the disk? | no: guard repaired but −0.0028 IoU; fallback-disk fragment joins are 82 % lesion |
| [2026-10-02](docs/results/2026-10-02-minarea.md) | tune `MinArea` (100–1000 px)? | no: every fold picks 500; inert below, above ~700 px the two smallest lesions go to IoU 0 |
| [2026-10-02 disk](docs/results/2026-10-02-minarea-disk.md) | `MinArea` × disk radius (0–30), jointly? | no: every fold picks (500, 15); the disk is near-inert under relax |
| [2026-10-09 U-Net only](docs/results/2026-10-09-unet-only-baseline.md) | what does the network alone score, `P > 0.5` with no post-processing? (68 images, 5-fold CV) | 0.4201 vs 0.5207 full: post-processing +0.10, CI [+0.065, +0.138], all recall |
| [2026-10-22 datasets](docs/results/2026-10-22-standardized-datasets.md) | what changes on the standardized `datasets/`? | 69 images, 871 points; 6 DRAC22 images without a usable seed (was 2); folds moved |
| [2026-10-23 CV](docs/results/2026-10-23-standardized-cv.md) | the current configuration on the standardized 69 images? (5-fold CV) | **0.5047** IoU (DRAC22 0.4357, TUH 0.6121); not comparable with 68-image figures |
| [2026-11-02 PyTorch inference](docs/results/2026-11-02-colab-parity-inference.md) | does the PyTorch port reproduce MATLAB's `P` from MATLAB's own weights? (69 images, 5-fold CV) | yes: 0.504706 vs 0.504699, max \|dIoU\| 2.2e-4; matches MATLAB CPU to 1.4e-7, the \|ΔP\| ≤ 1e-3 gate fails on MATLAB's GPU noise |
| [2026-11-02 PyTorch smoke](docs/results/2026-11-02-colab-parity-smoke-local.md) | does PyTorch training run, resume and match MATLAB's setup? (fold 1, 2 epochs, local GPU) | yes: counts equal, mean within 0.0038, loss falls, resume works; 565 s/epoch, ~15.7 h per CV locally |
| [2026-11-02 Colab smoke](docs/results/2026-11-02-colab-parity-smoke-colab.md) | do stages 1–3 pass on Colab, and what does stage 4 cost? (fold 1, 2 epochs, L4) | all pass; 290 s/epoch, ~8.1 h per CV; two seeds fit 120 CU at up to ~6.5 CU/h |
| [2026-11-05 parity CV](docs/results/2026-11-05-colab-parity-cv.md) | does a PyTorch-trained CV match MATLAB? (69 images, 5-fold CV, 2 RNGs) | means within 0.005 (0.5002, 0.5010 vs 0.5047), but RNG 0's CI reaches −0.024, failing the ±0.02 gate; noise floor per-image SD 0.062, CI ±0.015 |
| [2026-11-05 tau](docs/results/2026-11-05-tau-crossfit.md) | cross-fit `tau` over 0.40–0.70? (69 images, 5-fold CV, reference + 2 PyTorch caches) | no: −0.0039 CI [−0.0081, −0.0004]; worse on every cache; the curve peaks at 0.5 |
| 2026-11-05 overlay (record not public) | do false positives concentrate on the TUH crosshair? (27 TUH images, 5-fold CV) | no: FP rate 0.80× elsewhere; an oracle fix gains at most +0.0019 TUH IoU; the overlay stays in |
| [2026-11-07 budget](docs/results/2026-11-07-train-budget.md) | train 40 epochs, cosine LR, last epoch, inner images in training? (69 images, 5-fold CV, PyTorch) | **adopted: 0.5256**, +0.025 over each 20-epoch baseline, CIs [+0.008, +0.044] and [+0.008, +0.041]; one fold stalled and was retrained per the stall rule |
| [2026-11-10 budget P](docs/results/2026-11-10-train-budget-pmaps.md) | why does the budget help? (69 images, 5-fold CV `P`, post hoc) | `P` is more decisive (undecided area −0.086 on 61/66 images); gain sits on spill (+0.061) and under-coverage (+0.052); deep misses (`P` < 0.2 in GT) untouched |
| [2026-11-11 recipe](docs/results/2026-11-11-paper-recipe.md) | pretrained EfficientNet-B3 U-Net, CE + Dice, affine + CLAHE, one bundle? (69 images, 5-fold CV, PyTorch) | **adopted: 0.6125**, +0.087 CI [+0.056, +0.118] by a margin gate; spill ×2.20 → ×1.17, under-coverage ×0.52 → ×0.86 |
| [2026-11-11 refit](docs/results/2026-11-11-tau-floor-refit.md) | re-fit `tau` × `relaxFloor` on the new `P`? (69 images, 5-fold CV, cross-fitted) | no: +0.0004 CI [−0.003, +0.003]; folds disagree; both stay |
| [2026-11-12 seedless](docs/results/2026-11-12-seedless-fallback.md) | a seedless network for the 3 images with no points? (69 images, 5-fold CV, rule cross-fitted on the 18 images with 1–3 points) | **adopted: 0.6364**, +0.024; targets 0.550 mean IoU (`445` still 0), proxy 0.467 |
| [2026-11-12 fusion](docs/results/2026-11-12-fusion-weak-seeds.md) | bring the seedless `P` in at weak seeds, before relax? (69 images, 5-fold CV, cross-fitted) | no: +0.0010 CI [−0.0007, +0.0028]; 90 % of weak seeds have no seedless component either; relax stays |
| [2026-11-12 fusion U](docs/results/2026-11-12-fusion-uncertainty.md) | fuse the seedless `P` on images whose seeded `P` is unsure? (69 images, 5-fold CV, cross-fitted) | +0.0116 CI [+0.004, +0.022], but gate 2 fails: all 5 folds fuse every image and split 3–2 on τ_s; half the gain is `795` and `461`; not adopted, pending decision |
| [2026-11-12 fusion confirm](docs/results/2026-11-12-fusion-confirm.md) | does fusing every image replicate on a second seedless network? (69 images, 5-fold CV, cells fixed before training) | **yes: +0.0104** CI [+0.004, +0.019] (first run +0.0120); the sliver filter loses to plain fusion; adoption pending |

## History

Code for rejected or superseded work is kept as tags rather than as switches in the code.

| tag | what it holds |
|---|---|
| `archive/pre-cleanup` | the last commit before the 2026-09-25 cleanup: level set, experiment runners, `main/` snake pipeline, every switch the results records name |
| `archive/levelset-master` | `master` as the best GVF + DRLSE version |
| `archive/reduce-refine` | the refinement code: `run_refine`, cross-fitting, seed features, TTA |
| `archive/seedless-prob` | the seedless `P` arm and tiled inference |
| `archive/fusion` | `cnngvf.fuseWeakSeeds` and `run_fusion_grid`: seedless `P` at weak seeds |
| `archive/seedless-fallback` | `run_seedless_fit`: the no-point rule's cross-fit, and its top-1 keep option |
| `archive/relax-grid` | `run_relax_grid`: relax window × floor, cross-fitted off the CV cache |
| `archive/seed-hull` | `cnngvf.seedHull`: triangle fill and closing between linked seeds, after claiming |
| `archive/seed-hull-claim` | the seed hull's same-claim rule and the no-disk fallback switch |

## How work is recorded

1. One experiment, one branch.
2. The result goes in a new dated file in `docs/results/`; existing records are not edited.
3. Adopted: merge the branch and update `docs/SPEC.md` and the table above.
   Rejected: tag it `archive/<name>`, delete the branch, and delete its flag from the code.
