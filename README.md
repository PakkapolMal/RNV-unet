# RNV-Claude

Segments retinal neovascularization (RNV) lesions in OCTA en-face images, starting from the
localization points an upstream detector emits. A seed-conditioned U-Net predicts a
probability map `P`; the mask is `P > 0.5`, a relaxed threshold around seeds `P` left
empty, seed claiming, and a seed hull that fills between nearby seeds. MATLAB R2026a.

## Current result

Out-of-sample, 5-fold cross-validation, **69 images** (DRAC22 42 + TUH 27) from the
standardized [`datasets/`](datasets/), every image scored, whole-image scoring, current
configuration (`cnngvf.defaults()`, seed hull link 128 px, relax floor 0.20):

| | n | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|---|
| **all images** | 69 | **0.5047** | 0.5556 | 0.6328 | 0.6410 | 0.6975 |
| DRAC22 | 42 | 0.4357 | | 0.5573 | 0.5941 | 0.6070 |
| TUH | 27 | 0.6121 | | 0.7503 | 0.7139 | 0.8382 |

Every setting, including the 128 px hull link, was fixed on the earlier 68-image data, so
none was tuned on these images. Six DRAC22 images have no usable localization point and
score about 0; the mean over the other 63 (0.5511) is a labelled secondary figure only. Not
comparable with the earlier 68-image figures (0.5184 cross-fitted). Source:
[docs/results/2026-10-23-standardized-cv.md](docs/results/2026-10-23-standardized-cv.md); per
image: [docs/results/reference/2026-10-23-standardized-69.csv](docs/results/reference/2026-10-23-standardized-69.csv),
which `run_cache` checks against. This table is the only place the current number is quoted;
update it here when it changes.

## Run it

```matlab
cd cnn
test_cnngvf          % self-check
run_cv               % train + score 5 folds (~18 h); cfg.CV.skipTrain reuses networks
run_batch            % segment every image with the single trained network
run_steps("458.png") % every intermediate stage of one image
run_cache            % score the cached CV P; reproduces the table above in seconds
```

Datasets: the standardized, read-only [`datasets/`](datasets/) (69 images; see CLAUDE.md).
Details: [cnn/README.md](cnn/README.md).

## Layout

| path | what it is |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | rules for coding agents: which docs to update, what to verify, what to ask first |
| [`cnn/`](cnn/) | the pipeline: package `+cnngvf`, runners, self-check |
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

## History

Code for rejected or superseded work is kept as tags rather than as switches in the code.

| tag | what it holds |
|---|---|
| `archive/pre-cleanup` | the last commit before the 2026-09-25 cleanup: level set, experiment runners, `main/` snake pipeline, every switch the results records name |
| `archive/levelset-master` | `master` as the best GVF + DRLSE version |
| `archive/reduce-refine` | the refinement code: `run_refine`, cross-fitting, seed features, TTA |
| `archive/seedless-prob` | the seedless `P` arm and tiled inference |
| `archive/relax-grid` | `run_relax_grid`: relax window × floor, cross-fitted off the CV cache |
| `archive/seed-hull` | `cnngvf.seedHull`: triangle fill and closing between linked seeds, after claiming |
| `archive/seed-hull-claim` | the seed hull's same-claim rule and the no-disk fallback switch |

## How work is recorded

1. One experiment, one branch.
2. The result goes in a new dated file in `docs/results/`; existing records are not edited.
3. Adopted: merge the branch and update `docs/SPEC.md` and the table above.
   Rejected: tag it `archive/<name>`, delete the branch, and delete its flag from the code.
