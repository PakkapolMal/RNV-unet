# RNV-Claude

Segments retinal neovascularization (RNV) lesions in OCTA en-face images, starting from the
localization points an upstream detector emits. A seed-conditioned U-Net predicts a
probability map `P`; the mask is `P > 0.5`, a relaxed threshold around seeds `P` left
empty, and seed claiming. MATLAB R2026a.

## Current result

Out-of-sample, 5-fold cross-validation, **68 images** (DRAC22 41 + TUH 27), whole-image
scoring, current configuration (`cnngvf.defaults()`):

| relax floor | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| 0.20, pre-registered | **0.5097** | 0.5299 | 0.6489 | 0.6311 | 0.7579 |
| cross-fitted per fold | **0.5061** | 0.5371 | 0.6428 | 0.6025 | 0.7868 |

The same `P` through the deleted GVF + level-set stage scored 0.4708. Source:
[docs/results/2026-09-17-refine.md](docs/results/2026-09-17-refine.md); per image:
[docs/results/reference/2026-09-17-relax-0.20.csv](docs/results/reference/2026-09-17-relax-0.20.csv),
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

Datasets (`RNV-DRAC22NV/`, `RNV-DRAC22GT/`, `TUH/`) sit next to this repo, outside it.
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

## History

Code for rejected or superseded work is kept as tags rather than as switches in the code.

| tag | what it holds |
|---|---|
| `archive/pre-cleanup` | the last commit before the 2026-09-25 cleanup: level set, experiment runners, `main/` snake pipeline, every switch the results records name |
| `archive/levelset-master` | `master` as the best GVF + DRLSE version |
| `archive/reduce-refine` | the refinement code: `run_refine`, cross-fitting, seed features, TTA |
| `archive/seedless-prob` | the seedless `P` arm and tiled inference |
| `archive/relax-grid` | `run_relax_grid`: relax window × floor, cross-fitted off the CV cache |

## How work is recorded

1. One experiment, one branch.
2. The result goes in a new dated file in `docs/results/`; existing records are not edited.
3. Adopted: merge the branch and update `docs/SPEC.md` and the table above.
   Rejected: tag it `archive/<name>`, delete the branch, and delete its flag from the code.
