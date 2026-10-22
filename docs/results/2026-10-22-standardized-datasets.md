# 2026-10-22 — Switch to the standardized datasets

**Question.** The peer comparisons ran on each person's own copy of the data. The group
standardized one copy, `RNV-Claude/datasets/`, with its unclustered localization points. What
does moving to it change in this pipeline's domain?

**Prediction (written before running).** The domain grows by one image, `082.png`, which
carries no localization point. The folds will move, so the cached CV `P` and the reference
CSV no longer apply. No IoU is measured here.

**Domain.** 69 images (DRAC22 42 + TUH 27). Every image is scored. No training or scoring was
run, so this record contains no IoU.

## What the standardized data contains

| path | content |
|---|---|
| `datasets/DRAC22_raw/` | 42 images (`082.png` is new), plus older point CSVs that the pipeline does not read |
| `datasets/DRAC22_filled_gt/` | 42 masks (`082.png` added on 2026-10-22), plus a stray `fill.py` |
| `datasets/TUH_raw/` | 27 images `nv (N).png` |
| `datasets/TUH_filled_gt/` | 27 masks `gt (N).tif`. The old ones were `.png` |
| `datasets/DRAC_RNV_Points_Unclustered.csv` | 280 points on 39 images |
| `datasets/TU_RNV_Points_Unclustered.csv` | 591 points on 27 images. The CSV names them `nv (N).jpg` |

## Differences from the previous 68-image domain

| | before | now |
|---|---|---|
| images | 68 (41 + 27) | **69** (42 + 27) |
| points | 874 | **871** |
| usable at `seedTolerance` 20 | 842 | **837** |
| images with no points | 0 | **3**: `082.png`, `217.png`, `445.png` |
| images with no usable seed | 2: `299.png`, `445.png` | **6**: those 3, plus `299.png`, `573.png`, `664.png` |
| GT area range | 1 106 – 432 533 px | 601 – 432 602 px |

- The standardized DRAC22 CSV is not the same as `DRAC22_raw/DRAC_RNV_Points_Unclustered.csv`,
  which the pipeline read before. It has no points on `217.png` or `445.png`, and different
  points on `573.png` (2 points) and `664.png` (1 point instead of 2). All the new points on
  those two images are farther than 20 px from a lesion.
- The mask area range changed in both datasets, so some masks are different from the
  previous copies. A per-mask comparison is not possible: the previous copies were removed.
- `docs/folds.csv` was regenerated from `cnngvf.cvFolds`. 54 of the 69 images kept their
  fold. Every fold contains both datasets.

## Code changes

- `cnngvf.manifest` includes every image and stops with an error on an image that has no
  mask. Before, it skipped that image without a message.
- `run_batch` segments and scores every image in the manifest. Before, it used only the
  images that had points. An image without points predicts nothing and scores IoU 0 against
  its lesion. Smoke run: `082.png` and `445.png` both scored 0.
- `cnngvf.loadPoints` matches points to images by stem, because of the `.jpg` names in the
  TUH CSV.
- The frozen `clustered` arm and per-segment scoring were removed (tag
  `archive/clustered-arm`). Their `SegmentFilledGT` masks and clustered CSVs are not in the
  standardized data.

## Consequence

The current result in the README (0.5184 cross-fitted, 68 images, 5-fold CV) belongs to the
previous domain. It is not comparable with any figure on this domain. `run_cache` refuses
the cached `P` because the folds moved: `097.png` was in fold 2 and is not held out there
now. The new reference needs a full `run_cv` with training (about 18 h). With 6 of 42 DRAC22
images that cannot score above about 0, expect the new mean to be lower than 0.5184.
