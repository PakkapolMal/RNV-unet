# 2026-10-23 — Cross-validation on the standardized datasets

**Question.** What does the current configuration score on the standardized 69-image
`datasets/`, with every image scored?

**Prediction (from [2026-10-22-standardized-datasets.md](2026-10-22-standardized-datasets.md),
written before running).** Lower than the 68-image 0.5184, because 6 of 42 DRAC22 images
have no usable seed.

**Domain and protocol.** 69 images (DRAC22 42 + TUH 27), 5-fold CV from
[../folds.csv](../folds.csv), every network trained from scratch (`run_cv`, 14.3 h, RTX 5050
Laptop). Whole-image scoring. Current configuration: relax floor 0.20, seed hull link 128 px,
`MinArea` 500, fallback disk r 15. All of these were chosen on the previous 68-image data,
so none was tuned on these images.

## Result

| | n | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|---|
| **all images** | 69 | **0.5047** | 0.5556 | 0.6328 | 0.6410 | 0.6975 |
| DRAC22 | 42 | 0.4357 | | 0.5573 | 0.5941 | 0.6070 |
| TUH | 27 | 0.6121 | | 0.7503 | 0.7139 | 0.8382 |
| *secondary: without the 6 images below* | 63 | 0.5511 | 0.5692 | | | |

Per fold (mean IoU): 0.5440, 0.5313, 0.5116, 0.4724, 0.4611 (n = 14, 14, 14, 14, 13).

The prediction held: the mean is lower. The figure is **not comparable** with 0.5184, which
was measured on a different domain, with different points and different folds.

## The six images with no usable seed

| image | points | IoU | note |
|---|---|---|---|
| `082.png` | 0 | 0 | new image, no points |
| `217.png` | 0 | 0 | had 1 point in the previous CSV |
| `445.png` | 0 | 0 | had 1 (copied) point in the previous CSV |
| `299.png` | 4 | 0 | all points off-lesion, as before |
| `573.png` | 2 | 0 | points moved off-lesion in the standardized CSV |
| `664.png` | 1 | 0.106 | the point is within reach of a lesion edge at inference |

These are localization failures. Whole-image scoring charges them to the pipeline, as
intended. They lower the DRAC22 mean by about 0.07.

## Verification

- Per-image record: [reference/2026-10-23-standardized-69.csv](reference/2026-10-23-standardized-69.csv),
  now `cfg.CV.reference`. `run_cache` reproduces it from the cached `P`: 69 of 69 images,
  `max |dIoU|` 4.4e-16.
- `test_cnngvf` passes.
- The previous 68-image CV output was moved, not deleted, to
  `cnn/result/DRACTUH_unclustered/cv_68img_until_2026-10-22/`. The previous fold networks
  were archived with timestamps by `trainNet`.
