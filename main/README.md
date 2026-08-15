# RNV-Claude/main

Modular rewrite of the four `RNV-Cursor/` scripts. Theory reference: [../docs/CLAUDE.md](../docs/CLAUDE.md).

```
+rnv/            package — one stage per file, all take a cfg struct
  defaults.m     config + path resolution (no hard-coded drive letters)
  loadSeeds.m    localization CSV -> names + [cx cy]
  loadImage.m    image -> grayscale, denoised, + normalized edge map
  initContour.m  seed -> initial contour   ("raycast" | "adaptive")
  segmentSeed.m  init -> GVF ROI -> Snake2D -> mask
  postprocess.m  snake output -> clean mask ("contour" | "component")
  segmentImage.m all seeds of one image -> union mask
  metrics.m      iou / f1 / precision / recall
  evaluate.m     score the pipeline over the GT set
  saveOverlay.m  overlay PNG
run_batch.m      production run over the whole CSV
run_tuning.m     coordinate-descent tuner
test_rnv.m       self-check (synthetic images, no dataset needed)
```

## Usage

```matlab
cd RNV-Claude/main

test_rnv                        % verify the install

run_batch                       % ray-casting pipeline, full CSV
run_batch("adaptive")           % circle pipeline

run_tuning                      % tune Alpha/Beta/Delta/Kappa
run_tuning("adaptive")

cfg = rnv.defaults();           % anything is overridable
cfg.Snake.Kappa = 10;
cfg.Radius.r_max = 300;
run_batch(cfg);
```

Output goes to `result/RNV-Claude-<init>/`:
- `combined/<stem>.png` — union mask per image, read directly by `Compare_DRAC22_IOU.m`
- `overlay/<stem>.png` — image + seeds + final contours

## What changed from RNV-Cursor

**Deduplicated.** `rayCastInit`, `adaptiveRadiusInit`, `removeContourSelfIntersections`,
`selectSeedComponent`, `calc_metrics`, the edge-map block and the entire
coordinate-descent tuner each existed in two copies that had already drifted apart
(different `BASE_DIR`, `Verbose`, CSV column access). Now one copy each.

**Two scripts became one switch.** The ray-casting and adaptive pipelines differed
only in initializer + post-processing, so they are `cfg.Init` and `cfg.PostProc`.
Same for the two tuners.

**Vectorized.** Ray casting sampled 60 rays in a loop with one `interp2` per ray and
a scalar first-drop walk; it is now a single `interp2` over an `n_rays × n_radii`
grid with a `cummax`-based drop detection. Same for the adaptive radius sweep.

**Paths resolve from `mfilename('fullpath')`.** The old scripts had three different
hard-coded `BASE_DIR` values, two of them wrong (`D:\RNV Segmentation Research\...`
instead of `D:\Projects\RNV Segmentation Research\...`) and one pointing at `C:\`.

**CSV columns read positionally.** The runners used `localizations{idx,1}` while the
tuners used `localizations.image_name`; only one survives a header rename.

**Batch runner writes the union mask directly**, which makes the separate
`Combine_RNV_Masks.m` pass unnecessary.

### Deliberate behaviour changes (adaptive pipeline only)

1. `selectSeedComponent` picked the component whose *centroid* was nearest the seed.
   A seed inside a large concave lesion can be nearer another blob's centroid than
   its own. It now takes the component the seed actually falls in, falling back to
   nearest-centroid when the seed lands on background.
2. The radius-energy profile was smoothed with `conv(...,'same')`, which zero-pads
   and biases the first and last radii downward. Now `smoothdata(...,'movmean',...)`
   — same box filter, correct endpoints.

Verified against the old tuning logs (baseline params, images 458 + 461):

| pipeline | old IoU | new IoU |
|---|---|---|
| raycast  | 0.078334 / 0.416540 | 0.078334 / 0.416540 (identical) |
| adaptive | 0.157040 / 0.639970 | 0.164080 / 0.652810 (both fixes) |

The ray-casting path is numerically identical to `gvf_ray_v1.m`, so existing tuned
parameters transfer unchanged.

## Notes

- `rnv.defaults` puts `Fast-Gradient-Vector-Flow-main/` on the path for `Snake2D`
  and its helpers — no manual `addpath` needed.
- `cfg.StartRow = 40` keeps the original convention that rows 1–39 are outside the
  evaluated split.
- Neither pipeline includes the Frangi vesselness front-end
  (`rnv_gvf_ray_filter.m` has it). Add it as a `cfg.Vesselness` branch in
  `rnv.loadImage` if it is worth carrying forward.
