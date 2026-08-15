# RNV-Claude/CNN+GVF

Approach 1 of [../docs/spec.md](../docs/spec.md): a seed-conditioned U-Net predicts a
probability map, the map drives a Gradient Vector Flow field, and a DRLSE level set evolves
in that field to the final mask. Theory reference: [../docs/CLAUDE.md](../docs/CLAUDE.md).

```
+cnngvf/                package - one stage per file, all take a cfg struct
  defaults.m            config, path resolution, toolbox guard, GPU flag
  loadChannels.m        image -> [intensity, Frangi vesselness]
  seedPatch.m           crop geometry + Gaussian seed heat-map (shared)
  prepareData.m         seeds -> train/val patch datastores
  trainNet.m            unet + generalized Dice -> models/rnvNet.mat
  inferProbMap.m        batched patch inference + windowed stitch -> P
  buildForce.m          P -> GVF energy field f
  gvfField.m            f -> GVF field (u,v)
  initPhi.m             P + seeds -> phi0
  drlse.m               level-set evolution in the GVF field
  seedMasks.m           split a mask into the component each seed claims
  segGtFile.m           seed -> its SegmentFilledGT path
  evaluate.m            scoring, per-segment or whole-image (cfg.EvalMode)
  segmentImage.m        all of the above, one image
run_train.m             train from the clustered centres
run_batch.m             segment the whole CSV and score it
run_train_unclustered.m train from the raw unclustered detections
run_batch_unclustered.m segment + score those, whole-image
run_eval.m              re-score cached masks, no GPU, no re-segmentation
test_cnngvf.m           self-check (synthetic, no dataset, no trained model)
```

## Requirements

**MATLAB R2026a.** Needs Image Processing, Deep Learning, Computer Vision and (for the GPU
path) Parallel Computing Toolbox. R2025b on this machine has Image Processing only and will
fail immediately with a message from `cnngvf.defaults`.

**GPU memory.** At patch 512 an 8 GB laptop GPU has little headroom: the first attempt with
`MiniBatchSize` 4 died out of memory at iteration 0, in the initial validation pass.
`cfg.Net.miniBatch` is therefore 2 (validation follows `MiniBatchSize` — `trainingOptions`
exposes no `ValidationMiniBatchSize` for `adam`), and `cnngvf.trainNet` calls `gpuDevice(1)`
first so a stale allocation from an earlier run cannot starve it. Steady-state use is then
~1.1 GB of 8 GB. On a larger card, raising `miniBatch` is the first thing to try for speed.

## Usage

```matlab
cd RNV-Claude/CNN+GVF

test_cnngvf                     % verify the install first

% clustered centres, per-segment scoring
run_train                       % ~5 min/epoch on an RTX 5050; 30 epochs ~2.5 h
run_batch                       % ~13 s/image

% raw unclustered detections, whole-image scoring  (the stronger result)
run_train_unclustered           % -> models/rnvNet_unclustered.mat
run_batch_unclustered           % -> result/unclustered/

% re-score masks that already exist - no GPU, ~1 s instead of ~205 s
cfg = cnngvf.defaults(); cfg.EvalMode = "image";
run_eval(fullfile(cfg.Paths.Output, 'unclustered'), cfg);

cfg = cnngvf.defaults();        % anything is overridable
cfg.EvalMode     = "image";     % score the clustered run whole-image too
cfg.Drlse.iter   = 600;
run_batch(cfg);
```

Output goes to `RNV-Claude/CNN+GVF/result/`:
- `combined/<stem>.png` — union mask per image
- `segment/<stem>-<x>-<y>.png` — one mask per seed, named like the ground truth
- `overlay/<stem>.png` — image + seeds + final contours
- `prob/<stem>.png` — the CNN probability map, for diagnosing the force field
- `per_segment.csv` or `per_image.csv` (by `cfg.EvalMode`), `run_summary.txt`

Subdirectories hold the comparison runs: `clustered_image/` is the clustered pipeline scored
whole-image, `unclustered/` is the unclustered pipeline. `combined/` and `prob/` in
`result/` and `result/clustered_image/` are pixel-identical — same model, same seeds, only
the scoring differs.

## Evaluation

Metrics come from `rnv.metrics`, reused from `main/` so both pipelines are scored by
identical code. `cfg.EvalMode` picks the unit — see "Two point sets, two units of evaluation"
below for which to use when.

In `"segment"` mode, two consequences are worth knowing:

- **Ground-truth components no seed claims are not scored** — `separate_segments.py` writes
  no file for them. That is a property of the localization stage, not of this segmenter, so
  `evaluate` reports it separately as `unclaimedGtPx` instead of folding it into recall.
  `"image"` mode does charge for them, which is why its recall is much lower.
- **Two seeds inside one lesion both score against that lesion**, because the ground truth
  is written per seed, not per component. `cnngvf.seedMasks` mirrors that.

## How it differs from RNV-Claude/main

`main/` segments each seed independently with a parametric `Snake2D` and ORs the results.
Here **one level set evolves over the whole image** and the seeds only decide afterwards which
components survive. That is what lets a single lesion split into fronds — `458.png` has 12
ground-truth components against 7 seeds — which a parametric contour cannot do.

`rnv.loadSeeds`, `rnv.loadImage`, `rnv.loadMask` and `rnv.saveOverlay` are reused from
`main/+rnv` rather than copied; `cnngvf.defaults` puts `main/` on the path. The package is
named `+cnngvf` and not `+rnv` precisely so that reuse is possible — two folders named `+rnv`
on the MATLAB path would shadow each other.

## Two point sets, two units of evaluation

`DRAC22_RNV_Localization_Points_Sorted.csv` holds **clustered centres** — one DBSCAN
representative per lesion, 35 over the 17 ground-truth images.
`DRAC_RNV_Points_Unclustered.csv` holds the **raw detections** behind them: 184 on the same
images, 27 of which land outside every lesion.

They demand different scoring, and `cfg.EvalMode` selects it:

- **`"segment"`** — one row per seed, against `SegmentFilledGT/<stem>-<x>-<y>.png`. Valid only
  for the clustered centres, since those files are keyed by their coordinates.
- **`"image"`** — one row per image, union mask against the whole `FilledGT/<stem>.png`.
  Required for the unclustered points: they have no per-point ground truth, and several land
  on one lesion (62 on `1115.png` alone), so per-segment scoring would count one lesion once
  per detection and weight lesions by how many detections they attracted.

The two units are **not comparable**. Whole-image scoring also charges for lesions missed
entirely, which per-segment scoring structurally cannot see — on the same predictions, recall
falls from 0.681 to 0.527 purely because 13 ground-truth components have no clustered seed.
To compare two networks, run both under the same mode.

## Measured results

### Clustered vs unclustered training seeds (whole-image scoring)

Identical evaluation, identical GT; only the localization points differ.

| | mean IoU | median IoU | mean F1 | precision | recall |
|---|---|---|---|---|---|
| clustered centres | 0.4226 | 0.5030 | 0.5587 | 0.7770 | 0.5270 |
| **unclustered detections** | **0.5315** | **0.5896** | **0.6698** | 0.6319 | **0.8293** |

**+0.109 mean IoU, +0.087 median, improved on 12 of 17 images.**

The gain is concentrated exactly where the clustered pipeline was structurally broken — on
lesions far larger than one 512 px patch, which a single centre could never cover:

| image | GT px | clustered | unclustered | recall |
|---|---|---|---|---|
| 1115 | 239 227 | 0.044 | **0.811** | 0.044 → 0.872 |
| 461 | 77 384 | 0.206 | **0.791** | 0.207 → 0.928 |
| 1236 | 64 902 | 0.149 | 0.604 | 0.155 → 0.666 |
| 500 | 13 544 | 0.263 | 0.487 | 0.405 → 0.774 |
| 795 | 100 656 | 0.061 | 0.259 | 0.061 → 0.271 |

`1115.png` carries 62 unclustered detections across its lesion, so the network finally sees
it whole, as a union of overlapping patches. This is direct evidence that **coverage of the
probability map was the bottleneck** — the same conclusion the GVF diagnostics below reach
from the other direction.

The cost is precision, 0.777 → 0.632, and five images regress. The two worst are pure
over-segmentation: `576.png` predicts 101 731 px against 14 216 GT at recall 0.937, and
`573.png` 9 757 px against 2 016 at recall 0.996 — the lesion is found and then massively
overshot. Alongside that, 27 unclustered detections fall outside every lesion; they are
dropped during training but still generate patches at inference, where they can only add
false positives. Filtering those at inference is untried and is the cheapest remaining win.

### Per-segment scores (clustered centres only)

| configuration | mean IoU | precision | recall |
|---|---|---|---|
| patch 256, whole-mask labels, 60 epochs | **0.534** | 0.816 | 0.646 |
| patch 512, per-seed labels, 30 epochs | 0.493 | 0.719 | 0.681 |
| threshold `P > 0.6` alone, no GVF, no level set | 0.496 | 0.728 | 0.668 |

### Does the active contour stage earn its place? Yes — corrected

An earlier version of this README concluded it did not. **That conclusion was wrong**, for a
methodological reason worth recording: the two arms of that comparison were not built the same
way — the pipeline arm applied `cnngvf.seedMasks`, the threshold arm only `bwareaopen` — so
seed-driven component selection, not the level set, carried the difference. It was also
measured on the weakest of the three models.

Re-run on the current model with **identical post-processing in both arms**
(`imclose → imfill → seedMasks`), so the only variable is what produced the binary mask:

| arm | mean IoU | precision | recall |
|---|---|---|---|
| threshold `P > 0.60` | 0.5239 | 0.7393 | 0.6808 |
| **GVF + DRLSE** | **0.5403** | 0.7413 | 0.7054 |

**+0.017 overall, better on 13 of 17 images**, and on the four out-of-sample images
0.5595 → 0.5945, **+0.035**. The four regressions are small (worst −0.034). The gain is real
but modest, and it is mostly recall — the level set grows the contour off the thresholded core
into boundary the threshold clips.

Caveat: 13 of those 17 images are training images, so the out-of-sample figure rests on four
images. The cross-validation protocol below is what will settle the size of the effect.

The mechanism described below still holds and explains why the gain is modest rather than
large.

### Why the gain is modest: the level set converges to an iso-contour of P

The level set moves 583–1171 px per lesion and converges by iteration 100. It also genuinely
*forms* the curve: without it there is no contour, only a pixel mask.

The reason it improves the mask by 0.017 rather than transforming it is *where* it converges.
With `f = |∇P|`, the GVF field points at the ridge of
`|∇P|`, and that ridge is the steepest part of `P` — the `P ≈ 0.5` iso-contour. So the
evolution drags the boundary from wherever `tau` put it back towards the `P ≈ 0.5` level.
**The level set is largely an iso-contour finder for `P`**, which bounds how far it can
depart from a threshold of `P`. Its contribution is regularization and a modest recall gain,
not independent localization.

Three attempts to enlarge the effect all failed, measured on the earlier patch-512 model:

| hypothesis | test | result |
|---|---|---|
| the force field is `P`-derived, so add image information | `Force.source="blend"`, w=0.7/0.5/0.3 | 0.4946 / 0.4958 / 0.4974 — no change |
| IoU is blind to boundary quality, `bfscore` will show more | `bfscore` at matched `tau` | 0.641 vs 0.657 for raw threshold |
| the curvature term over-smooths the fronds | `lambda` 5 → 0.2 | raggedness 4.75 → 4.81, IoU flat |

So the ~0.017 gain appears to be what this force field can deliver; enlarging it needs a force
field carrying information `P` does not already contain, and the raw intensity gradient has
been tested and is too speckled to supply it.

The raggedness index (perimeter / √area) shows the level set produces 4.75 against a ground
truth of 6.31 — it *is* over-smoothing RNV's fern-like boundary, which
[../docs/CLAUDE.md](../docs/CLAUDE.md) warns about for the snake's β. But that smoothing
survives a 25× cut in `lambda`, so it comes from the distance-regularized evolution itself,
not the curvature weight.

The dominant errors are also the wrong shape for contour refinement to fix. On the clustered
pipeline the worst cases were detection failures, not boundary-placement errors — `1115` at
recall 0.044, `795` at 0.061, `461` at 0.207 — and no active contour recovers a lesion the
CNN never responded to. **The bottleneck is the probability map, not the refinement**, which
is why the unclustered points bought +0.11 IoU where every post-processing lever bought
~0.01–0.02.

### Caveats on these numbers

- The GVF/DRLSE comparison was measured on the patch-512 model, which is the weaker of the
  two. The *mechanism* above is model-independent — it holds for any `P` — but the exact
  margins would move on a better network.
- The 256 → 512 comparison changed **four** variables at once (patch size, label source,
  epochs 60→30, miniBatch 8→2), so the regression cannot be attributed to any one of them.
  Per-seed labels dropped the foreground fraction from ~0.147 to ~0.004, which with
  generalized Dice and half the epochs is the leading suspect, but that is untested. A proper
  ablation varies one at a time.
- The clustered → unclustered comparison also moves more than one thing: 30 training seeds
  become 137, `perSeed` drops 16 → 4 to hold the epoch size, and the foreground fraction
  jumps from 0.004 to 0.436. The *direction* is unambiguous and the mechanism (patch coverage
  of large lesions) is confirmed per-image, but the magnitude is not attributable to seed
  count alone.
- The unclustered training seeds are drawn area-weighted in effect — `1115.png` supplies 62 of
  185 points, `517.png` supplies 2 — while whole-image scoring weights every image equally.
- The 256-patch model that scored best per-segment (0.534) was overwritten before this was
  understood. `cnngvf.trainNet` now archives any existing network with its timestamp before
  saving, but recovering that one needs a retrain.
- The localization CSVs have been revised repeatedly during development (`StartRow`, the
  clustered row count, and the unclustered points for `573`/`664` have all changed). Any
  number quoted here is tied to the CSV in place when it was measured; `run_eval` exists
  partly so a stored run can be re-scored cheaply after such a change.

## Deviations from the spec, and why

- **No `f = P` force source.** The spec lists it. The GVF of `P` points at lesion *cores*,
  not *boundaries*, so under the advection sign in `cnngvf.drlse` it collapses the front to a
  point. `buildForce` offers only boundary maps (`"probGrad"`, `"blend"`).
- **Balloon term `alfa = 0`.** The spec suggests ±1.5, which assumes a contour initialized far
  from the object. `phi0` here comes from the probability map and already sits on the boundary,
  and being binary rather than a distance function it gives the Dirac no narrow band — so a
  balloon pushes on the whole image and floods the background. Measured on the synthetic scene
  in `test_cnngvf`: `alfa` 0 → IoU 0.965 stable to 600 iterations, −0.2 → 0.735, −0.5 → 0.130.
- **Edge stopper is gain-corrected.** The classic `g = 1/(1+|∇(G*I)|²)` assumes a 0–255 image.
  On a probability map it yields `g ≈ 0.999` everywhere, i.e. no stopper at all.
  `cnngvf.drlse` normalises the gradient by its own peak and applies `cfg.Drlse.edgeGain`.
- **Patch is 512, not the spec's 256** (still inside its 128–512 range). Measured on this
  dataset, 13 of 35 ground-truth segments need a seed-centred window wider than 256 px, and
  the CNN cannot predict what the patch does not contain. At 256 every clipped segment scored
  badly (`1115` needs 391×1501 → IoU 0.019; `795` needs 501×1815 → 0.074) while every segment
  that fitted scored 0.50–0.88 — precision 0.82 against recall 0.65. At 512, 32 of 35 are
  fully covered and mean coverage is 97.5%. `461`, `535`, `795` and `1115` need 525–1815 px
  and stay clipped; only full-image inference would reach them.
- **Training labels are the seed's own segment**, not the whole `FilledGT` cropped to the
  patch. This makes the training task the one the evaluation scores — "segment the lesion at
  the heat-map" — and gives the seed heat-map channel a job. With whole-image labels a patch
  holding two lesions marks both as foreground and the neighbour leaks into this seed's
  component.
- **No cached patch file.** The spec's `extractPatches` writes patches to disk. Cropping on
  demand from the 17 in-memory images is less code, avoids a multi-GB `.mat`, and
  re-randomises jitter and flips every epoch — augmentation a fixed cache cannot give.
- **No `parfor`.** Parallel Computing Toolbox is used through `gpuArray` (level set),
  `canUseGPU`, and GPU training. Several workers contending for one laptop GPU is slower than
  running serially, so there is no worker-pool branch.

## Known limits and next steps

- 17 ground-truth images. That is thin for a U-Net; `../docs/spec.md` suggests pretraining on
  ROSE or OCTA-500.
- **Clustered centres under-cover the ground truth.** `458.png` has 7 centres against 12
  components; 13 components have no centre at all. Regions no seed reaches stay at `P = 0` by
  design. The unclustered detections largely fix this and are the recommended training input.
- **27 unclustered detections lie outside every lesion.** They are dropped in training but
  still generate patches at inference, where they can only add false positives. Filtering them
  at inference is the cheapest available precision win and is untried.
- **The GVF/DRLSE stage is currently inert.** It forms the contour but does not improve on
  thresholding `P`. Either accept that as a finding, or give the force field information `P`
  does not already carry — a vesselness- or Frangi-driven field would be the next candidate,
  since the raw intensity gradient has been tested and is too speckled.
- The dominant remaining errors are detection failures and over-segmentation, both properties
  of `P`. Effort spent on the probability map will pay more than effort spent on the contour.
- `573.png`'s ground truth is a data-collection error and should be regenerated.
