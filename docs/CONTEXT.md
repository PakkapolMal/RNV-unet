---
title: RNV Segmentation — Domain Glossary
description: The canonical vocabulary of the RNV segmentation context. One term, one meaning.
kind: glossary
status: current
dataset: DRAC22 + TUH (68 images)
updated: 2026-09-30
---

# RNV Segmentation

Turning localization points into lesion boundaries on OCTA en-face retinal
images. A prior stage says *where* a lesion is; this context decides *what
shape* it is.

Implementation: [`../cnn/`](../cnn/). Specification: [SPEC.md](SPEC.md). Measurements:
[results/](results/). Theory: [THEORY.md](THEORY.md).

## Language

### The clinical subject

**RNV**:
Retinal neovascularization — abnormal, fragile new vessels growing in
proliferative diabetic retinopathy. The thing being segmented.
_Avoid_: NV, neovascularisation, tuft, lesion (when RNV specifically is meant)

**Lesion**:
One connected region of RNV in one image. An image holds several, and a single
lesion may itself break into disconnected fronds.
_Avoid_: blob, object, tuft, region

**Frond**:
A thin, fern-like branch of a lesion. Names the shape that makes RNV hard:
boundaries are deeply concave rather than convex, and a lesion's fronds may be
disconnected from each other in the mask.
_Avoid_: tendril, filament, branch

**Inter-frond gap**:
Lesion pixels between fronds that the probability map leaves empty, usually a
concavity opening onto the lesion's outer edge, so hole filling cannot reach it.
The annotator counts it as lesion; the network, which responds to the vessels,
does not.
_Avoid_: hole, fragmentation, concavity (for the error itself)

### The two stages

**Localization**:
The upstream stage that finds *where* lesions are and emits points. Not part of
this context — its output is an input here.
_Avoid_: detection (as a stage name), finding

**Segmentation**:
This context. Converts points plus image into a binary mask per image.
_Avoid_: delineation, contouring

**Seed**:
A localization point, `[cx cy]` = (column, row), that this context segments
*from*. The umbrella term; when the distinction matters, say Detection or
Cluster Centre.
_Avoid_: point, marker, landmark, coordinate

**Detection**:
A raw localization point, before clustering. Many fall on one lesion — 62 on
`1115.png` — which is what lets seed-centred patches tile a lesion too large for
one patch.
_Avoid_: unclustered point, raw seed

**Cluster Centre**:
The single DBSCAN representative of a lesion's detections. One per lesion, and
the coordinates the per-seed ground truth is named after.
_Avoid_: clustered point, centroid, representative

**Claim**:
What a seed does to a connected component: it takes the component it falls
inside, or the nearest qualifying one. Components no seed claims are dropped.
This is the step that makes the output *localization-driven* rather than a plain
threshold.
_Avoid_: assign, match, select

**Near-miss**:
A detection landing just outside a lesion — within the seed tolerance. A
disagreement about where the boundary is, not a localization error, and labelled
with the lesion it missed.
_Avoid_: off-lesion point, boundary error, outlier

**Detector Error**:
A detection too far from any lesion to be a near-miss. Rejected from training.
Distinguished from a near-miss purely by distance, and the two separate cleanly.
_Avoid_: false positive (ambiguous — the segmenter has those too), bad point, outlier

### The data

**Domain**:
The set of images this configuration trains and scores on — every image with
ground truth across the active datasets. Currently 68.
_Avoid_: dataset (for the whole set), corpus, cohort

**Dataset**:
One source of images, masks and localization CSVs, with its own naming rule.
Two exist: **DRAC22** (41 images) and **TUH** (27 images).
_Avoid_: source, cohort, site

**Manifest**:
The table that resolves an image name to its image file, its mask, and its
dataset. The single join between an image and its ground truth — necessary
because DRAC22 pairs `458.png` with `458.png` while TUH pairs `nv (1).png` with
`gt (1).png`.
_Avoid_: index, catalogue, registry

**FilledGT**:
The whole-image ground-truth mask, one per image, with lesion interiors filled.
The scoring target.
_Avoid_: label, truth mask, annotation

**SegmentFilledGT**:
One ground-truth mask per cluster centre, named `<stem>-<x>-<y>.png`. DRAC22
only, keyed to cluster-centre coordinates.
_Avoid_: per-seed GT, split GT

**Acquisition Overlay**:
The burned-in device graphics on every TUH image — a magenta vertical and cyan
horizontal crosshair spanning the frame, plus a corner watermark. Deliberately
left in place.
_Avoid_: artifact, watermark (for the whole thing), annotation

### The pipeline

**Patch**:
A 512 px square crop centred near a seed, and the unit the network consumes.
Sized so a lesion fits: what the patch excludes, the network cannot predict.
_Avoid_: tile, window, crop, ROI

**Seed Heat-map**:
A Gaussian bump at the seed's position *within the patch*, appended as an input
channel. What makes the network seed-conditioned — answering "which lesion does
this point belong to" rather than "where is RNV here".
_Avoid_: attention map, seed channel, position encoding

**Probability Map**:
`P`, the network's per-pixel RNV probability over the whole image, stitched from
per-seed patches. Regions no seed reaches stay 0 by design. The measured
bottleneck of the whole pipeline.
_Avoid_: prediction, output, softmax, confidence map

**Seed Fallback**:
What a seed gets when the threshold left it with nothing — about 30% of seeds, for whom
it *is* the output. Currently **relax**: threshold `P` once at a low floor near the seed
and take the component under it; a small disk only when that finds nothing claimable.
_Avoid_: rescue, backup, default mask

**Fragment**:
A disconnected piece of the predicted mask within one lesion, such as an
isolated seed-fallback disk beside the lesion's main body. A prediction
failure, unlike a frond, which is a shape of the lesion itself.
_Avoid_: frond (for a prediction piece), island, blob

**Seed Hull**:
The region between seeds that lie close together, added to the mask after claiming. It
recovers inter-frond gaps where detections bracket them. It covers only the inside of the
seed cloud and never a lesion's outer rim.
_Avoid_: convex hull (a different, larger region), concave hull, envelope

### Experiments

**Arm**:
One named, comparable configuration: which datasets, which point set, how many
input channels, which model and output directory. Two runs are comparable only
within an arm, or across arms over the same domain.
_Avoid_: variant, mode, config, experiment

**Fold**:
One of the k held-out image sets in cross-validation. Deterministic and
area-stratified, so two arms over the same domain compare paired image by image.
_Avoid_: split, partition, bucket

**Out-of-sample**:
Scored by a model that saw the image in neither training nor checkpoint
selection. The only kind of score quotable here — `OutputNetwork="best-validation"`
means validation images are *not* held out.
_Avoid_: held-out, validation score, test score

**Evaluation Unit**:
What one scored row represents — a whole image (`"image"`) or a single seed
(`"segment"`). The two are not comparable, and only `"image"` is valid across
both datasets.
_Avoid_: metric level, granularity, scope

### Retired

Terms for the GVF + level-set stages deleted on 2026-09-25. They survive in
[results/](results/) and nowhere in the code.

**Force Field**:
`f = |∇P|`, the energy field the GVF diffuses. A *boundary* map, never `P`
itself: the field of `P` points at lesion cores and collapses the front.
_Avoid_: edge map, energy, gradient

**GVF Field**:
`(u,v)`, the Gradient Vector Flow diffusion of the force field. Extends a
force that exists only near a boundary across the whole image, giving capture
range and reach into concavities.
_Avoid_: flow, vector field, gradient field

**Edge Stopper**:
`g`, small where `P` has a strong boundary. The brake that keeps the front from
crossing an edge.
_Avoid_: stopping function, edge indicator, g-function

**Level Set**:
`φ`, the signed function whose zero crossing is the contour — negative inside.
One `φ` holds every lesion in an image and splits and merges natively, which is
why there is no per-lesion bookkeeping.
_Avoid_: phi (in prose), contour, snake, front
