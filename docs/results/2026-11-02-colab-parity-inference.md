# PyTorch port, stages 0–2: inference with MATLAB's weights (2026-11-02)

Brief: [../briefs/colab-parity.md](../briefs/colab-parity.md), stages 0–2. Domain: the
standardized 69 images (DRAC22 42 + TUH 27), 5-fold CV. Each held-out image is inferred with
the weights of the MATLAB fold network that held it out
(`models/cv/DRACTUH_unclustered_fold<k>.mat`, trained 2026-10-23), then scored by
`run_cache` against `reference/2026-10-23-standardized-69.csv`. Nothing was trained.

**Prediction (from the brief):** stage 2 passes with max |ΔP| about 1e-5.
**Outcome:** the port is exact. The first stage-2 gate still **fails as written**, because
the reference `P` came from MATLAB's GPU, and MATLAB's GPU is 6e-4 away from its own CPU.

## Stage 1: fixtures, all pass

| check | result |
|---|---|
| `seedPatch` crop, 6 cases (all edges, jittered and half-pixel centres) | geometry and values exact; heat-map ≤ 1e-6 |
| dihedral (`fliplr`, `flipud`, `rot90` 1–3) | exact |
| `generalizedDice` | relative error ≤ 1e-7 |
| training seeds / validation seeds / patches per epoch, all 5 folds | equal to the counts the MATLAB runs recorded |

**`generalizedDice`.** The first port (weights 1/(ΣT)², no epsilon) was off by 1.2e-5
relative, and returned NaN on a patch with no lesion, where MATLAB returns ≈ 0. Fitting the
fixture gave MATLAB's form, with ε = `eps('single')` = 2⁻²³:

- weights: 1/((ΣT)² + ε)
- Dice: (2·num + ε)/(den + ε)

The documentation does not give this, which is why the brief asked for fixtures.

**Patch counts.** These also check the `cnngvf.seedLabels` refactor. Its labels, exported to
Python, reproduce the training-seed counts that `prepareData` logged before the refactor, for
every fold.

## Stage 2: MATLAB weights in PyTorch

Forward pass of fold 1, one real patch, PyTorch GPU against MATLAB CPU: max |ΔY| 4.2e-7.

| gate (brief) | threshold | measured | |
|---|---|---|---|
| max \|ΔP\| against the cached MATLAB `P`, 69 images | ≤ 1e-3 | **2.4e-3** (`1115.png`; median over images 6.5e-4) | **fail** |
| `run_cache` max \|dIoU\| against the reference | ≤ 1e-3 | **2.2e-4** (`487.png`); 31 of 69 images moved | pass |

`run_cache` on the PyTorch `P`, 69 images, 5-fold CV: mean IoU **0.504706**, against the
reference 0.504699. Median 0.5557, F1 0.6328, precision 0.6410, recall 0.6975.

**Why gate 1 fails.** One image (`nv (21).png`, fold 1), five ways:

| comparison | max \|ΔP\| |
|---|---|
| MATLAB GPU rerun vs the cached `P` | 3.0e-8 |
| MATLAB CPU vs MATLAB GPU | 3.6e-4 |
| **PyTorch GPU (fp32, TF32 off) vs MATLAB CPU** | **1.4e-7** |
| PyTorch GPU vs MATLAB GPU | 3.6e-4 |
| MATLAB GPU vs CPU, one patch | 6.1e-4 (median 5e-7) |

PyTorch reproduces MATLAB's CPU arithmetic. MATLAB's GPU path (RTX 5050) differs from both
by a few 1e-4, and that path wrote the reference cache. The differences are diffuse: about
1e-4 on 1–4 % of pixels. Over all 69 images they flip 271 pixels across `tau`, which is what
the 2.2e-4 dIoU reflects. A geometry or blending bug would show up as differences near 0.1,
in specific places.

The 1e-3 threshold assumed the reference `P` was exact to float32. It is not: MATLAB's own
GPU result is up to 2.4e-3 away from any exact computation.

## What this means

- Architecture, weight layout, patch geometry, heat-map, raised-cosine blending, loss and the
  cache format all match MATLAB to float32 precision.
- The only part of the trainer not yet tested is training itself (stage 3 onward).
- Whether to proceed past a gate that failed for this reason is decided outside this record.
  `colab/test_parity.py` keeps the brief's 1e-3 assertion unchanged, so it currently stops
  there.
