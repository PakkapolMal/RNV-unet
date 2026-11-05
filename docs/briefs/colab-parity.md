---
title: PyTorch Trainer Parity on Colab
description: Port U-Net training and inference to PyTorch so longer training can run on Colab, and prove the port reproduces the MATLAB network before any lever is tried on it. Stages, fixtures, prediction and gate written before any run.
kind: implementation-brief
audience: Claude Code
target: RNV-Claude/colab (new, Python + PyTorch) and RNV-Claude/cnn (one export runner)
depends-on: results/2026-10-23-standardized-cv.md, research_analysis/rnv-score-levers-2026-10-29.md
status: executed 2026-11-05
---

# PyTorch Trainer Parity on Colab

> **Stage 2 gate, 2026-11-02.** max |ΔP| ≤ 1e-3 against the cached MATLAB `P` failed
> (2.4e-3). The cause is MATLAB's GPU arithmetic, not the port: PyTorch matches MATLAB's
> CPU `P` to 1.4e-7, and the `run_cache` gate passed (max |dIoU| 2.2e-4). The user accepted
> the failure and stage 3 proceeds. `test_parity.py` now checks `P` against MATLAB's CPU
> result. Record: [../results/2026-11-02-colab-parity-inference.md](../results/2026-11-02-colab-parity-inference.md).
>
> **Stage 4 gate, 2026-11-05.** The ±0.02 CI gate failed for RNG 0 (CI reaches −0.024, after
> one stalled fold) and passed for RNG 1. Both means are within 0.005 IoU of MATLAB. The
> user accepted parity: the PyTorch trainer is now the trainer, and each later arm trains
> once. Record: [../results/2026-11-05-colab-parity-cv.md](../results/2026-11-05-colab-parity-cv.md).

## Why

Every lever left is on the training side (R1–R3, R5, R6, R8 of the 2026-10-29 levers
analysis). R2 alone doubles the training budget to about 28 h per CV, which this machine
cannot run. Colab can, but Colab has no MATLAB. So the U-Net trainer moves to PyTorch.

A different framework is a different trainer. Its initialisation, RNG and kernels all differ,
so the CV number will move. Before any lever is measured on the port, this brief has to
show two things:

1. **Inference is exact.** MATLAB's own trained weights, run in PyTorch, give the cached `P`.
2. **Training is equivalent.** A PyTorch-trained CV lands within a pre-registered margin of
   the reference.

Also measured: the **training noise floor**. It has never been measured (levers analysis,
R1), and every later paired comparison needs it.

## Scope

**Ported:** patch sampling (`prepareData` / `seedPatch`), the U-Net, the loss, training
(`trainNet`) and `P` inference (`inferProbMap`).

**Not ported:** loading, labelling, folds, post-processing and scoring stay in MATLAB.
- MATLAB exports the inputs: channels, labels, seeds, folds and inner validation.
- Python writes `P` back in the `probCache` format.
- `run_cache` scores it, against the reference, with the leakage guard unchanged.

Python never reads `datasets/`, so its quirks (TUH `.jpg` names, the `rgb2gray` weights)
cannot diverge.

**Not changed:** anything in the configuration. The run uses 20 epochs, `miniBatch` 2,
lr 1e-3 constant, `best-validation`, `perSeed` 4, `balanceByImage`, jitter ±32 and the
dihedral augmentation only. Using a larger GPU's memory for a bigger batch is a lever, not
parity.

## What the MATLAB network is

Read from `models/cv/DRACTUH_unclustered_fold1.mat`, 2026-11-02. This is the spec the
PyTorch model must match.

| part | value |
|---|---|
| input | 512×512×2, `zerocenter` with a per-channel mean that `trainnet` computes from the training data (fold 1: [0.3080, 0.0245]) |
| encoder | 4 stages of conv3×3 → ReLU → conv3×3 → ReLU, padding 1, with 64/128/256/512 filters, each followed by maxpool 2×2. No BatchNorm. |
| dropout | p = 0.5 after encoder stage 4, before its pool, and again after the bridge |
| bridge | conv3×3 1024 → ReLU → conv3×3 1024 → ReLU → dropout |
| decoder | 4 stages: transposed conv 2×2 stride 2 → ReLU → concat [encoder skip, upsampled] → 2 × (conv3×3 → ReLU) |
| head | conv1×1 to 2 channels → softmax. `P` is channel 2. |
| size | 46 learnables, 31 031 234 parameters |
| optimiser | Adam: β 0.9 / 0.999, ε 1e-8, **L2 regularisation 1e-4 added to the gradient** (MATLAB's default; this is not AdamW), no gradient clipping, no LR schedule |
| init | MATLAB defaults (Glorot-uniform weights, zero bias). PyTorch's defaults differ and must be overridden. |
| loss | `1 - mean(generalizedDice(Y, T))`, averaged over the batch |
| selection | validation loss once per epoch on the inner-validation patches (no jitter, no augmentation); keep the best epoch |
| fold 1 | 599 training seeds, 2 397 patches per epoch; inner validation `193.png` and `429.png`, 11 seeds |

Each fold's `.mat` file stores its `info`, so the patch counts for every fold are known
without a rerun.

ONNX export is not installed, so the weights move as a plain `.mat` of numeric arrays.

## Stages and gates

The stages run in order. Each one must pass before the next starts.

### Stage 0: export (local MATLAB, minutes)

A new runner, `run_export`, writes one bundle in `.mat` v7, which `scipy.io.loadmat` reads:

- **Per image:**
  - `chans`: the output of `loadChannels`, as single;
  - the FilledGT label matrix;
  - the seeds;
  - each seed's component id after `seedTolerance` (0 = dropped from training).
- **Per fold:** the test images, the inner-validation images, the MATLAB `info` (patch
  counts) and the trained MATLAB weights.
- **Fixtures**, each written by the MATLAB function itself:
  - about 20 `seedPatch` crops with fixed centres, including centres near an edge (zero
    padding);
  - the label crop and every dihedral transform applied to one patch;
  - `generalizedDice` on a few fixed `Y` / `T` pairs, including a patch with no lesion.

### Stage 1: fixtures (local Python or Colab CPU, minutes)

`colab/test_parity.py` checks these against the fixtures:

- crop, heat-map and label: exact (heat-map ≤ 1e-6);
- dihedral transforms: exact;
- loss: ≤ 1e-6;
- patch count per fold: equal to MATLAB's `info.patchesPerEpoch`.

**Gate:** every check passes. The fixtures settle conventions such as the generalized-Dice
denominator and weight layout. Read the fixtures, not the documentation.

### Stage 2: inference parity, MATLAB weights (Colab GPU, minutes)

Load each fold's MATLAB weights into the PyTorch model. Run inference on that fold's test
images, with TF32 off and fp32. Write `P` in the `probCache` format, with `meta.fold` set.
Then run `run_cache(cfg, <that dir>)` locally.

**Gate:**
- max |ΔP| ≤ 1e-3 against the MATLAB cached `P` on every image;
- `run_cache` max |dIoU| ≤ 1e-3 against `2026-10-23-standardized-69.csv` (69 images,
  5-fold CV).

This proves the architecture, the weight layout, the patch geometry, the raised-cosine
blending and the cache format, all without any training. Training is the only part left
untested after this stage.

### Stage 3: one-fold smoke (Colab GPU, under 1 h)

Train fold 1 for 1 epoch, then resume from the checkpoint for a second epoch.

**Checks:**
- the computed zero-centre mean is within 0.005 of MATLAB's fold-1 mean;
- the loss falls;
- resume reproduces the epoch count and the best-validation state;
- the seconds per epoch on the chosen GPU (T4, L4 or A100).

**Budget gate:** project the CU cost of stage 4. If it exceeds 120 CU for both seeds
together, stop and report, and run seed B later. If GPU memory allows, run the two seeds as
concurrent processes on one GPU, since batch 2 leaves most of a 24 GB card idle.

### Stage 4: parity CV (Colab GPU, two training seeds)

Run the full 5-fold CV twice, with `torch` seeds A and B and the same folds and inner
validation. Run one fold per Colab session. Checkpoint every epoch to Drive (model,
optimiser, best-validation state, RNG, epoch) so a disconnect costs at most one epoch. Score
both runs locally with `run_cache`.

## Prediction (written before any run)

- **Stage 2:** passes with max |ΔP| about 1e-5. Any failure is a layout or convention bug.
- **Stage 4:**
  - mean ΔIoU (PyTorch − MATLAB reference, 69 images, 5-fold CV) within ±0.01 for each
    seed;
  - per-image |ΔIoU| is large on small lesions, as retraining always does there;
  - seed A vs seed B mean |Δ| is about 0.005–0.015;
  - the per-image SD of A − B is larger than its mean.

## Gate (stage 4)

**Parity passes** when, for **each** seed, the paired-bootstrap 95 % CI of mean ΔIoU against
the reference lies within **±0.02**:
- 69 images, 5-fold CV;
- 10 000 resamples over images.

Per dataset, report the mean Δ (DRAC22 42 and TUH 27). A dataset whose mean Δ is beyond
±0.04 is flagged but does not fail the gate.

**Why ±0.02.**
- Parity has to catch a porting bug. A bug usually costs far more than 0.02.
- Later levers are judged PyTorch vs PyTorch, paired. A small, honest offset from MATLAB
  does not bias them.
- The margin is not tuned on these images. It is fixed here.

**If it fails:**
- **Both seeds low by more than 0.02:** a porting bug. Return to stage 1 and add a fixture
  for whatever differs. Do not tune anything.
- **Seeds disagree with each other by more than either one disagrees with MATLAB:** the
  noise floor dominates and the gate is uninformative. Record that and ask before going on.

## Also reported, not gated

- **Noise floor:** the paired-bootstrap CI and the per-image SD of seed A − seed B. This is
  R1's main deliverable, and it sets the threshold every later lever must clear.
- **2-model ensemble:** the mean of A's and B's `P`, scored by `run_cache`. Report it as a
  secondary figure. It is a new configuration and is not adopted by this brief.
- **Runtime and CU cost** per fold, and the GPU used.

## Files

| path | what |
|---|---|
| `cnn/run_export.m` | stage 0 bundle and fixtures |
| `colab/rnvtorch.py` | model, sampler, loss, train loop with checkpoint and resume, inference, `probCache` writer |
| `colab/test_parity.py` | stage 1 and stage 2 checks, plain `assert` |
| `colab/train.ipynb` | Colab driver: mount Drive, copy the bundle to local disk, train one fold, infer |

Docs in the same commits:
- `cnn/README.md`: layout and usage;
- `docs/SPEC.md`: a "Training on Colab" section, plus a note that the `P` cache can come
  from either trainer.

## After the gate

- **Pass:** the PyTorch trainer becomes the training path for the lever series. R2 is the
  first lever, judged against seed A paired, and must clear the noise floor. The MATLAB
  reference stays adopted until a lever beats it. Adopting a PyTorch-trained configuration
  means a new `docs/results/reference/*.csv` and README table, per CLAUDE.md.
- **Fail:** record it and stop. The MATLAB trainer stays the only trainer.

Either way, the outcome goes in `docs/results/YYYY-MM-DD-colab-parity.md`. This brief is
then frozen with a banner that links to it.
