# PyTorch trainer on Colab: stages 1–3 pass on an L4, 290 s/epoch (2026-11-02)

Brief: [../briefs/colab-parity.md](../briefs/colab-parity.md), stages 1–3, run by the user
in `colab/train.ipynb` on a Colab NVIDIA L4. The bundle and code were the same as for the
local runs ([inference](2026-11-02-colab-parity-inference.md),
[local smoke](2026-11-02-colab-parity-smoke-local.md)). Domain: the standardized 69 images
(DRAC22 42 + TUH 27), 5-fold CV. The smoke run is fold 1, seed 0, 2 epochs. No IoU is
quoted.

**Prediction:** the same checks pass as locally, and the projected cost of stage 4 stays
under the brief's 120 CU for both seeds.

## Stages 1–2: all pass

Results are identical to the local run, apart from the comparison with MATLAB's CPU:

| check | local RTX 5050 | Colab L4 |
|---|---|---|
| forward pass vs MATLAB CPU, one patch | 3.6e-7 | 4.2e-7 |
| whole-image `P` vs MATLAB CPU, `nv (4).png` | 3.6e-7 | 4.2e-7 |
| max \|ΔP\| vs the MATLAB GPU cache, 69 images | 2.4e-3 | 2.4e-3 |

## Stage 3: all pass

| check | measured on L4 |
|---|---|
| fold 1 training seeds / patches per epoch / validation seeds | 599 / 2 397 / 11, equal to MATLAB's |
| zero-centre mean vs MATLAB | [0.3041, 0.0245] vs [0.3080, 0.0245]; max difference 0.0039 (gate 0.005) |
| stop after epoch 1, then rerun | resumed after epoch 1 |
| training loss | 0.4379 → 0.3777 (validation 0.4078 → 0.3537) |
| output network | best-validation epoch, 2 |
| held-out `P` | written for all 14 fold-1 test images |
| seconds per epoch (fp32, TF32 off, batch 2) | 301 and 279; mean **290** |

At 290 s/epoch, one 5-fold CV of 20 epochs takes about **8.1 h** on the L4. That is 1.9×
faster than the local GPU (565 s/epoch, 15.7 h).

## Budget gate (brief, stage 3)

Stage 4 is two CVs, about 16.2 GPU-hours, plus some overhead per session (copying the
bundle, inference, checkpoint writes to Drive). The 120 CU cap therefore holds for any L4
rate up to about **6.5 CU/h**, allowing ~10 % overhead. Colab shows the actual rate in its
resource panel.
