# PyTorch trainer, stage-3 smoke run on the local GPU (2026-11-02)

Brief: [../briefs/colab-parity.md](../briefs/colab-parity.md), stage 3. This run checks the
code, not the score. Domain: the standardized 69 images (DRAC22 42 + TUH 27). The run is
fold 1 of the 5-fold CV, seed 0, with 2 epochs instead of 20. No IoU is quoted: two epochs
measure nothing about segmentation.

The brief puts stage 3 on Colab, where it also sets the CU budget for stage 4. This run is
on the RTX 5050 Laptop GPU (8 GB) and covers the code. Colab's seconds per epoch is still
to be measured.

**Prediction (from the brief):**
- the zero-centre mean is within 0.005 of MATLAB's;
- the loss falls;
- resuming reproduces the epoch count and the best-validation state.

## Result: every check passes

| check | measured |
|---|---|
| fold 1 training seeds / patches per epoch / validation seeds | 599 / 2 397 / 11, equal to MATLAB's |
| zero-centre mean, PyTorch vs MATLAB | [0.3042, 0.0245] vs [0.3080, 0.0245]; max difference 0.0038 (gate 0.005) |
| stop after epoch 1, then rerun | resumed after epoch 1; log epochs [1, 2] |
| training loss | 0.4217 → 0.3746 (validation 0.4256 → 0.3562) |
| output network | best-validation epoch, 2 |
| held-out `P` | written for all 14 fold-1 test images, in the `probCache` format |
| seconds per epoch, fp32, TF32 off, batch 2 | 593 and 538 (mean 565) |

At 565 s/epoch, one 5-fold CV of 20 epochs takes about 15.7 h on this GPU. MATLAB's
2026-10-23 CV took about 15 h on the same GPU, judging from the fold networks' save times.

**The mean differs by 0.0038.** Both trainers compute it over one pass of their own
training patches. Those patches are drawn with replacement and with random jitter, so the
two runs see different patches.

**Same seed, different epoch 1.** An earlier attempt, which stopped on a resume bug fixed
since, logged training loss 0.4202 at epoch 1 with the same seed. cuDNN autotuning and
atomic accumulation are not deterministic. That is acceptable here, because the parity gate
is statistical. It does mean that two runs with one seed are not identical.
