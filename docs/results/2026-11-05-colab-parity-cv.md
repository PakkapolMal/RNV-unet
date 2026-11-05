# PyTorch parity CV: within 0.005 IoU of MATLAB, but RNG 0 fails the ±0.02 CI gate (2026-11-05)

Brief: [../briefs/colab-parity.md](../briefs/colab-parity.md), stage 4. Domain: the
standardized 69 images (DRAC22 42 + TUH 27). Protocol: 5-fold CV with the reference's folds
and inner validation.

The PyTorch trainer ran twice on a Colab L4, with `--rng 0` and `--rng 1`. ("Seed A / seed B"
in the brief.) The configuration is unchanged: 20 epochs, minibatch 2, lr 1e-3, best
validation, fp32 with TF32 off. Each image's `P` comes from the fold that held it out. It
was scored by `run_cache` with the current post-processing, against
`reference/2026-10-23-standardized-69.csv` (MATLAB, 0.5047). The paired bootstrap uses
10 000 resamples over the 69 images.

**Prediction (from the brief):**
- mean ΔIoU within ±0.01 of MATLAB for each RNG;
- RNG 0 vs RNG 1 mean |Δ| about 0.005–0.015, with a per-image SD larger than the mean.

**Gate:** for **each** RNG, the 95 % CI of mean ΔIoU against the reference lies within
±0.02.

## Result

| run | mean IoU | median | F1 | precision | recall |
|---|---|---|---|---|---|
| MATLAB reference | 0.5047 | 0.5556 | 0.6328 | 0.6410 | 0.6975 |
| PyTorch RNG 0 | 0.5002 | 0.5267 | 0.6296 | 0.6431 | 0.6870 |
| PyTorch RNG 1 | 0.5010 | 0.5252 | 0.6298 | 0.6591 | 0.6738 |
| mean of the two `P` (secondary) | 0.5051 | 0.5408 | 0.6323 | 0.6591 | 0.6782 |

| paired difference | mean | 95 % CI | per-image SD | images with \|Δ\| > 0.05 |
|---|---|---|---|---|
| RNG 0 − MATLAB | −0.0045 | [−0.0239, +0.0167] | 0.0865 | 25 |
| RNG 1 − MATLAB | −0.0037 | [−0.0186, +0.0120] | 0.0663 | 20 |
| **RNG 0 − RNG 1 (noise floor)** | −0.0008 | [−0.0152, +0.0140] | **0.0623** | 26 |
| ensemble − MATLAB | +0.0004 | [−0.0165, +0.0189] | 0.0761 | 22 |
| ensemble − mean of its members | +0.0045 | [−0.0040, +0.0120] | 0.0342 | 6 |

**Gate: fails as written.**
- RNG 0's CI reaches −0.0239, past −0.02.
- RNG 1 passes.

**Prediction: holds.**
- Both means are within ±0.005 of MATLAB.
- RNG 0 − RNG 1 has mean −0.0008 with a per-image SD of 0.062.

Per-image IoU correlates with MATLAB at 0.931 (RNG 0) and 0.960 (RNG 1), and between the
two RNGs at 0.964.

Per dataset, the breakout of the pooled folds (not a protocol):

| | n | MATLAB | RNG 0 | RNG 1 | ensemble |
|---|---|---|---|---|---|
| DRAC22 | 42 | 0.4357 | 0.4343 | 0.4385 | 0.4344 |
| TUH | 27 | 0.6121 | 0.6026 | 0.5982 | 0.6150 |

Neither dataset's mean Δ comes near the brief's ±0.04 flag: the largest is TUH, RNG 1,
at −0.014.

## Why RNG 0 fails: one stalled fold, on top of a wide noise floor

| fold | n | MATLAB | RNG 0 | RNG 1 |
|---|---|---|---|---|
| 1 | 14 | 0.5440 | 0.5364 | 0.5144 |
| 2 | 14 | 0.5313 | 0.5315 | 0.5465 |
| 3 | 14 | 0.5116 | 0.5032 | 0.5231 |
| 4 | 14 | 0.4724 | **0.4058** | 0.4545 |
| 5 | 13 | 0.4611 | 0.5258 | 0.4636 |

**RNG 0's fold 4 stalled.**
- Its training loss started at 0.629 and plateaued at about 0.39 from epoch 3. Every other
  fold of either RNG ends between 0.28 and 0.31.
- Best validation picked epoch 7.
- That fold scores 0.067 below MATLAB, and 2 of RNG 0's 3 largest per-image losses (`576`,
  `559`) are in it.
- RNG 1's fold 4 also started high (0.761) but recovered to 0.307, and so did its fold 5
  (0.604 → 0.300).

Starting high therefore happens on some folds and RNGs. Staying high, once in 10 fold
trainings here, is what lost RNG 0 the gate. Whether MATLAB's trainer ever stalls like this
is unknown, because the 2026-10-23 run kept no training curves.

**The margin sits at the noise floor.** Two PyTorch trainings that differ only in their
random draws already give a CI half-width of about 0.015. That leaves about 0.005 for any
real offset from MATLAB before a ±0.02 CI gate fails. The brief chose ±0.02 before this
noise was known.

## What this measures, beyond the gate

**The training noise floor** (levers analysis R1), for 69 images, 5-fold CV, one training
per arm:
- per-image SD of the difference between two retrains: **0.062**;
- 95 % CI half-width of the mean difference: **about ±0.015**.

A lever compared on a single retraining per arm cannot be told from noise unless it
moves the mean by more than about 0.015. The levers analysis predicts +0.01 to +0.03 for
R2 and R3.

**The 2-model ensemble** (secondary, not adopted):
- 0.5051, which is +0.0045 over its members' mean; CI [−0.004, +0.012] includes 0;
- the per-image spread halves (SD 0.034 against its members).

**Cost:** about 280–325 s/epoch on the L4, about 8.4 h per CV.
