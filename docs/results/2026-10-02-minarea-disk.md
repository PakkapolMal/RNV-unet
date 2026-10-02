---
title: MinArea × fallback disk radius, cross-fitted
kind: measurement-record
dataset: DRAC22 + TUH (68)
updated: 2026-10-02
---

# MinArea × fallback disk radius, cross-fitted

Brief, prediction and gate: [../briefs/minarea-disk.md](../briefs/minarea-disk.md), committed
before any run. Follows [2026-10-02-minarea](2026-10-02-minarea.md), and corrects its account
of why `421.png` and `217.png` fail (see below).

**Question.** Does any (`MinArea`, `seedRadius`) pair beat the default (500, 15) on the current
configuration, including a radius of 0?

**Protocol.** 68 images (DRAC22 41, TUH 27), cached 5-fold CV `P` through `run_cache`,
whole-image scoring. Current configuration: `tau` 0.5, relax 0.20 at 128 px, boundary
claiming, `hullLink` 128 with the fill-any-triangle rule. Grid `MinArea` ∈ {250, 500, 750,
1000} × `seedRadius` ∈ {0, 10, 15, 20, 30}, 20 cells. For each fold, choose the cell with
the best mean IoU on the other four folds and score it on the held-out fold. Ties within
0.001 go to the default, then the smallest radius, then the smallest `MinArea`. Paired per
image against the default, which reproduces `2026-09-30-hull-128.csv` to 5e-16. Percentile
bootstrap, 10 000 resamples, `rng(20260917)`, 95 % CI. Script: a scratch `.m` file, not
committed.

## Result: every fold chooses (500, 15). The gate fails and the default stays.

Cross-fitted IoU 0.5207, identical to the default on all 68 images (Δ 0, 0 up / 0 down).

| gate | result | |
|---|---|---|
| 1. ΔIoU 95 % CI lower bound > 0 | 0.0000 | fail |
| 2. precision guard, three smallest GT-area quintiles | 0.5291 → 0.5291 | pass |
| 3. ≥ 4 of 5 folds choose the same cell | 5 of 5 on (500, 15) | pass, on the default |

Without the tie rule the folds would have chosen (250, 10), (250, 10), (500, 30), (500, 20) and
(500, 20). Each beat the default on its training folds by at most 0.0009, so all were tied.

### Surface (tuned on the images it scores; not a result)

Mean IoU, 68 images:

| `MinArea` \ r | 0 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| 250 | 0.5206 | 0.5207 | 0.5207 | 0.5206 | 0.5200 |
| **500** | 0.5204 | 0.5205 | ***0.5207*** | 0.5207 | 0.5204 |
| 750 | 0.4951 | 0.4951 | 0.4952 | 0.4954 | 0.4950 |
| 1000 | 0.4970 | 0.4971 | 0.4971 | 0.4974 | 0.4971 |

At `MinArea` 500, per radius:

| r | IoU | precision | recall | DRAC22 | TUH | images changed | ΔIoU vs r 15, 95 % CI |
|---|---|---|---|---|---|---|---|
| 0 | 0.5204 | 0.6281 | 0.7750 | 0.4835 | 0.5764 | 9 | −0.0002 [−0.0011, +0.0004] |
| 10 | 0.5205 | 0.6280 | 0.7751 | 0.4835 | 0.5766 | 9 | −0.0002 [−0.0009, +0.0004] |
| **15** | **0.5207** | 0.6268 | 0.7762 | 0.4831 | 0.5777 | — | — |
| 20 | 0.5207 | 0.6257 | 0.7771 | 0.4826 | 0.5785 | 9 | +0.0000 [−0.0008, +0.0009] |
| 30 | 0.5204 | 0.6228 | 0.7791 | 0.4810 | 0.5803 | 9 | −0.0003 [−0.0031, +0.0024] |

**On this `P` the disk is close to inert.** At `MinArea` 500 it changes only 9 of 68 images,
and every radius from 0 to 30 lies within 0.0003 of the default. A larger disk trades
precision for recall (0.6281 / 0.7750 at r 0 → 0.6228 / 0.7791 at r 30), and the devices
split again: TUH rises with the radius (0.5764 → 0.5803) and DRAC22 falls (0.4835 → 0.4810).
The relax floor showed the same split ([2026-09-30 paired](2026-09-30-relax-floor-015-paired.md)).

**Unclaimable disks behave like r 0, nearly.** In every cell where πr² < `MinArea`, the score
is within 0.0002 of r 0 at the same `MinArea`, with 7 images differing by at most 0.0054.
That is where a disk joins a neighbouring component through `imclose` and gets claimed.

### Correction: the `MinArea` cliff is not about the disk

[2026-10-02-minarea](2026-10-02-minarea.md) put the loss above 700 px down to a ~700 px
component "about the disk's area". It is not the disk. Both seeds **fired**:

| image | `P` at the seed | its `P > 0.5` component | final mask at 500 |
|---|---|---|---|
| `421` | 1.000 | 695 px | 699 px |
| `217` | 1.000 | 698 px | 708 px |

A seed standing on the thresholded foreground never reaches the fallback, so no radius
reaches them. At `MinArea` ≥ 750 both stay at IoU 0 for every r, including 20 and 30. At
r 0 and `MinArea` 500 they keep their 0.871 / 0.836. The cliff is the classifier's own small
component falling under `MinArea`. That `708 ≈ 707`, the disk's area, was a coincidence.
**The binding constraint on `MinArea` is the smallest lesion the network fires on (~700 px
here), not the disk.**

### Against the prediction

| # | predicted | measured | |
|---|---|---|---|
| 1 | (500, r 0): −0.002 to −0.006; `217` → 0 | −0.0002; `217` unchanged at 0.836 | **wrong**: the disk barely acts, and `217` is a fired seed |
| 2 | r 20 within ±0.003; r 30 −0.001 to −0.005 | +0.0000; −0.0003 | r 20 right; r 30 right direction, 3× smaller |
| 3 | `MinArea` ≥ 750 with r ≥ 20 partly rescues `421` / `217` | both 0 in every cell | **wrong**, same cause as 1 |
| 4 | cells where the disk is too small to claim ≈ r 0, not exactly | within 0.0002, 7 images differ | right |
| 5 | cross-fitted Δ −0.004 to +0.002, gate fails, default stays | Δ 0, 5 / 5 folds on the default | right |

## Conclusion

(500, 15) stays. The disk radius is close to inert under relax, because relax serves most
seeds that miss the threshold. `MinArea` is limited by the network's smallest fired
components, not by the disk. Neither setting is a lever on this `P`.
