function [negA, negB] = negativeSeeds(gt, I, seeds)
%CNNGVF.NEGATIVESEEDS Where a training seed with an all-background target may sit.
%
%  [negA, negB] = cnngvf.negativeSeeds(gt, I, seeds)
%
%  gt    : h x w logical FilledGT mask
%  I     : h x w intensity, the first channel of cnngvf.loadChannels
%  seeds : S x 2 [cx cy], the image's detections
%
%  negA : S x 1 logical, detections more than 50 px from every lesion: the
%         detector's false positives
%  negB : h x w logical, pixels more than 50 px from every lesion with
%         intensity at least the 75th percentile of the non-lesion pixels:
%         normal vasculature
%
%  50 px is the low edge of the gap in off-lesion distances (docs/SPEC.md,
%  "Seed tolerance"): a detection 20-50 px out is neither positive nor
%  negative. Distance is to the BOUNDARY at the rounded seed, as in
%  cnngvf.seedLabels. Uses ground truth, so training only; run_export writes
%  both, and colab/rnvtorch.py samples them.
%  docs/briefs/2026-11-14-hard-neg-seeds.md.
%
%  See also CNNGVF.SEEDLABELS, RUN_EXPORT.

d = bwdist(gt);
[h, w] = size(gt);
row = min(max(round(seeds(:, 2)), 1), h);
col = min(max(round(seeds(:, 1)), 1), w);
negA = d(sub2ind([h w], row, col)) > 50;
negB = d > 50 & I >= prctile(I(~gt), 75);
end
