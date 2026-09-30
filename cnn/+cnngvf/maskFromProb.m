function [mask, perSeed] = maskFromProb(P, seeds, cfg)
%CNNGVF.MASKFROMPROB Turn the probability map into the final mask.
%
%  [mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg)
%
%  P       : probability map from cnngvf.inferProbMap
%  seeds   : S x 2 [cx cy]
%
%  mask    : logical h x w, the union of every claimed lesion
%  perSeed : S x 1 cell, one mask per seed
%
%     P > Post.tau -> seed fallback -> imclose -> imfill -> seedMasks
%
%  This replaced a GVF + DRLSE level set that measured -0.0017 IoU (not
%  significant) against it on identical P; see docs/results/.
%
%  See also CNNGVF.SEGMENTIMAGE, CNNGVF.SEEDFALLBACK, CNNGVF.SEEDMASKS.

BW = P > cfg.Post.tau;

% A seed the threshold left with nothing still gets an answer; for ~30% of
% seeds this is the output.
BW = BW | cnngvf.seedFallback(P, seeds, BW, cfg);

BW = imclose(BW, cfg.Post.MorphElement);

% FilledGT has its lesion interiors filled, so an unfilled hole would score as
% a false negative.
BW = imfill(BW, 'holes');

[perSeed, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);

% Experiment, docs/briefs/seed-hull.md: fill between linked seeds.
[perSeed, mask] = cnngvf.seedHull(perSeed, mask, seeds, cfg);
end
