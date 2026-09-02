function [mask, perSeed] = maskFromProb(P, seeds, cfg)
%CNNGVF.MASKFROMPROB The reduced pipeline: a mask straight off the threshold.
%
%  [mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg)
%
%  P       : probability map from cnngvf.inferProbMap
%  seeds   : S x 2 [cx cy]
%
%  mask    : logical h x w, the union of every claimed lesion
%  perSeed : S x 1 cell, one mask per seed
%
%  Replaces stages 5-9 of cnngvf.segmentImage - the force field, the GVF
%  diffusion, the edge stopper and the DRLSE evolution - with the threshold on P
%  that cnngvf.initPhi already computes internally. Nothing here is new: the
%  threshold, the morphology and the claiming were all already in the pipeline,
%  and this is what is left when the level set between them is taken out.
%
%  The whole block it replaces was measured at +0.017 IoU over thresholding the
%  same P (../docs/UNCLUSTERED.md, "What the level set contributes"), and the
%  ceiling is structural rather than a tuning failure: the GVF field points at
%  the |grad P| ridge, which is the P ~ 0.5 iso-contour, so the level set is
%  largely an iso-contour finder for P.
%
%  The order below is the order the frozen "threshold P > 0.60" arm used -
%  imclose, imfill, seedMasks - so the two are directly comparable. Note it does
%  NOT include initPhi's imopen(disk 1): that exists to keep speckle out of a
%  level-set INITIALIZATION, which is a different job from producing the answer.
%
%  Returns the same [mask, perSeed] contract as the level-set path, so
%  run_batch's per-seed writing and run_eval's recovery of seeds from those
%  filenames both keep working unchanged.
%
%  See also CNNGVF.SEGMENTIMAGE, CNNGVF.INITPHI, CNNGVF.SEEDMASKS, RUN_ABLATE.

BW = P > cfg.Post.tau;

% A seed the threshold left with nothing gets a disk, from the same
% cnngvf.seedDisks the level set uses - same radius, same trigger - so that
% Post.mode is the only thing differing between the two paths. What it MEANS
% differs, though: for the level set a disk is an initialization that the
% evolution then grows or discards, while here nothing evolves and the disk
% survives into the output at its literal radius. That is why it is a factor in
% the ablation rather than a settled default.
if cfg.Post.diskFallback
    BW = BW | cnngvf.seedDisks(size(P), seeds, BW, cfg);
end

if cfg.Post.imclose
    BW = imclose(BW, cfg.Post.MorphElement);
end

% Unconditional: the ground truth in FilledGT has its lesion interiors filled,
% so an unfilled hole is scored as a false negative.
BW = imfill(BW, 'holes');

% Once, not twice. The level-set path claims at initialization AND after
% evolution; with no evolution there is nothing to initialize, so the two
% collapse into one call.
[perSeed, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);
end
