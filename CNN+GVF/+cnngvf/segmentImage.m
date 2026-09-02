function [mask, perSeed, contours, img, P] = segmentImage(src, seeds, net, cfg)
%CNNGVF.SEGMENTIMAGE Segment every RNV lesion in one image (Approach 1).
%
%  [mask, perSeed, contours, img, P] = cnngvf.segmentImage(imgPath, seeds, net, cfg)
%
%  seeds    : S x 2 [cx cy] localization points for this image
%  net      : dlnetwork from cnngvf.trainNet
%
%  mask     : logical h x w, the union of every claimed lesion
%  perSeed  : S x 1 cell, one mask per seed. This is the unit the evaluation
%             scores, against RNV-DRAC22GT/SegmentFilledGT/<stem>-<x>-<y>.png
%  contours : cell of N x 2 [row col] boundaries, ready for rnv.saveOverlay
%  img      : struct from rnv.loadImage, so callers can draw without re-reading
%  P        : the CNN probability map, returned for debugging the force field
%
%  Stages, per ../docs/spec.md Approach 1:
%     channels -> P -> f -> GVF field -> phi0 -> DRLSE -> cleanup
%
%  Unlike RNV-Claude/main, seeds are not segmented independently and OR'd. One
%  level set evolves over the whole image and the seeds decide afterwards
%  which components survive, which is what lets a lesion split into fronds.
%
%  See also CNNGVF.INFERPROBMAP, CNNGVF.DRLSE, RUN_BATCH.

[img, chans] = cnngvf.loadChannels(src, cfg);

P       = cnngvf.inferProbMap(chans, seeds, net, cfg);
f       = cnngvf.buildForce(P, img.I, cfg);
[u, v]  = cnngvf.gvfField(f, cfg);
phi0    = cnngvf.initPhi(P, seeds, cfg);
BW      = cnngvf.drlse(P, phi0, u, v, cfg);

BW = imclose(BW, cfg.Post.MorphElement);
BW = imfill(BW, 'holes');    % the ground truth in RNV-DRAC22GT/FilledGT is filled

[perSeed, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);

contours = bwboundaries(mask, 'noholes');
end
