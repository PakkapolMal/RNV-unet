function [mask, perSeed, contours, img, P] = segmentImage(src, seeds, net, cfg)
%CNNGVF.SEGMENTIMAGE Segment every RNV lesion in one image.
%
%  [mask, perSeed, contours, img, P] = cnngvf.segmentImage(imgPath, seeds, net, cfg)
%
%  seeds    : S x 2 [cx cy] localization points for this image
%  net      : dlnetwork from cnngvf.trainNet
%
%  mask     : logical h x w, the union of every claimed lesion
%  perSeed  : S x 1 cell, one mask per seed
%  contours : cell of N x 2 [row col] boundaries, ready for cnngvf.saveOverlay
%  img      : struct from cnngvf.loadChannels, so callers can draw without re-reading
%  P        : the CNN probability map
%
%  Stages:  channels -> P -> threshold + fallback -> cleanup -> seed claiming
%
%  Seeds are not segmented independently and OR'd: one mask is built for the
%  whole image and the seeds decide which components survive, which is what
%  lets a lesion split into fronds.
%
%  See also CNNGVF.INFERPROBMAP, CNNGVF.MASKFROMPROB, RUN_BATCH.

[img, chans] = cnngvf.loadChannels(src, cfg);
P = cnngvf.inferProbMap(chans, seeds, net, cfg);
[mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg);
contours = bwboundaries(mask, 'noholes');
end
