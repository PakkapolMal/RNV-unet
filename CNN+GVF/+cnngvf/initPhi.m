function [phi0, BW0] = initPhi(P, seeds, cfg)
%CNNGVF.INITPHI Build the DRLSE level-set function from the probability map.
%
%  [phi0, BW0] = cnngvf.initPhi(P, seeds, cfg)
%
%  P    : probability map from cnngvf.inferProbMap
%  phi0 : signed level-set function, cfg.Drlse.c0 * (1 - 2*BW0). Negative
%         inside the lesion, positive outside - the sign convention the
%         balloon term in cnngvf.drlse assumes.
%  BW0  : the coarse foreground phi0 encodes
%
%  Thresholding at cfg.Drlse.tau and keeping the seeded components gives a
%  robust start: GVF's long capture range then tolerates an imperfect one.
%  When a seed's neighbourhood has nothing above tau - the CNN missed it, which
%  17 training images make likely - that seed still gets a small disk, so the
%  balloon and GVF terms have something to grow rather than silently dropping
%  the lesion.
%
%  See also CNNGVF.DRLSE, CNNGVF.SEEDMASKS.

BW0 = P > cfg.Drlse.tau;
BW0 = imopen(BW0, strel('disk', 1));

% minArea 0: a small blob here is a lesion that has not grown yet.
[~, BW0] = cnngvf.seedMasks(BW0, seeds, 0, cfg.Post.ClaimRule);

if cfg.Post.diskFallback
    BW0 = BW0 | cnngvf.seedDisks(size(P), seeds, BW0, cfg);
end

phi0 = cfg.Drlse.c0 * (1 - 2 * double(BW0));
end
