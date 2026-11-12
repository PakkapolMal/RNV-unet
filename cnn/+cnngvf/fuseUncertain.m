function [Pf, fused] = fuseUncertain(Pk, Ps, cfg)
%CNNGVF.FUSEUNCERTAIN Bring the seedless P in on images whose seeded P is unsure.
%
%  [Pf, fused] = cnngvf.fuseUncertain(Pk, Ps, cfg)
%
%  Pk : seeded P;  Ps : seedless P (same size)
%  Pf : the map cnngvf.maskFromProb then runs on, unchanged
%
%  When cnngvf.seededUncertainty(Pk) >= Post.fuseCut, Pf is the pixel max of Pk
%  and Ps where Ps > Post.fuseTauS (0 elsewhere), over the whole image;
%  otherwise Pf = Pk. fuseCut = Inf is off. Claiming downstream still drops
%  any seedless blob no seed reaches. Experiment: docs/briefs/fusion-uncertainty.md.

fused = cnngvf.seededUncertainty(Pk) >= cfg.Post.fuseCut;   % Inf: never, -Inf: always
Pf = Pk;
if fused
    Pf = max(Pk, Ps .* (Ps > cfg.Post.fuseTauS));
end
end
