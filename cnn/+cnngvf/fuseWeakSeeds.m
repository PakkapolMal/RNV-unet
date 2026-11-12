function [Pf, status] = fuseWeakSeeds(Pk, Ps, seeds, cfg)
%CNNGVF.FUSEWEAKSEEDS Bring the seedless P in at the seeds the seeded P left empty.
%
%  [Pf, status] = cnngvf.fuseWeakSeeds(Pk, Ps, seeds, cfg)
%
%  Pk     : seeded P;  Ps : seedless P (same size)
%  Pf     : the map cnngvf.maskFromProb then runs on, unchanged
%  status : S x 1, 0 = not weak, 1 = weak and given seedless evidence, 2 = weak
%           and still left to relax
%
%  A weak seed is one not standing on Pk > Post.tau: exactly the seeds the relax
%  fallback serves. Within the relax window of each weak seed, cfg.Post.weakSeed:
%
%    "relax"          Pf = Pk (the current pipeline)
%    "seedlessFirst"  Pf = 1 on the seedless component under the seed
%                     (Ps > weakSeedTau, 8-connected through the seed pixel, at
%                     least MinArea); relax still serves the seeds without one
%    "maxRelax"       Pf = max(Pk, Ps where Ps > weakSeedTau), then relax on Pf
%
%  Window, rounding, connectivity and MinArea are relax's own
%  (cnngvf.seedFallback). Experiment: docs/briefs/fusion.md.
%
%  See also CNNGVF.SEEDFALLBACK, CNNGVF.MASKFROMPROB.

Pf = Pk;
S = size(seeds, 1);
status = zeros(S, 1);
if cfg.Post.weakSeed == "relax" || S == 0
    return;
end
[h, w] = size(Pk);
win = cfg.Post.relaxWindow;
BW = Pk > cfg.Post.tau;
Pss = Ps .* (Ps > cfg.Post.weakSeedTau);   % seedless evidence below tau_s is not used

for s = 1:S
    row = min(max(round(seeds(s, 2)), 1), h);
    col = min(max(round(seeds(s, 1)), 1), w);
    if BW(row, col)
        continue;   % not weak: the threshold already covers it
    end
    r0 = max(row - win, 1); r1 = min(row + win, h);
    c0 = max(col - win, 1); c1 = min(col + win, w);
    switch cfg.Post.weakSeed
        case "seedlessFirst"
            sub = Pss(r0:r1, c0:c1) > 0;
            sr = row - r0 + 1;
            sc = col - c0 + 1;
            comp = false(size(sub));
            if sub(sr, sc)
                comp = bwselect(sub, sc, sr, 8);
            end
            if nnz(comp) >= cfg.Post.MinArea
                blk = Pf(r0:r1, c0:c1);
                blk(comp) = 1;
                Pf(r0:r1, c0:c1) = blk;
                status(s) = 1;
            else
                status(s) = 2;
            end
        case "maxRelax"
            Pf(r0:r1, c0:c1) = max(Pf(r0:r1, c0:c1), Pss(r0:r1, c0:c1));
            status(s) = 1 + ~(Pf(row, col) > cfg.Post.tau);
        otherwise
            error('cnngvf:fuseWeakSeeds:mode', ...
                'Unknown Post.weakSeed "%s": use "relax", "seedlessFirst" or "maxRelax".', cfg.Post.weakSeed);
    end
end
end
