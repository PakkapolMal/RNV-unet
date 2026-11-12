function [mask, perSeed] = cachedMask(cacheDir, img, seeds, cfg)
%CNNGVF.CACHEDMASK One image's final mask from the cached CV P, as it is scored.
%
%  [mask, perSeed] = cnngvf.cachedMask(cacheDir, img, seeds, cfg)
%
%  cacheDir : the seeded CV P cache;  seeds : the image's S x 2 points
%
%  With Post.seedlessCache set: no points -> stage 9 (cnngvf.noPointMask);
%  otherwise stage 4a (cnngvf.fuseSeedless) before cnngvf.maskFromProb. Every
%  read goes through cnngvf.probCache's leakage guard. run_cache and
%  run_figures both call this, so a figure always shows the scored mask.

seedless = strlength(cfg.Post.seedlessCache) > 0;
if isempty(seeds) && seedless
    % No points: the seedless network's mask (docs/results/2026-11-12-seedless-fallback.md).
    c = cnngvf.probCache("read", cfg.Post.seedlessCache, img, cfg);
    [mask, perSeed] = deal(cnngvf.noPointMask(double(c.P), cfg), {});
    return;
end
P = double(cnngvf.probCache("read", cacheDir, img, cfg).P);
if seedless
    % Stage 4a (docs/results/2026-11-12-fusion-confirm.md).
    cs = cnngvf.probCache("read", cfg.Post.seedlessCache, img, cfg);
    P = cnngvf.fuseSeedless(P, double(cs.P), cfg);
end
[mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg);
end
