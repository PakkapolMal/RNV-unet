function [D, dropped] = seedFallback(P, seeds, BW, cfg)
%CNNGVF.SEEDFALLBACK What to output for a seed the threshold left with nothing.
%
%  D = cnngvf.seedFallback(P, seeds, BW, cfg)
%
%  P     : probability map
%  BW    : the thresholded foreground; a seed already standing on it is skipped
%  D     : logical, what to OR into BW
%  dropped : S x 1 logical, seeds relax found nothing for and, with
%            cfg.Post.relaxDisk = false, dropped instead of given the disk
%
%  On the 68-image domain 258 of 874 seeds never produce P > tau, so for them
%  this function IS the answer.
%
%  cfg.Post.fallbackMode
%
%    "relax"  threshold P once at cfg.Post.relaxFloor inside a window of
%             cfg.Post.relaxWindow px around the seed, and take the component
%             under the seed: a lesion the CNN saw weakly gets its own outline
%             instead of a circle. The window stops a low floor flooding a broad
%             plateau. One threshold, not a ladder down to the first component
%             that clears MinArea - a ladder stops at an arbitrary contour and
%             lands smaller than the disk. When nothing claimable is found, the
%             seed gets the disk, so relax never removes the recall floor.
%
%    "disk"   a disk of cfg.Post.seedRadius at the seed. The frozen arms.
%
%  See also CNNGVF.SEEDDISKS, CNNGVF.MASKFROMPROB.

switch cfg.Post.fallbackMode
    case "disk"
        D = cnngvf.seedDisks(size(P), seeds, BW, cfg);
        dropped = false(size(seeds, 1), 1);

    case "relax"
        [D, dropped] = relaxFallback(P, seeds, BW, cfg);

    otherwise
        error('cnngvf:seedFallback:badMode', ...
            'Unknown Post.fallbackMode "%s". Use "disk" or "relax".', ...
            cfg.Post.fallbackMode);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [D, dropped] = relaxFallback(P, seeds, BW, cfg)
[h, w] = size(P);
D = false(h, w);
dropped = false(size(seeds, 1), 1);

win    = cfg.Post.relaxWindow;
floor_ = cfg.Post.relaxFloor;

for s = 1:size(seeds, 1)
    row = min(max(round(seeds(s, 2)), 1), h);
    col = min(max(round(seeds(s, 1)), 1), w);
    if BW(row, col)
        continue;   % already covered by a thresholded component
    end

    % Window, clipped at the image border.
    r0 = max(row - win, 1); r1 = min(row + win, h);
    c0 = max(col - win, 1); c1 = min(col + win, w);
    sub = P(r0:r1, c0:c1);
    sr  = row - r0 + 1;
    sc  = col - c0 + 1;

    comp = false(size(sub));
    if sub(sr, sc) > floor_
        comp = bwselect(sub > floor_, sc, sr, 8);
    end

    if nnz(comp) >= cfg.Post.MinArea
        D(r0:r1, c0:c1) = D(r0:r1, c0:c1) | comp;
    elseif cfg.Post.relaxDisk
        D = D | cnngvf.seedDisks([h w], seeds(s, :), BW, cfg);
    else
        dropped(s) = true;   % EXPERIMENT, docs/briefs/seed-hull-claim.md: no disk, no claim
    end
end
end
