function D = seedFallback(P, seeds, BW, cfg)
%CNNGVF.SEEDFALLBACK What to output for a seed the threshold left with nothing.
%
%  D = cnngvf.seedFallback(P, seeds, BW, cfg)
%
%  P     : probability map
%  BW    : the thresholded foreground; a seed already standing on it is skipped
%  D     : logical, what to OR into BW
%
%  This is the single largest post-processing effect measured on this pipeline
%  (-0.016 IoU at p = 1.3e-7, nine times the level set's), and it is not a rare
%  safety net: on the 68-image domain **258 of 874 seeds - 29.5%, across 34 of
%  the 68 images - never produce P > tau at all**. For every one of them this
%  function IS the answer.
%
%  cfg.Post.fallbackMode
%
%    "disk"   a disk of cfg.Drlse.seedRadius at the seed. The original, and the
%             default. It was chosen when this was a level-set INITIALIZATION
%             that the evolution would grow or discard; in the reduced pipeline
%             nothing grows, so the disk survives into the output at its literal
%             radius - ~707 px at r = 15, against ground-truth lesions of
%             1 106 - 432 533 px. It is a floor on recall, not a shape.
%
%    "relax"  lower the threshold locally around the seed until the model itself
%             says something is there, and take that component. Same job, but
%             shaped by P rather than by fixed geometry: a lesion the CNN saw
%             weakly gets its actual outline instead of a circle.
%
%             Bounded two ways, because a low threshold on a broad plateau
%             floods: the search is confined to a window of cfg.Post.relaxWindow
%             px around the seed, and it never goes below cfg.Post.relaxFloor.
%
%             It thresholds ONCE at the floor rather than laddering down to the
%             first component that clears MinArea. A ladder stops at an
%             arbitrary contour whose only property is enclosing 500 px, and it
%             lands smaller than the disk it replaced - both failure modes were
%             caught by test_cnngvf. One permissive threshold asks the
%             meaningful question, and leaves the floor as the single knob.
%
%  A seed that still finds nothing claimable - P never gives a big enough
%  component even at the floor - falls back to the disk. So this mode can only
%  add shape, never remove the recall floor.
%
%  See also CNNGVF.SEEDDISKS, CNNGVF.MASKFROMPROB, CNNGVF.INITPHI, RUN_ABLATE.

switch cfg.Post.fallbackMode
    case "disk"
        D = cnngvf.seedDisks(size(P), seeds, BW, cfg);

    case "relax"
        D = relaxFallback(P, seeds, BW, cfg);

    otherwise
        error('cnngvf:seedFallback:badMode', ...
            'Unknown Post.fallbackMode "%s". Use "disk" or "relax".', ...
            cfg.Post.fallbackMode);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function D = relaxFallback(P, seeds, BW, cfg)
[h, w] = size(P);
D = false(h, w);
if isempty(seeds)
    return;
end

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

    % One permissive threshold, not a ladder down to the first component that
    % happens to clear MinArea - that stops at an arbitrary contour whose only
    % property is enclosing 500 px, and it lands SMALLER than the disk it
    % replaced. Thresholding once at the floor asks a meaningful question
    % instead: what outline does the CNN give this lesion when read
    % permissively? The floor is then the single knob, and it is what
    % run_ablate(arm, "fallback") sweeps.
    comp = false(size(sub));
    if sub(sr, sc) > floor_
        comp = bwselect(sub > floor_, sc, sr, 8);
    end

    if nnz(comp) >= cfg.Post.MinArea
        D(r0:r1, c0:c1) = D(r0:r1, c0:c1) | comp;
    else
        % P never gives a claimable component near this seed, even at the floor.
        % Keep the recall floor rather than dropping the lesion.
        D = D | cnngvf.seedDisks([h w], seeds(s, :), BW, cfg);
    end
end
end
