function [masks, union, compId, claimDist] = seedMasks(BW, seeds, minArea, rule, maxDist)
%CNNGVF.SEEDMASKS Split a mask into the one component each seed claims.
%
%  [masks, union] = cnngvf.seedMasks(BW, seeds, minArea)
%  [masks, union] = cnngvf.seedMasks(BW, seeds, minArea, rule)
%  [masks, union, compId, claimDist] = cnngvf.seedMasks(...)
%  [...] = cnngvf.seedMasks(BW, seeds, minArea, rule, maxDist)
%
%  maxDist   : a seed on background claims only a component within maxDist px
%              (seed pixel to its nearest pixel); beyond that it claims nothing.
%              Default Inf, the historical behaviour. cnngvf.maskFromProb passes
%              cfg.Post.claimMaxDist; the level-set path never does.
%
%  compId    : S x 1 label (bwconncomp order) of the claimed component, 0 = none
%  claimDist : S x 1 0 when the seed is inside it, else the distance in px from
%              the seed pixel to its nearest pixel, Inf when nothing was claimed.
%              Computed only when asked for; feeds the per-seed trace
%              (../docs/REFINE.md, step 0).
%
%  BW      : logical mask, any number of connected components
%  seeds   : S x 2 [cx cy]
%  minArea : components below this size can never be claimed
%  rule    : how a seed on background picks a component, "centroid" (default)
%            or "boundary". See below.
%
%  masks : S x 1 cell. masks{s} is the logical component seed s claims, or []
%          when nothing qualifies. Two seeds inside one lesion return the same
%          component twice - which is exactly what the ground truth does, since
%          separate_segments.py writes one file per seed, not per component.
%  union : the OR of all claimed components. Components no seed claims are
%          dropped entirely.
%
%  Each seed claims the component it falls inside, or - when it lands on
%  background because the CNN was confident slightly off-centre - the nearest
%  qualifying component. "Nearest" has two definitions and they disagree:
%
%    "centroid"  distance to each component's centre of mass. The original
%                rule, and the one the frozen arms were measured with.
%    "boundary"  distance to the nearest pixel OF a component, via bwdist.
%                The same rule cnngvf.prepareData uses to label training
%                seeds, and the bug RNV-Claude/main's selectSeedComponent was
%                fixed for: a seed just outside a large concave lesion can be
%                nearer another blob's centroid than its own, so the centroid
%                rule hands it the wrong lesion.
%
%  Neither rule caps the distance by default - a seed always claims something if
%  anything qualifies. Training drops seeds beyond cfg.Patch.seedTolerance, but
%  that test reads ground truth and so cannot run at inference; maxDist is the
%  ground-truth-free cap on the prediction side, measured in REFINE.md step 1.
%
%  Defaults to "centroid" when omitted, so a 3-argument call behaves exactly as
%  it always has. Callers inside the pipeline pass cfg.Post.ClaimRule; the arm
%  pins it, and the frozen arms pin it to "centroid".
%
%  Per-seed masks are what the evaluation compares against
%  RNV-DRAC22GT/SegmentFilledGT/<stem>-<x>-<y>.png; the union is what gets
%  written as the image-level mask.
%
%  minArea is a parameter, not cfg.Post.MinArea, because the two callers want
%  different thresholds: at initialization a small blob is a lesion that has
%  not grown yet (pass 0), while after evolution it is noise.
%
%  See also CNNGVF.INITPHI, CNNGVF.SEGMENTIMAGE, CNNGVF.EVALUATE.

if nargin < 4 || strlength(rule) == 0
    rule = "centroid";
end
if nargin < 5 || isempty(maxDist)
    maxDist = Inf;
end

nSeeds    = size(seeds, 1);
masks     = cell(nSeeds, 1);
union     = false(size(BW));
compId    = zeros(nSeeds, 1);     % label of the claimed component, 0 = none
claimDist = inf(nSeeds, 1);       % 0 inside it, else px to its nearest pixel

CC = bwconncomp(BW);
if CC.NumObjects == 0 || nSeeds == 0
    return;
end

stats = regionprops(CC, 'Area', 'Centroid');
ok    = [stats.Area] >= minArea;
if ~any(ok)
    return;
end

L      = labelmatrix(CC);
cen    = vertcat(stats.Centroid);
[h, w] = size(BW);

% Boundary rule: distance transform of the QUALIFYING components only, so
% nearest(r,c) indexes a pixel that is already known to belong to a component
% above minArea - the same construction cnngvf.prepareData uses against the
% ground-truth mask. Computed once for the image, not once per seed.
nearest = [];
if rule == "boundary"
    okBW = false(h, w);
    okBW(vertcat(CC.PixelIdxList{ok})) = true;
    [~, nearest] = bwdist(okBW);
end

for s = 1:nSeeds
    r = min(max(round(seeds(s, 2)), 1), h);
    c = min(max(round(seeds(s, 1)), 1), w);
    k = L(r, c);

    dist = 0;
    if k == 0 || ~ok(k)
        if rule == "boundary"
            k = L(nearest(r, c));
        else
            d = hypot(cen(:, 1) - seeds(s, 1), cen(:, 2) - seeds(s, 2));
            d(~ok) = inf;
            [~, k] = min(d);
        end
        if nargout >= 4 || isfinite(maxDist)
            [pr, pc] = ind2sub([h w], CC.PixelIdxList{k});
            dist = min(hypot(pr - r, pc - c));
        end
        if dist > maxDist
            continue;   % too far: claims nothing, masks{s} stays [], claimDist Inf
        end
    end

    m = false(h, w);
    m(CC.PixelIdxList{k}) = true;
    masks{s}     = m;
    union        = union | m;
    compId(s)    = k;
    claimDist(s) = dist;
end
end
