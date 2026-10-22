function [masks, union] = seedMasks(BW, seeds, minArea, rule)
%CNNGVF.SEEDMASKS Split a mask into the one component each seed claims.
%
%  [masks, union] = cnngvf.seedMasks(BW, seeds, minArea)
%  [masks, union] = cnngvf.seedMasks(BW, seeds, minArea, rule)
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
%                seeds. A seed just outside a large concave lesion can be
%                nearer another blob's centroid than its own, so the centroid
%                rule hands it the wrong lesion.
%
%  Neither rule caps the distance - a seed always claims something if anything
%  qualifies. Training drops seeds beyond cfg.Patch.seedTolerance, but that
%  test reads ground truth and so cannot run at inference.
%
%  Defaults to "centroid" when omitted. The pipeline passes cfg.Post.ClaimRule:
%  "boundary" by default, "centroid" on the frozen arms.
%
%  The union of the per-seed masks is what gets written and scored as the
%  image-level mask.
%
%  See also CNNGVF.MASKFROMPROB, CNNGVF.EVALUATE.

if nargin < 4 || strlength(rule) == 0
    rule = "centroid";
end

nSeeds = size(seeds, 1);
masks  = cell(nSeeds, 1);
union  = false(size(BW));

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

    if k == 0 || ~ok(k)
        if rule == "boundary"
            k = L(nearest(r, c));
        else
            d = hypot(cen(:, 1) - seeds(s, 1), cen(:, 2) - seeds(s, 2));
            d(~ok) = inf;
            [~, k] = min(d);
        end
    end

    m = false(h, w);
    m(CC.PixelIdxList{k}) = true;
    masks{s} = m;
    union    = union | m;
end
end
