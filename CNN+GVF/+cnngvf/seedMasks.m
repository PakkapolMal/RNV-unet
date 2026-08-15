function [masks, union] = seedMasks(BW, seeds, minArea)
%CNNGVF.SEEDMASKS Split a mask into the one component each seed claims.
%
%  [masks, union] = cnngvf.seedMasks(BW, seeds, minArea)
%
%  BW      : logical mask, any number of connected components
%  seeds   : S x 2 [cx cy]
%  minArea : components below this size can never be claimed
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
%  qualifying component by centroid.
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

for s = 1:nSeeds
    r = min(max(round(seeds(s, 2)), 1), h);
    c = min(max(round(seeds(s, 1)), 1), w);
    k = L(r, c);

    if k == 0 || ~ok(k)
        d = hypot(cen(:, 1) - seeds(s, 1), cen(:, 2) - seeds(s, 2));
        d(~ok) = inf;
        [~, k] = min(d);
    end

    m = false(h, w);
    m(CC.PixelIdxList{k}) = true;
    masks{s} = m;
    union    = union | m;
end
end
