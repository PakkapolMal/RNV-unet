function [perSeed, mask] = seedHull(perSeed, mask, seeds, cfg)
%CNNGVF.SEEDHULL Fill inter-frond gaps between seeds that lie close together.
%
%  [perSeed, mask] = cnngvf.seedHull(perSeed, mask, seeds, cfg)
%
%  Runs after claiming, so it only adds pixels and cannot revive a blob no
%  seed claimed. Every triangle of the seeds' Delaunay triangulation whose
%  three seeds are pairwise within cfg.Post.hullLink px is filled, and so are
%  holes enclosed by those triangles. It adds only inside the seed cloud and
%  never touches a lesion's rim. 0 turns it off.
%
%  hullLink = 128 was cross-fitted over 64-256; at 64 px the fill was inert.
%  docs/results/2026-09-30-seed-hull-d.md.
%
%  The added pixels go into mask, and each seed's mask grows to its component
%  of the new mask, so mask stays the union of perSeed.
%
%  See also CNNGVF.MASKFROMPROB, CNNGVF.SEEDMASKS.

d = cfg.Post.hullLink;
if d <= 0 || size(seeds, 1) < 3
    return;
end

F = linkedTriangles(seeds, d, size(mask)) & ~mask;
if ~any(F(:))
    return;
end
mask = mask | F;

% Re-claim on the new mask: a seed takes the components its old mask and its
% own pixel now fall in.
[h, w] = size(mask);
idx = sub2ind([h w], min(max(round(seeds(:, 2)), 1), h), min(max(round(seeds(:, 1)), 1), w));
L = bwlabel(mask);
for s = 1:numel(perSeed)
    comps = L(idx(s));
    if ~isempty(perSeed{s})
        comps = [comps; L(perSeed{s})]; %#ok<AGROW>
    end
    comps = unique(comps(comps > 0));
    if ~isempty(comps)
        perSeed{s} = ismember(L, comps);
    end
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function F = linkedTriangles(seeds, d, sz)
F = false(sz);

% delaunayTriangulation merges duplicate points, so triangulate unique ones.
u = unique(seeds, 'rows', 'stable');
if size(u, 1) < 3
    return;
end
ws = warning('off', 'MATLAB:delaunayTriangulation:EmptyTriWarnId');   % collinear
restore = onCleanup(@() warning(ws));
DT = delaunayTriangulation(u);
T  = DT.ConnectivityList;
if isempty(T)
    return;
end

len  = @(a, b) hypot(u(T(:, a), 1) - u(T(:, b), 1), u(T(:, a), 2) - u(T(:, b), 2));
keep = max([len(1, 2), len(2, 3), len(1, 3)], [], 2) <= d;
if ~any(keep)
    return;
end

% Label pixels by triangle, only over the kept triangles' bounding box.
v  = u(unique(T(keep, :)), :);
c0 = max(floor(min(v(:, 1))), 1); c1 = min(ceil(max(v(:, 1))), sz(2));
r0 = max(floor(min(v(:, 2))), 1); r1 = min(ceil(max(v(:, 2))), sz(1));
[X, Y] = meshgrid(c0:c1, r0:r1);
t = pointLocation(DT, X(:), Y(:));
in = ~isnan(t);
in(in) = keep(t(in));
F(r0:r1, c0:c1) = reshape(in, size(X));
% A sub-pixel seed can round to a pixel just outside its own triangle; mark it
% so the seed still claims the fill.
F(sub2ind(sz, min(max(round(v(:, 2)), 1), sz(1)), min(max(round(v(:, 1)), 1), sz(2)))) = true;
F = imfill(F, 'holes');
end
