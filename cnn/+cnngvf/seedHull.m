function [perSeed, mask] = seedHull(perSeed, mask, seeds, cfg)
%CNNGVF.SEEDHULL Fill inter-frond gaps between seeds whose heat-map bumps overlap.
%
%  [perSeed, mask] = cnngvf.seedHull(perSeed, mask, seeds, cfg)
%
%  Runs after claiming, so it only adds pixels and cannot revive a blob no
%  seed claimed. Two seeds are linked when they are within
%  2 * cfg.Patch.sigmaSeed: two Gaussians of width sigma sum to one peak iff
%  their centres are at most 2 sigma apart. A seed with no link adds nothing,
%  which leaves sparse seeds - mostly small lesions - alone.
%
%  cfg.Post.seedHull  (experiment: docs/briefs/seed-hull.md)
%    "none"       nothing
%    "triangles"  every Delaunay triangle whose three seeds are pairwise
%                 linked, holes enclosed by those triangles filled
%    "close"      imclose(disk sigmaSeed) over the components that linked
%                 seeds claimed: bridges gaps up to 2 * sigmaSeed wide
%
%  The added pixels go into mask, and each seed's mask grows to its component
%  of the new mask, so mask stays the union of perSeed.
%
%  See also CNNGVF.MASKFROMPROB, CNNGVF.SEEDMASKS.

if cfg.Post.seedHull == "none" || size(seeds, 1) < 2
    return;
end

d = 2 * cfg.Patch.sigmaSeed;
D = sqrt((seeds(:, 1) - seeds(:, 1)').^2 + (seeds(:, 2) - seeds(:, 2)').^2);
linked = D <= d;
linked(1:size(D, 1) + 1:end) = false;

switch cfg.Post.seedHull
    case "triangles"
        F = linkedTriangles(seeds, D, d, size(mask));

    case "close"
        E = false(size(mask));
        for s = find(any(linked, 2))'
            if ~isempty(perSeed{s})
                E = E | perSeed{s};
            end
        end
        F = imclose(E, strel('disk', cfg.Patch.sigmaSeed));

    otherwise
        error('cnngvf:seedHull:badMode', ...
            'Unknown Post.seedHull "%s". Use "none", "triangles" or "close".', ...
            cfg.Post.seedHull);
end

F = F & ~mask;
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

function F = linkedTriangles(seeds, D, d, sz)
F = false(sz);

% delaunayTriangulation merges duplicate points, so triangulate unique ones.
[u, iu] = unique(seeds, 'rows', 'stable');
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

Du   = D(iu, iu);
edge = @(a, b) Du(sub2ind(size(Du), T(:, a), T(:, b))) <= d;
keep = edge(1, 2) & edge(2, 3) & edge(1, 3);
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
F = imfill(F, 'holes');
end
