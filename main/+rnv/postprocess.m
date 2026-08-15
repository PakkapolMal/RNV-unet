function [mask, O] = postprocess(O, J, seed, img, cfg)
%RNV.POSTPROCESS Turn raw Snake2D output into a cleaned binary lesion mask.
%
%  [mask, O] = rnv.postprocess(O, J, seed, img, cfg)
%
%  O    : contour from Snake2D, N x 2 as [row col]
%  J    : binary mask from Snake2D
%  seed : [cx cy]
%  cfg  : cfg.PostProc selects the strategy
%           "contour"   - repair self-intersections in O (the snake can
%                         pinch into figure-8s at low Beta), keep the loop
%                         nearest the seed, then rasterise. Returns the
%                         repaired O.
%           "component" - ignore O; keep the connected component of J that
%                         the seed falls in.
%
%  mask : cleaned logical mask, or [] when nothing survived (caller skips).
%
%  See also RNV.SEGMENTSEED.

switch cfg.PostProc
    case "contour"
        [O(:, 2), O(:, 1)] = repairContour(O(:, 2), O(:, 1), seed);
        mask = poly2mask(O(:, 2), O(:, 1), img.h, img.w);
        mask = imfill(mask, 'holes');
        mask = bwareaopen(mask, cfg.Post.MinArea);
        mask = imclose(mask, cfg.Post.MorphElement);
        if ~any(mask(:))
            mask = [];
        end

    case "component"
        mask = seedComponent(J, seed, cfg);

    otherwise
        error('rnv:postprocess:badMode', 'Unknown PostProc "%s".', cfg.PostProc);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [x_clean, y_clean] = repairContour(x, y, seed)
% Remove self-intersections, keep the loop closest to the seed, resample to
% the original point count. Falls back to the input if repair degenerates.
n_orig = numel(x);

warning('off', 'MATLAB:polyshape:repairedBySimplify');
cleanup = onCleanup(@() warning('on', 'MATLAB:polyshape:repairedBySimplify'));
poly = polyshape(x, y, 'Simplify', true);

if poly.NumRegions > 1
    parts = regions(poly);
    [gx, gy] = centroid(parts);
    [~, best] = min(hypot(gx - seed(1), gy - seed(2)));
    poly = parts(best);
end

poly = rmholes(poly);
[xb, yb] = boundary(poly);

valid   = ~isnan(xb) & ~isnan(yb);
x_clean = xb(valid);
y_clean = yb(valid);

if numel(x_clean) > 3
    t_old   = linspace(0, 1, numel(x_clean));
    t_new   = linspace(0, 1, n_orig);
    x_clean = interp1(t_old, x_clean, t_new, 'spline')';
    y_clean = interp1(t_old, y_clean, t_new, 'spline')';
else
    warning('rnv:postprocess:repairFailed', ...
        'Self-intersection repair degenerate at seed (%.0f, %.0f); keeping raw contour.', ...
        seed(1), seed(2));
    x_clean = x;
    y_clean = y;
end
end


function mask = seedComponent(J, seed, cfg)
% Keep the component the seed lies in; fall back to the nearest centroid.
Jf = imfill(J, 'holes');
CC = bwconncomp(Jf);
if CC.NumObjects == 0
    mask = [];
    return;
end

stats = regionprops(CC, 'Area', 'Centroid');
ok    = [stats.Area] > cfg.Post.MinArea;
if ~any(ok)
    mask = [];
    return;
end

L = labelmatrix(CC);
r = min(max(round(seed(2)), 1), size(L, 1));
c = min(max(round(seed(1)), 1), size(L, 2));
best = L(r, c);

if best == 0 || ~ok(best)
    cen = vertcat(stats.Centroid);
    d = hypot(cen(:, 1) - seed(1), cen(:, 2) - seed(2));
    d(~ok) = inf;
    [~, best] = min(d);
end

mask = false(size(J));
mask(CC.PixelIdxList{best}) = true;
mask = imclose(mask, cfg.Post.MorphElement);
end
