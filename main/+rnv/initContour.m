function P = initContour(seed, img, cfg)
%RNV.INITCONTOUR Build the initial snake contour around a localization seed.
%
%  P = rnv.initContour(seed, img, cfg)
%
%  seed : [cx cy] localization point (column, row)
%  img  : struct from rnv.loadImage (uses .edge, .h, .w)
%  cfg  : struct from rnv.defaults; cfg.Init selects the strategy
%           "raycast"  - per-angle boundary estimate. Casts cfg.Radius.n_points
%                        rays from the seed, walks outward along the edge map
%                        and stops at the first significant drop past a peak,
%                        i.e. where the ray crossed the lesion boundary into
%                        dark background. Handles non-circular lesions.
%           "adaptive" - one global radius: the circle whose mean edge energy
%                        peaks, advanced to the first drop. Always a circle.
%
%  P    : n_points x 2 contour as [row col], ready for Snake2D.
%
%  A snake only finds the boundary it starts near, so this is the single
%  most consequential stage of the pipeline. Both strategies are vectorised:
%  all rays/radii are sampled in one interp2 call.
%
%  See also RNV.SEGMENTSEED, SNAKE2D.

cx = seed(1);
cy = seed(2);
R  = cfg.Radius;

theta = (0:R.n_points-1)' * (2*pi / R.n_points);   % n x 1

switch cfg.Init
    case "raycast"
        radii = rayCastRadii(cx, cy, theta, img, R);
    case "adaptive"
        radii = repmat(adaptiveRadius(cx, cy, img, R), R.n_points, 1);
    otherwise
        error('rnv:initContour:badInit', 'Unknown Init "%s".', cfg.Init);
end

% Cartesian, clamped to the image. Snake2D wants [row col].
P = [clampTo(cy + radii .* sin(theta), img.h), ...
     clampTo(cx + radii .* cos(theta), img.w)];
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function radii = rayCastRadii(cx, cy, theta, img, R)
% Per-angle first-drop boundary radius.
r_vec = R.r_min : R.r_max;                    % 1 x nr

XQ = clampTo(cx + cos(theta) * r_vec, img.w); % n x nr
YQ = clampTo(cy + sin(theta) * r_vec, img.h);

E = interp2(img.edge, XQ, YQ, 'linear', 0);
E = smoothdata(E, 2, 'gaussian', R.smoothing_window);

% First index at which the profile has fallen peak_drop_tol below the
% running maximum -- that is where the ray left the lesion edge.
runMax   = cummax(E, 2);
dropped  = (runMax - E) ./ (runMax + eps) >= R.peak_drop_tol;
[hit, k] = max(dropped, [], 2);
k(~hit)  = size(E, 2);                        % never dropped: use full ray

% Radius = the peak itself, i.e. argmax of the profile up to the drop.
inPrefix = (1:size(E, 2)) <= k;
Eprefix  = E;
Eprefix(~inPrefix) = -inf;
[~, peakIdx] = max(Eprefix, [], 2);
radii = r_vec(peakIdx)';

% Guard: a peak too close to the seed is speckle, not the lesion boundary.
tooClose = radii < R.min_peak_radius;
if any(tooClose)
    far = r_vec >= R.min_peak_radius;
    [~, farIdx] = max(E(:, far), [], 2);
    r_far = r_vec(far);
    radii(tooClose) = r_far(farIdx(tooClose))';
end

% Angular regularization on the closed radius profile: median filter kills
% single-ray spikes, Gaussian smoothing keeps the boundary biologically
% plausible. Circular padding avoids a seam at theta = 0.
pad    = 5;
padded = [radii(end-pad+1:end); radii; radii(1:pad)];
padded = smoothdata(medfilt1(padded, 5), 'gaussian', 5);
radii  = padded(pad+1 : end-pad);
end


function r_final = adaptiveRadius(cx, cy, img, R)
% Single global radius from the mean edge energy of concentric circles.
r_vec = R.r_min : R.r_step : R.r_max;              % 1 x nk
t     = (0:R.n_points-1)' * (2*pi / R.n_points);   % np x 1

XC = clampTo(cx + cos(t) * r_vec, img.w);          % np x nk
YC = clampTo(cy + sin(t) * r_vec, img.h);

E = mean(interp2(img.edge, XC, YC, 'linear', 0), 1);
E = smoothdata(E, 'movmean', R.smoothing_window);

[peakVal, peakIdx] = max(E);
if r_vec(peakIdx) < R.min_peak_radius
    far = r_vec >= R.min_peak_radius;
    [peakVal, relIdx] = max(E(far));
    r_far   = r_vec(far);
    peakIdx = find(r_vec == r_far(relIdx), 1);
end

% Advance outward to the first significant drop, else stay on the peak.
tail = peakIdx+1 : numel(E);
k    = find((peakVal - E(tail)) ./ (peakVal + eps) >= R.peak_drop_tol, 1);
if isempty(k)
    r_final = r_vec(peakIdx);
else
    r_final = r_vec(tail(k));
end
end


function v = clampTo(v, hi)
v = min(max(v, 1), hi);
end
