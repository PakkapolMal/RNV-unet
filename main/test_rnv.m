function test_rnv()
%TEST_RNV Self-check for the rnv package. Runs on synthetic images only,
%  so it needs no dataset. Errors on the first failed assertion.
%
%  Usage:  test_rnv
%
%  Covers the non-trivial logic: metric arithmetic, both initializers'
%  first-drop radius detection, contour repair of a self-intersecting
%  (figure-8) snake, seed-component selection, and an end-to-end segmentation
%  of a known blob.

fprintf('test_rnv\n');

testMetrics();          fprintf('  ok  metrics\n');
testRayCastInit();      fprintf('  ok  raycast init\n');
testAdaptiveInit();     fprintf('  ok  adaptive init\n');
testContourRepair();    fprintf('  ok  contour repair\n');
testSeedComponent();    fprintf('  ok  seed component\n');
testEndToEnd();         fprintf('  ok  end-to-end segmentation\n');

fprintf('all passed\n');
end


% =========================================================================

function testMetrics()
a = false(10); a(1:5, 1:5) = true;     % 25 px
b = false(10); b(3:7, 3:7) = true;     % 25 px, 9 px overlap

m = rnv.metrics(a, b);
assert(abs(m.iou - 9/41) < 1e-12, 'iou');
assert(abs(m.precision - 9/25) < 1e-12, 'precision');
assert(abs(m.recall - 9/25) < 1e-12, 'recall');
assert(abs(m.f1 - 9/25) < 1e-9, 'f1');

m = rnv.metrics(a, a);
assert(m.iou == 1 && m.f1 == 1, 'identical masks must score 1');

m = rnv.metrics(false(10), false(10));
assert(m.iou == 1, 'empty vs empty is a correct negative');

m = rnv.metrics(a, false(10));
assert(m.iou == 0 && m.f1 == 0, 'no overlap must score 0');
end


function testRayCastInit()
% A disc of radius 80: every ray should land on the boundary.
[img, cfg, seed, trueR] = syntheticDisc("raycast");
P = rnv.initContour(seed, img, cfg);

assert(isequal(size(P), [cfg.Radius.n_points, 2]), 'P must be n_points x 2');
assert(all(P(:, 1) >= 1 & P(:, 1) <= img.h), 'rows out of bounds');
assert(all(P(:, 2) >= 1 & P(:, 2) <= img.w), 'cols out of bounds');

r = hypot(P(:, 2) - seed(1), P(:, 1) - seed(2));
assert(max(abs(r - trueR)) < 12, ...
    'raycast radii off by %.1f px (want < 12)', max(abs(r - trueR)));
end


function testAdaptiveInit()
[img, cfg, seed, trueR] = syntheticDisc("adaptive");
P = rnv.initContour(seed, img, cfg);

r = hypot(P(:, 2) - seed(1), P(:, 1) - seed(2));
assert(std(r) < 1e-6, 'adaptive init must be a circle');
% Wider tolerance than raycast: this strategy deliberately advances past the
% energy peak to the first drop, so it lands slightly outside the boundary.
assert(abs(r(1) - trueR) < 25, ...
    'adaptive radius %.1f, want ~%d', r(1), trueR);
end


function testContourRepair()
% Figure-8: the repair must return a simple loop, not the crossed one.
t = linspace(0, 2*pi, 60)';
x = 200 + 80 * sin(2 * t);
y = 200 + 80 * sin(t);

cfg  = testConfig("raycast");
img  = struct('h', 400, 'w', 400);
seed = [200, 130];   % inside the upper lobe

[mask, O] = rnv.postprocess([y, x], [], seed, img, cfg);

assert(~isempty(mask), 'repair produced no mask');
assert(isequal(size(O), [60, 2]), 'repair must preserve the point count');
assert(mask(round(seed(2)), round(seed(1))), 'seed must lie inside the mask');

% A simple polygon's area equals its filled mask area; a figure-8's does not.
assert(abs(polyarea(O(:, 2), O(:, 1)) - nnz(mask)) / nnz(mask) < 0.1, ...
    'repaired contour is still self-intersecting');
end


function testSeedComponent()
% Two blobs. The seed sits in the smaller one, whose centroid is further
% from the seed than the big blob's would be for a naive nearest-centroid
% rule -- containment must win.
J = false(300);
J(20:60, 20:60)    = true;    % small, contains the seed
J(100:280, 100:280) = true;   % large

cfg = testConfig("adaptive");
seed = [40, 40];

mask = rnv.postprocess([], J, seed, struct('h', 300, 'w', 300), cfg);
assert(~isempty(mask), 'no component selected');
assert(mask(40, 40), 'must keep the component containing the seed');
assert(~mask(200, 200), 'must not keep the other component');

% Seed in background: falls back to the nearest valid centroid.
mask = rnv.postprocess([], J, [285, 285], struct('h', 300, 'w', 300), cfg);
assert(~isempty(mask) && mask(200, 200), 'fallback should pick the large blob');

% Everything below MinArea -> empty, caller skips.
small = false(300); small(1:5, 1:5) = true;
mask = rnv.postprocess([], small, [3, 3], struct('h', 300, 'w', 300), cfg);
assert(isempty(mask), 'sub-MinArea components must be rejected');
end


function testEndToEnd()
% Full pipeline on a bright disc: the snake must recover most of it.
[img, cfg, seed, trueR] = syntheticDisc("raycast");
cfg.Snake.GIterations = 100;   % keep the self-check fast
cfg.Snake.Iterations  = 100;

mask = rnv.segmentSeed(seed, img, cfg);
assert(~isempty(mask), 'segmentation returned nothing');

[X, Y] = meshgrid(1:img.w, 1:img.h);
truth  = hypot(X - seed(1), Y - seed(2)) <= trueR;

m = rnv.metrics(mask, truth);
assert(m.iou > 0.7, 'end-to-end IoU %.3f, want > 0.7', m.iou);
end


% =========================================================================
% HELPERS
% =========================================================================

function cfg = testConfig(initMethod)
cfg = rnv.defaults(initMethod);
cfg.Snake.Verbose = false;
end

function [img, cfg, seed, trueR] = syntheticDisc(initMethod)
% Bright disc on a dark noisy background, sized like a small OCTA crop.
n     = 400;
seed  = [200, 200];
trueR = 80;

[X, Y] = meshgrid(1:n, 1:n);
I = 0.15 + 0.7 * double(hypot(X - seed(1), Y - seed(2)) <= trueR);

rng(0);
I = min(max(I + 0.02 * randn(n), 0), 1);

cfg = testConfig(initMethod);
cfg.Radius.r_max = 180;   % the disc is well inside this
img = rnv.loadImage(I, cfg);
end
