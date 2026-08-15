function test_cnngvf()
%TEST_CNNGVF Self-check for the CNN+GVF pipeline.
%
%  test_cnngvf
%
%  Runs on synthetic data - no dataset, no trained network - by stubbing the
%  CNN with a blurred, noised copy of a known mask. Everything after the CNN
%  (force field, GVF, initialization, level set, cleanup) is exercised for
%  real, which is the half where a silent numerical mistake would hide.
%
%  See also CNNGVF.DEFAULTS, RUN_TRAIN, RUN_BATCH.

cfg = cnngvf.defaults();
fprintf('test_cnngvf | %s | device: %s\n\n', version('-release'), cfg.GpuName);

testPatchGeometry(cfg);
testGvfField(cfg);
testRefinement(cfg);
testSeedComponents(cfg);
testSegGtNaming(cfg);
testGpuMatchesCpu(cfg);

fprintf('\nAll checks passed.\n');
end


% =========================================================================
% CHECKS
% =========================================================================

function testPatchGeometry(cfg)
% The crop must be centred where we say it is, must survive a seed near the
% border, and the heat-map must peak on the seed rather than the patch centre.
p    = cfg.Patch.size;
half = floor(p / 2);

chans = reshape(1:(300 * 400), 300, 400);       % distinct value per pixel
chans = cat(3, chans, chans);

% Centred crop, no jitter: the patch centre is the seed.
seed = [200 150];
[X, rows, cols] = cnngvf.seedPatch(chans, seed, seed, cfg);
assert(isequal(size(X), [p p 3]), 'seedPatch returned %s', mat2str(size(X)));
assert(rows(half + 1) == seed(2) && cols(half + 1) == seed(1), ...
    'crop is not centred on the seed');
assert(X(half + 1, half + 1, 1) == single(chans(seed(2), seed(1), 1)), ...
    'patch centre pixel does not match the image pixel at the seed');

[~, iPeak] = max(X(:, :, 3), [], 'all');
[pr, pc]   = ind2sub([p p], iPeak);
assert(pr == half + 1 && pc == half + 1, 'heat-map does not peak on the seed');

% Jittered crop: the heat-map must follow the seed, not the patch centre.
center = seed + [40 -25];
Xj = cnngvf.seedPatch(chans, seed, center, cfg);
[~, iPeak] = max(Xj(:, :, 3), [], 'all');
[pr, pc]   = ind2sub([p p], iPeak);
assert(pr == half + 1 - (center(2) - seed(2)) && ...
       pc == half + 1 - (center(1) - seed(1)), ...
    'heat-map peak did not track the seed through the jitter');

% Border seed: still a full-size patch, zero-padded outside the image.
edgeSeed = [5 5];
Xe = cnngvf.seedPatch(chans, edgeSeed, edgeSeed, cfg);
assert(isequal(size(Xe), [p p 3]), 'border crop was not padded to full size');
assert(all(Xe(1:half - edgeSeed(2), :, 1) == 0, 'all'), ...
    'rows above the image should be zero-padded');

fprintf('  patch geometry            ok\n');
end


function testGvfField(cfg)
% The field must be finite, correctly sized, and point towards the structure
% from both sides. This is the check that catches a mex that loads but returns
% garbage.
f = zeros(128);
f(40:88, 40:88) = 1;
f = imgaussfilt(f, 2);

[u, v] = cnngvf.gvfField(f, cfg);

assert(isequal(size(u), size(f)) && isequal(size(v), size(f)), ...
    'gvfField returned the wrong size');
assert(all(isfinite(u(:))) && all(isfinite(v(:))), 'gvfField returned non-finite values');
assert(abs(max(hypot(u(:), v(:))) - 1) < 1e-9, 'field was not scaled to unit peak');

% Left of the square the field points right (+x); right of it, left.
assert(u(64, 20) > 0, 'field left of the structure does not point inwards');
assert(u(64, 108) < 0, 'field right of the structure does not point inwards');
assert(v(20, 64) > 0, 'field above the structure does not point inwards');
assert(v(108, 64) < 0, 'field below the structure does not point inwards');

fprintf('  gvf field                 ok\n');
end


function testRefinement(cfg)
% End to end from a stubbed probability map. Two disjoint concave blobs: the
% level set must recover both, which is the topological split a parametric
% snake could not do.
[truth, seeds] = syntheticScene();
P = stubProbMap(truth);

mask = refine(P, seeds, cfg);

assert(iou(mask, truth) > 0.8, 'IoU %.3f is below 0.8', iou(mask, truth));
assert(bwconncomp(mask).NumObjects == 2, ...
    'expected 2 components, got %d - the level set did not split', ...
    bwconncomp(mask).NumObjects);

fprintf('  drlse refinement          ok (IoU %.3f)\n', iou(mask, truth));
end


function testSeedComponents(cfg)
% Unclaimed components are dropped; a seed on background falls back to the
% nearest component; sub-minArea blobs are never claimed.
BW = false(200);
BW(20:60,  20:60)  = true;    % claimed by a seed inside it
BW(120:160, 120:160) = true;  % claimed by a nearby seed on background
BW(100:104, 20:24)  = true;   % 25 px, below minArea
BW(20:60,  140:180) = true;   % unclaimed

seeds = [40 40; 118 140];     % second seed sits just outside its blob
[masks, out] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea);

assert(out(40, 40), 'component containing a seed was dropped');
assert(out(140, 140), 'nearest-centroid fallback failed for an off-blob seed');
assert(~any(out(100:104, 20:24), 'all'), 'a sub-minArea blob was claimed');
assert(~any(out(20:60, 140:180), 'all'), 'an unclaimed component survived');

% Per-seed masks: each seed gets its own component, and they are disjoint here.
assert(numel(masks) == 2, 'expected one mask per seed');
assert(masks{1}(40, 40) && ~masks{1}(140, 140), 'seed 1 claimed the wrong component');
assert(masks{2}(140, 140) && ~masks{2}(40, 40), 'seed 2 claimed the wrong component');
assert(isequal(masks{1} | masks{2}, out), 'union does not match the per-seed masks');

% Two seeds inside one component must both return that component - the ground
% truth writes one file per seed, not per component.
both = cnngvf.seedMasks(BW, [30 30; 50 50], cfg.Post.MinArea);
assert(isequal(both{1}, both{2}), 'two seeds in one lesion returned different masks');

fprintf('  seed mask selection       ok\n');
end


function testSegGtNaming(cfg)
% The evaluation keys ground truth by seed coordinates, so the filename must
% match what separate_segments.py wrote. Checked against the real directory:
% a naming drift here would silently score every segment against nothing.
assert(strcmp(cnngvf.segGtFile(cfg, "458.png", [631 641]), ...
    fullfile(cfg.Paths.SegGT, '458-631-641.png')), 'segGtFile naming changed');

[names, seeds] = rnv.loadSeeds(cfg);
missing = strings(0, 1);
for k = 1:numel(names)
    p = cnngvf.segGtFile(cfg, names(k), seeds(k, :));
    if ~isfile(p)
        missing(end + 1, 1) = string(p); %#ok<AGROW>
    end
end
% Not assert: it evaluates its message arguments even when the check passes,
% and missing(1) does not exist in the passing case.
if ~isempty(missing)
    error('%d of %d seeds have no ground-truth segment file, first: %s', ...
        numel(missing), numel(names), missing(1));
end

fprintf('  segment gt naming         ok (%d seeds all resolve)\n', numel(names));
end


function testGpuMatchesCpu(cfg)
% The gpuArray branch must not change the maths.
if ~cfg.UseGpu
    fprintf('  gpu / cpu agreement       skipped (no gpu)\n');
    return;
end

[truth, seeds] = syntheticScene();
P = stubProbMap(truth);

cpu = cfg; cpu.UseGpu = false;
assert(isequal(refine(P, seeds, cfg), refine(P, seeds, cpu)), ...
    'the GPU and CPU level sets disagree');

fprintf('  gpu / cpu agreement       ok\n');
end


% =========================================================================
% HELPERS
% =========================================================================

function [truth, seeds] = syntheticScene()
% Two disjoint blobs, each with a deep concavity a plain edge force would not
% enter.
truth = false(256);
[xx, yy] = meshgrid(1:256, 1:256);

blob1 = (xx - 70).^2  + (yy - 70).^2  <= 40^2;
blob1(50:90, 70:110) = false;                       % bite out of blob 1

blob2 = (xx - 180).^2 + (yy - 180).^2 <= 35^2;
blob2(175:185, 180:215) = false;                    % narrow slit into blob 2

truth = truth | blob1 | blob2;
seeds = [70 70; 180 180];
end


function P = stubProbMap(truth)
% What a decent but imperfect CNN would emit: soft edges plus speckle.
rng(0);
P = imgaussfilt(double(truth), 3);
P = min(max(P + 0.05 * randn(size(P)), 0), 1);
end


function mask = refine(P, seeds, cfg)
% The post-CNN half of cnngvf.segmentImage, verbatim.
f      = cnngvf.buildForce(P, P, cfg);
[u, v] = cnngvf.gvfField(f, cfg);
phi0   = cnngvf.initPhi(P, seeds, cfg);
BW     = cnngvf.drlse(P, phi0, u, v, cfg);

BW   = imfill(imclose(BW, cfg.Post.MorphElement), 'holes');
[~, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea);
end


function s = iou(a, b)
s = nnz(a & b) / max(nnz(a | b), 1);
end
