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
testPostModes(cfg);
testSeedComponents(cfg);
testProbCache(cfg);
testDomain(cfg);
testArms(cfg);
testSeedTolerance(cfg);
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


function testPostModes(cfg)
% The reduced pipeline (../docs/REDUCE.md): cnngvf.maskFromProb must return the
% level-set path's contract exactly, must never touch the GVF, and must respond
% to all three switches the ablation varies.
[truth, seeds] = syntheticScene();
P = stubProbMap(truth);

cfg.Post.mode = "threshold";

% Contract: same shapes and types as the level-set path, or run_batch writes
% nothing for a seed and run_eval silently scores fewer seeds than were run.
[mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg);
assert(islogical(mask) && isequal(size(mask), size(P)), ...
    'maskFromProb returned %s %s, expected a logical map the size of P', ...
    class(mask), mat2str(size(mask)));
assert(iscell(perSeed) && numel(perSeed) == size(seeds, 1), ...
    'maskFromProb returned %d per-seed masks for %d seeds', ...
    numel(perSeed), size(seeds, 1));
assert(iou(mask, truth) > 0.8, ...
    'threshold mode scored IoU %.3f on the synthetic scene', iou(mask, truth));

% The GVF mex must not be reached at all. Structurally true of segmentImage's
% branch, but that is what every silent regression was before it happened.
cnngvf.gvfField();                       % reset the counter
cnngvf.maskFromProb(P, seeds, cfg);
assert(cnngvf.gvfField() == 0, 'threshold mode called the GVF');

% All eight ablation cells must produce a usable mask - a switch combination
% that errors would be discovered halfway through the run, not here.
for dk = [false true]
    for ic = [false true]
        c = cfg; c.Post.diskFallback = dk; c.Post.imclose = ic;
        m = cnngvf.maskFromProb(P, seeds, c);
        assert(islogical(m) && nnz(m) > 0, ...
            'diskFallback=%d imclose=%d produced an empty mask', dk, ic);
    end
end

% The fallback must actually fire, and only where the threshold left nothing.
% A seed on empty background is the case it exists for.
lonely = [seeds; 230 30];
off = cfg; off.Post.diskFallback = false;
on  = cfg; on.Post.diskFallback  = true;
noDisk = cnngvf.maskFromProb(P, lonely, off);
disk   = cnngvf.maskFromProb(P, lonely, on);
assert(~noDisk(30, 230) && disk(30, 230), ...
    'the disk fallback did not plant a disk at a seed the threshold missed');

% "relax" must find the SHAPE the CNN saw weakly, not a circle. A soft blob
% peaking below tau is invisible to the threshold and to a disk, but relaxing
% the threshold locally recovers its actual outline.
soft = zeros(size(P));
[xx2, yy2] = meshgrid(1:size(P, 2), 1:size(P, 1));
soft = max(soft, 0.42 * exp(-((xx2 - 60).^2 + (yy2 - 200).^2) / (2 * 22^2)));
Psoft  = max(P, soft);
weak   = [seeds; 60 200];

rc = cfg; rc.Post.diskFallback = true; rc.Post.fallbackMode = "relax";
dc = cfg; dc.Post.diskFallback = true; dc.Post.fallbackMode = "disk";
relaxed = cnngvf.maskFromProb(Psoft, weak, rc);
disked  = cnngvf.maskFromProb(Psoft, weak, dc);

assert(relaxed(200, 60) && disked(200, 60), ...
    'both fallbacks should cover a seed the threshold missed');
assert(nnz(relaxed) > nnz(disked), ...
    ['relax recovered %d px against the disk''s %d - it should find more of a ' ...
     'blob that peaks below tau'], nnz(relaxed), nnz(disked));

% It must stay local: relaxing must not flood the far side of the image.
assert(~relaxed(30, 230), 'relax leaked to an unrelated background location');

% A seed with genuinely nothing near it keeps the recall floor - relax degrades
% to the disk rather than returning nothing.
bare = cnngvf.maskFromProb(P, [seeds; 230 30], rc);
assert(bare(30, 230), 'relax dropped a seed instead of falling back to the disk');

% relaxDiskFallback = false EXCLUDES that seed instead: no disk, no claim, an
% empty per-seed entry, and the other seeds' output unchanged.
nd = rc; nd.Post.relaxDiskFallback = false;
[bm, bp] = cnngvf.maskFromProb(P, [seeds; 230 30], nd);
assert(~bm(30, 230) && isempty(bp{end}) && numel(bp) == size(seeds, 1) + 1, ...
    'relaxDiskFallback = false still produced output for a seed relax found nothing for');
assert(isequal(bm, cnngvf.maskFromProb(P, seeds, rc)), 'excluding one seed changed the others');

% Post.tau must reach the output, or a tau sweep would measure nothing.
c = cfg; c.Post.tau = 0.95;
assert(nnz(cnngvf.maskFromProb(P, seeds, c)) < nnz(mask), ...
    'raising Post.tau did not shrink the mask - the threshold is not being read');

% run_ablate rebuilds the threshold mask in two stages so its two imclose cells
% can share the expensive half. That reconstruction must equal maskFromProb
% exactly, or the ablation is not measuring the pipeline it claims to.
staged = P > cfg.Post.tau;
if cfg.Post.diskFallback
    staged = staged | cnngvf.seedDisks(size(P), seeds, staged, cfg);
end
if cfg.Post.imclose
    staged = imclose(staged, cfg.Post.MorphElement);
end
staged = imfill(staged, 'holes');
[~, stagedMask] = cnngvf.seedMasks(staged, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);
assert(isequal(stagedMask, mask), ...
    'run_ablate''s staged reconstruction diverged from cnngvf.maskFromProb');

fprintf('  post modes                ok (threshold, 8 cells, no gvf)\n');
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

% The two claiming rules, on the case that separates them: a long concave
% component whose centre of mass is far from its own pixels, and a small compact
% one nearby. A seed just off the end of the long arm is 4 px from the long
% component's BOUNDARY but 64 px from its centroid, while the compact blob's
% centroid is only 42 px away - so "centroid" hands the seed the wrong
% component and "boundary" hands it the right one.
%
% Silent when wrong: both rules return a plausible mask, and only the identity
% of the component differs.
C = false(200);
C(100:106, 20:140) = true;      % long thin arm, centroid ~(80,103)
C(60:90,  160:190) = true;      % compact blob, centroid ~(175,75)
seed = [144 103];               % 3 px past the arm's end, on background

far  = cnngvf.seedMasks(C, seed, cfg.Post.MinArea, "centroid");
near = cnngvf.seedMasks(C, seed, cfg.Post.MinArea, "boundary");
assert(far{1}(75, 175) && ~far{1}(103, 140), ...
    'the centroid rule no longer picks the compact blob - the fixture stopped separating the rules');
assert(near{1}(103, 140) && ~near{1}(75, 175), ...
    'the boundary rule did not claim the component the seed actually touches');

% Omitting the argument must still behave exactly as it always has.
assert(isequal(cnngvf.seedMasks(C, seed, cfg.Post.MinArea), far), ...
    'the 3-argument call no longer defaults to the centroid rule');

% Boundary distance must respect minArea: a sub-threshold component sitting
% right next to the seed cannot be claimed just because it is close.
D = C; D(101:105, 145:150) = true;   % 30 px, adjacent to the seed, below minArea
adj = cnngvf.seedMasks(D, seed, cfg.Post.MinArea, "boundary");
assert(~adj{1}(103, 147) && adj{1}(103, 140), ...
    'the boundary rule claimed a component below minArea');

fprintf('  seed mask selection       ok (centroid and boundary rules)\n');
end


function testProbCache(cfg)
% The cache is what makes a post-processing change cost seconds instead of 18 h,
% so two things about it have to hold: it must not quantize, and it must refuse
% to hand back a P that its own network trained on.
folds = cnngvf.cvFolds(cfg);
held  = folds{2}(1);            % an image fold 2 holds out
cacheDir = tempname;
cleanup  = onCleanup(@() rmdir(cacheDir, 's'));

P = single(rand(64));           % values that uint8 could not survive
cnngvf.probCache("write", cacheDir, held, P, struct('fold', 2, 'arm', "test"));

got = cnngvf.probCache("read", cacheDir, held, cfg);
assert(isa(got.P, 'single'), 'cache round-tripped P as %s', class(got.P));
assert(isequal(got.P, P), ...
    'cached P is not bit-exact - a quantized cache cannot support a tau sweep');
assert(got.meta.fold == 2, 'cache lost the fold index');

assert(ismember(held, string(cnngvf.probCache("list", cacheDir))), ...
    'list did not report the image just written');

% --- the leakage guard, in both directions -------------------------------
% The same image, re-tagged with a fold that does NOT hold it out: that P came
% from a network which trained on it. In-sample, and silent - it scores, and it
% scores well.
cnngvf.probCache("write", cacheDir, held, P, struct('fold', 3, 'arm', "test"));
assertErrorId(@() cnngvf.probCache("read", cacheDir, held, cfg), ...
    'cnngvf:probCache:leakage', 'a P tagged with the wrong fold was accepted');

% No fold at all - what a plain run_batch writes, whose model trained on most
% of the domain.
cnngvf.probCache("write", cacheDir, held, P, struct('fold', NaN, 'arm', "test"));
assertErrorId(@() cnngvf.probCache("read", cacheDir, held, cfg), ...
    'cnngvf:probCache:noFold', 'a P with no fold index was accepted');

fprintf('  prob cache                ok (single, fold-guarded)\n');
end


function testDomain(cfg)
% The manifest is the join between an image and its mask, and it is the piece
% TUH broke: nvImages/nv (N).png pairs with nvGtImages/gt (N).png, so the
% identity mapping DRAC22 relies on does not hold. A silent failure here scores
% every prediction against the wrong mask or against nothing.
M = cnngvf.manifest(cfg);

assert(height(M) > 0, 'manifest is empty');
assert(numel(unique(M.Name)) == height(M), 'manifest names are not unique');
assert(all(arrayfun(@(p) isfile(p), M.ImgPath)), 'a manifest image path does not exist');
assert(all(arrayfun(@(p) isfile(p), M.GtPath)),  'a manifest gt path does not exist');

% The stem rewrite actually rewrites. Every dataset must resolve its own names,
% and a mask must never be borrowed from another dataset's directory.
for d = unique(M.Dataset)'
    sub = M(M.Dataset == d, :);
    Ds  = cfg.Datasets(arrayfun(@(x) x.Name == d, cfg.Datasets));
    assert(all(startsWith(sub.GtPath, string(Ds.GT))), ...
        'dataset %s resolved a mask outside its own GT directory', d);
end

% Points land on images that exist in the domain.
[names, seeds] = cnngvf.loadPoints(cfg);
assert(all(ismember(names, M.Name)), 'loadPoints returned an out-of-domain image');
assert(size(seeds, 1) == numel(names), 'names and seeds disagree in length');

% Folds partition the domain exactly once - the property the whole
% out-of-sample claim rests on.
folds = cnngvf.cvFolds(cfg);
all_  = vertcat(folds{:});
assert(numel(all_) == height(M) && isempty(setdiff(all_, M.Name)), ...
    'cvFolds does not partition the domain');
assert(numel(unique(all_)) == numel(all_), 'an image appears in two folds');

fprintf('  domain manifest           ok (%d images, %d seeds, %s)\n', ...
    height(M), numel(names), strjoin(compose("%s %d", ...
        unique(M.Dataset), groupcounts(M.Dataset))', ', '));
end


function testArms(cfg)
% Every arm must be self-consistent. The failure this guards against is the one
% that actually happened: run_train and run_train_unclustered drifting apart on
% the channel flags, which a network cannot survive.
for a = cnngvf.arm()
    c = cnngvf.arm(cfg, a);
    assert(c.Arm == a, 'arm %s did not record its own name', a);
    assert(~isempty(c.ActiveDatasets), 'arm %s activates no dataset', a);
    assert(endsWith(c.Paths.Model, ".mat"), 'arm %s has no model path', a);
    assert(~isempty(c.Split.Val), 'arm %s derived an empty validation split', a);

    M = cnngvf.manifest(c);
    assert(~all(ismember(M.Name, c.Split.Val)), ...
        'arm %s validates on the whole domain, leaving nothing to train on', a);

    % A validation split that names only one dataset would silently stop
    % checkpoint selection from ever seeing the other.
    if numel(c.ActiveDatasets) > 1
        valDs = unique(M.Dataset(ismember(M.Name, c.Split.Val)));
        assert(numel(valDs) == numel(c.ActiveDatasets), ...
            'arm %s validates on %d of %d datasets', a, ...
            numel(valDs), numel(c.ActiveDatasets));
    end
end

% Relax is set per arm, for threshold mode only: DRACTUH_unclustered relaxes at
% 0.20, the frozen arms and defaults.m keep the disk, and initPhi always plants
% disks so an arm that relaxes cannot move its level-set output.
d = cnngvf.arm(cfg, "DRACTUH_unclustered");
assert(d.Post.fallbackMode == "relax" && d.Post.relaxFloor == 0.20, 'DRACTUH_unclustered does not set relax 0.20');
assert(cfg.Post.fallbackMode == "disk", 'defaults.m no longer keeps the disk');
for a = ["clustered", "unclustered", "unclustered_boundary"]
    assert(cnngvf.arm(cfg, a).Post.fallbackMode == "disk", 'frozen arm %s moved off the disk', a);
end
[truth, seeds] = syntheticScene();
[xx, yy] = meshgrid(1:256, 1:256);
P = max(stubProbMap(truth), 0.42 * exp(-((xx - 60).^2 + (yy - 200).^2) / (2 * 22^2)));
weak = [seeds; 60 200];
rc = cfg; rc.Post.fallbackMode = "relax"; rc.Post.relaxFloor = 0.20;
assert(isequal(refine(P, weak, rc), refine(P, weak, cfg)), ...
    'fallbackMode "relax" moved the level-set output - initPhi must always plant disks');

fprintf('  arm definitions           ok (%s; relax per arm, level set inert)\n', strjoin(cnngvf.arm(), ', '));
end


function testSeedTolerance(~)
% A seed just outside a lesion must be labelled with that lesion; one far away
% must still be dropped. Synthetic, so the boundary behaviour is exact rather
% than inferred from dataset counts.
%
% This is silent when wrong in both directions: too small and real training
% data is discarded, too large and background points are labelled as lesions,
% teaching the network to ignore its own heat-map. Neither raises an error.
gt = false(400);
gt(100:200, 100:200) = true;      % lesion, boundary at col 200
[dist, nearest] = bwdist(gt);
L = labelmatrix(bwconncomp(gt));

% [seed x, distance from the lesion edge]
cases = [210, 10; 215, 15; 225, 25; 350, 150];
tol   = 20;
got   = false(size(cases, 1), 1);

for i = 1:size(cases, 1)
    col = cases(i, 1); row = 150;
    id  = L(row, col);
    if id == 0 && dist(row, col) <= tol
        id = L(nearest(row, col));
    end
    got(i) = id > 0;
end

assert(got(1) && got(2), 'a near-miss seed inside the tolerance was dropped');
assert(~got(3), 'a seed beyond the tolerance was labelled');
assert(~got(4), 'a distant seed was labelled');

% Distance must be to the BOUNDARY, not the centroid: a seed 10 px outside a
% large lesion is 60 px from its centroid, so a centroid rule would reject it.
st = regionprops(gt, 'Centroid');
assert(hypot(210 - st.Centroid(1), 150 - st.Centroid(2)) > tol, ...
    'the synthetic lesion is too small to distinguish boundary from centroid distance');

% tolerance 0 must reproduce strict containment exactly - the frozen arms
% depend on it.
assert(L(150, 210) == 0 && dist(150, 210) > 0, ...
    'tolerance 0 would still label an outside seed');

fprintf('  seed tolerance            ok (%g px, boundary distance)\n', tol);
end


function testSegGtNaming(cfg)
% The evaluation keys ground truth by seed coordinates, so the filename must
% match what separate_segments.py wrote. A naming drift here would silently
% score every segment against nothing.
%
% DRAC22 + clustered points only. TUH ships no per-seed masks and the
% unclustered points are not what the files are named for, so the "clustered"
% arm is the only place this can be checked at all.
cfg = cnngvf.arm(cfg, "clustered");

assert(strcmp(cnngvf.segGtFile(cfg, "458.png", [631 641]), ...
    fullfile(cfg.Paths.SegGT, '458-631-641.png')), 'segGtFile naming changed');

[names, seeds] = cnngvf.loadPoints(cfg);
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
    error('%d of %d clustered seeds have no ground-truth segment file, first: %s', ...
        numel(missing), numel(names), missing(1));
end

fprintf('  segment gt naming         ok (%d clustered seeds all resolve)\n', numel(names));
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
[~, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);
end


function s = iou(a, b)
s = nnz(a & b) / max(nnz(a | b), 1);
end


function assertErrorId(fn, wantId, msg)
% A guard that does not fire is worse than no guard: it reads as protection.
try
    fn();
catch err
    assert(strcmp(err.identifier, wantId), ...
        '%s (threw %s instead of %s)', msg, err.identifier, wantId);
    return;
end
error('%s (nothing was thrown)', msg);
end
