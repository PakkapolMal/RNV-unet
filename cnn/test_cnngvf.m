function test_cnngvf()
%TEST_CNNGVF Self-check for the segmentation pipeline.
%
%  test_cnngvf
%
%  The post-CNN checks run on synthetic data - no trained network - by
%  stubbing the CNN with a blurred, noised copy of a known mask. The prob
%  cache, run_cache, domain, arm and segment-naming checks need the datasets.
%
%  See also CNNGVF.DEFAULTS, RUN_TRAIN, RUN_BATCH.

cfg = cnngvf.defaults();
fprintf('test_cnngvf | %s | device: %s\n\n', version('-release'), cfg.GpuName);

testPatchGeometry(cfg);
testMaskFromProb(cfg);
testSeedComponents(cfg);
testSeedHull(cfg);
testProbCache(cfg);
testRunCache(cfg);
testDomain(cfg);
testArms(cfg);
testSeedTolerance(cfg);

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


function testMaskFromProb(cfg)
% End to end from a stubbed probability map: two disjoint concave blobs must
% both come back, as separate components, with one mask per seed.
[truth, seeds] = syntheticScene();
P = stubProbMap(truth);

[mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg);
assert(islogical(mask) && isequal(size(mask), size(P)), ...
    'maskFromProb returned %s %s, expected a logical map the size of P', ...
    class(mask), mat2str(size(mask)));
assert(iscell(perSeed) && numel(perSeed) == size(seeds, 1), ...
    'maskFromProb returned %d per-seed masks for %d seeds', ...
    numel(perSeed), size(seeds, 1));
assert(iou(mask, truth) > 0.8, 'IoU %.3f is below 0.8', iou(mask, truth));
assert(bwconncomp(mask).NumObjects == 2, ...
    'expected 2 components, got %d', bwconncomp(mask).NumObjects);

% "relax" must find the SHAPE the CNN saw weakly, not a circle. A soft blob
% peaking below tau is invisible to the threshold and to a disk, but relaxing
% the threshold locally recovers its actual outline.
[xx, yy] = meshgrid(1:size(P, 2), 1:size(P, 1));
Psoft = max(P, 0.42 * exp(-((xx - 60).^2 + (yy - 200).^2) / (2 * 22^2)));
weak  = [seeds; 60 200];

rc = cfg; rc.Post.fallbackMode = "relax";
dc = cfg; dc.Post.fallbackMode = "disk";
relaxed = cnngvf.maskFromProb(Psoft, weak, rc);
disked  = cnngvf.maskFromProb(Psoft, weak, dc);

assert(relaxed(200, 60) && disked(200, 60), ...
    'both fallbacks should cover a seed the threshold missed');
assert(nnz(relaxed) > nnz(disked), ...
    ['relax recovered %d px against the disk''s %d - it should find more of a ' ...
     'blob that peaks below tau'], nnz(relaxed), nnz(disked));
assert(~relaxed(30, 230), 'relax leaked to an unrelated background location');

% A seed with genuinely nothing near it keeps the recall floor under both
% modes: relax degrades to the disk rather than returning nothing.
bare = [seeds; 230 30];
r = cnngvf.maskFromProb(P, bare, rc);
d = cnngvf.maskFromProb(P, bare, dc);
assert(r(30, 230), 'relax dropped a seed instead of falling back to the disk');
assert(d(30, 230), 'the disk fallback did not plant a disk');

% Post.tau must reach the output, or a tau sweep would measure nothing. Under
% the disk: relax would hand the seeds back their components at the floor.
c = dc; c.Post.tau = 0.95;
assert(nnz(cnngvf.maskFromProb(P, seeds, c)) < nnz(cnngvf.maskFromProb(P, seeds, dc)), ...
    'raising Post.tau did not shrink the mask - the threshold is not being read');

fprintf('  mask from P               ok (IoU %.3f, relax and disk)\n', iou(mask, truth));
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


function testSeedHull(cfg)
% Three seeds pairwise within hullLink fill their triangle; a lone seed and a
% triangle with one long edge add nothing; 0 leaves the mask untouched.
% docs/briefs/seed-hull-d.md.
d     = 96;
seeds = [100 100; 100 + d 100; 100 + d/2 100 + 0.8*d; 400 400];
blob  = @(r, c) (abs((1:512)' - r) <= 4) & (abs((1:512) - c) <= 4);
perSeed = {blob(100, 100); blob(100, 100 + d); []; blob(400, 400)};   % seed 3 claimed nothing
mask  = perSeed{1} | perSeed{2} | perSeed{4};

cfg.Post.hullLink = 0;
[p0, m0] = cnngvf.seedHull(perSeed, mask, seeds, cfg);
assert(isequal(m0, mask) && isequal(p0, perSeed), 'hullLink 0 changed the mask');

cfg.Post.hullLink = d;
[p1, m1] = cnngvf.seedHull(perSeed, mask, seeds, cfg);
c = round(mean(seeds(1:3, :)));
assert(m1(c(2), c(1)), 'the linked triangle was not filled');
assert(isequal(m1(300:end, 300:end), mask(300:end, 300:end)), 'a lone seed added pixels');
assert(isequal(m1, p1{1} | p1{2} | p1{3} | p1{4}), 'mask is not the union of perSeed');
assert(p1{3}(c(2), c(1)), 'a seed with no claim did not take the filled triangle');

far = seeds; far(3, :) = [100 + d/2, 100 + 1.2*d];   % one edge longer than d
[~, m2] = cnngvf.seedHull(perSeed, mask, far, cfg);
assert(isequal(m2, mask), 'a triangle with an edge beyond hullLink was filled');

fprintf('  seed hull                 ok (triangle, long edge, lone seed, off)\n');
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


function testRunCache(cfg)
% run_cache is how a post-P change is verified, so its plumbing must be exact:
% it scores precisely the cached images, pairs each with its own seeds and
% mask, and reports a zero difference against a matching reference.
folds = cnngvf.cvFolds(cfg);
img   = folds{1}(1);
M     = cnngvf.manifest(cfg);
gt    = cnngvf.loadMask(M.GtPath(M.Name == img));
P     = single(imgaussfilt(double(gt), 3));

tmp = tempname;
cleanup = onCleanup(@() rmdir(tmp, 's'));
cacheDir = fullfile(tmp, 'probcache');
cnngvf.probCache("write", cacheDir, img, P, struct('fold', 1, 'arm', "test"));

[names, seeds] = cnngvf.loadPoints(cfg);
expected = cnngvf.metrics(cnngvf.maskFromProb(double(P), seeds(names == img, :), cfg), gt).iou;

c = cfg;
c.Paths.Output  = tmp;
c.CV.reference  = fullfile(tmp, 'ref.csv');
writetable(table(img, expected, 'VariableNames', {'Image', 'Iou'}), c.CV.reference);

[perImage, ~, maxDiff] = run_cache(c, cacheDir);
assert(height(perImage) == 1 && perImage.Image == img, 'run_cache scored the wrong image set');
assert(perImage.Iou == expected, 'run_cache IoU %.6f differs from maskFromProb %.6f', ...
    perImage.Iou, expected);
% Not exactly 0: writetable keeps 15 significant digits.
assert(maxDiff < 1e-12, 'run_cache reported %.3g against a matching reference', maxDiff);

fprintf('  run_cache                 ok (%s, reference diff %.1g)\n', img, maxDiff);
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

% docs/folds.csv is the published split; it must be what cvFolds deals.
if cfg.Arm == "DRACTUH_unclustered"
    [~, info] = cnngvf.cvFolds(cfg);
    F = readtable(fullfile(fileparts(cfg.Paths.Main), 'docs', 'folds.csv'), 'TextType', 'string');
    [ok, j] = ismember(info.Image, F.image);
    assert(height(F) == height(info) && all(ok) && isequal(F.fold(j), info.Fold) ...
        && isequal(F.dataset(j), info.Dataset), ...
        'docs/folds.csv no longer matches cnngvf.cvFolds; regenerate it and record why');
end

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

% defaults() IS the current arm, and asking for it by name changes nothing.
assert(cfg.Arm == "DRACTUH_unclustered", 'defaults() is no longer the current arm');
assert(isequal(cnngvf.arm(cfg, "DRACTUH_unclustered"), cfg), ...
    'cnngvf.arm("DRACTUH_unclustered") modified the defaults');
assert(cfg.Post.fallbackMode == "relax" && cfg.Post.relaxFloor == 0.20, ...
    'the current configuration does not relax at 0.20');
for a = ["unclustered", "unclustered_boundary"]
    assert(cnngvf.arm(cfg, a).Post.fallbackMode == "disk", 'frozen arm %s moved off the disk', a);
end

fprintf('  arm definitions           ok (%s)\n', strjoin(cnngvf.arm(), ', '));

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
