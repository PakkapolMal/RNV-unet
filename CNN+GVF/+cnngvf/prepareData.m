function [dsTrain, dsVal, info] = prepareData(cfg)
%CNNGVF.PREPAREDATA Build the training and validation patch datastores.
%
%  [dsTrain, dsVal, info] = cnngvf.prepareData(cfg)
%
%  Reads the localization CSV, keeps the seeds whose image has a ground-truth
%  mask, and returns two transform datastores that crop seed-centred patches
%  on demand. Each read yields {X, T}:
%     X : patch x patch x 3 single - [intensity, vesselness, seed heat-map]
%     T : patch x patch x 2 single - one-hot [background, RNV]
%
%  The label is the seed's OWN ground-truth segment
%  (SegmentFilledGT/<stem>-<x>-<y>.png), not the whole image mask. That makes
%  the training task identical to the one the evaluation scores - "segment the
%  lesion at the heat-map" rather than "segment all RNV in this patch" - and
%  is what gives the seed heat-map channel something to do. With whole-image
%  labels a patch containing two lesions marks both as foreground, and a
%  neighbouring lesion leaks into this seed's predicted component.
%
%  info : struct with the image lists, seed counts and patches per epoch.
%
%  Patches are cropped on the fly rather than materialised to disk. The 17
%  source images fit in ~140 MB of memory, while 400+ cached 256x256x3 single
%  patches would be well over a gigabyte - and re-randomising the jitter and
%  the flip/rotation every epoch gives free augmentation that a fixed cache
%  cannot. Nothing downstream reads a patch file, so none is written.
%
%  The split is by image (cfg.Split.Val), never by patch: two patches from the
%  same lesion must not straddle train and validation.
%
%  See also CNNGVF.TRAINNET, CNNGVF.SEEDPATCH, CNNGVF.LOADCHANNELS.

% Training seeds come from cfg.Paths.LocalizationTrain, which defaults to the
% same clustered CSV the evaluation uses but can point at a denser point set.
seedCfg = cfg;
seedCfg.Paths.Localization = cfg.Paths.LocalizationTrain;
[names, seeds] = rnv.loadSeeds(seedCfg);

% --- Keep only seeds whose image has a ground-truth mask -------------------
images  = unique(names, 'stable');
hasGt   = arrayfun(@(n) isfile(fullfile(cfg.Paths.GT, char(n))), images);
images  = images(hasGt);

% Cross-validation: drop the fold's held-out images before anything else, so
% they reach neither training nor inner validation.
if isfield(cfg.Split, 'Exclude') && ~isempty(cfg.Split.Exclude)
    excluded = ismember(images, string(cfg.Split.Exclude));
    if any(excluded)
        fprintf('Excluding %d held-out image(s) from training: %s\n', ...
            nnz(excluded), strjoin(images(excluded)', ', '));
        images = images(~excluded);
    end
end

if isempty(images)
    error('cnngvf:prepareData:noGt', 'No ground-truth masks found in %s', cfg.Paths.GT);
end

keep  = ismember(names, images);
names = names(keep);
seeds = seeds(keep, :);

% --- Load each image once --------------------------------------------------
chans = cell(numel(images), 1);
for i = 1:numel(images)
    [~, c] = cnngvf.loadChannels(fullfile(cfg.Paths.Img, char(images(i))), cfg);
    chans{i} = single(c);
end

imgIdx = arrayfun(@(n) find(images == n, 1), names);

% --- One ground-truth segment per seed -------------------------------------
% The label is the FilledGT connected component the seed falls in, computed
% here rather than read from SegmentFilledGT. That is the same rule
% separate_segments.py applies - verified to reproduce all 35 segment files
% exactly - but it also works for point sets those files are not keyed to,
% which is what lets the unclustered points be used for training.
%
% A seed on background cannot be labelled (training it against pure background
% would teach the network to ignore its own heat-map), so it is dropped.
segMasks = cell(numel(names), 1);
onBackground = false(numel(names), 1);

for i = 1:numel(images)
    sel = find(names == images(i))';
    filled = rnv.loadMask(fullfile(cfg.Paths.GT, char(images(i))));
    CC = bwconncomp(filled);
    L  = labelmatrix(CC);
    [h, w] = size(filled);

    for k = sel
        row = min(max(round(seeds(k, 2)), 1), h);
        col = min(max(round(seeds(k, 1)), 1), w);
        id  = L(row, col);
        if id == 0
            onBackground(k) = true;
            continue;
        end
        m = false(h, w);
        m(CC.PixelIdxList{id}) = true;
        segMasks{k} = m;
    end
end

if any(onBackground)
    fprintf('Dropped %d of %d training seeds that land outside every lesion.\n', ...
        nnz(onBackground), numel(names));
    names    = names(~onBackground);
    seeds    = seeds(~onBackground, :);
    segMasks = segMasks(~onBackground);
    imgIdx   = imgIdx(~onBackground);
end

if isempty(names)
    error('cnngvf:prepareData:noSeeds', ...
        'No usable training seeds from %s', cfg.Paths.LocalizationTrain);
end

% --- Patient-level split ---------------------------------------------------
isVal    = ismember(images, cfg.Split.Val);
valSeed  = isVal(imgIdx);

if ~any(valSeed) || all(valSeed)
    error('cnngvf:prepareData:badSplit', ...
        ['cfg.Split.Val leaves one side of the split empty. It lists %s; ' ...
         'ground-truth images are %s.'], ...
        strjoin(cfg.Split.Val, ', '), strjoin(images', ', '));
end

% --- Datastores ------------------------------------------------------------
% Training reads seeds with a fresh jitter and dihedral transform each time.
% Validation is deterministic (no jitter, no augmentation) so its loss is
% comparable across epochs.
trainRows = find(~valSeed);
valIdx    = find(valSeed);

if cfg.Patch.balanceByImage
    % Equal patches per IMAGE, not per seed. Seed counts track lesion size and
    % how many detections a lesion attracted, so sampling uniformly per seed
    % hands the epoch to whichever images have the most: on the unclustered
    % points 1115.png supplied 39% of all training seeds and the top two images
    % 54%, and the resulting network over-predicted large diffuse regions
    % (validation precision 0.52 against 0.67 on training images).
    %
    % Every seed is still reachable - the budget is spread over an image's
    % seeds rather than any being discarded - so this rebalances exposure
    % without throwing data away.
    trainImgs = unique(imgIdx(trainRows))';
    budget    = max(1, round(cfg.Patch.perSeed * numel(trainRows) / numel(trainImgs)));

    % Draw the budget uniformly with replacement from the image's own seeds,
    % rather than taking the first N of them: an image with more seeds than the
    % budget would otherwise be pinned to the same subset in CSV order for the
    % whole run. Duplicates cost nothing because every read re-randomises the
    % jitter and the dihedral transform, so the same seed yields a different
    % patch each time.
    trainIdx = zeros(budget * numel(trainImgs), 1);
    at = 1;
    for i = trainImgs
        rows = trainRows(imgIdx(trainRows) == i);
        trainIdx(at:at + budget - 1) = rows(randi(numel(rows), budget, 1));
        at = at + budget;
    end
else
    trainIdx = repmat(trainRows, cfg.Patch.perSeed, 1);
end

dsTrain = makeDatastore(trainIdx(:), imgIdx, seeds, chans, segMasks, cfg, true);
dsVal   = makeDatastore(valIdx(:),   imgIdx, seeds, chans, segMasks, cfg, false);

info.trainImages    = images(~isVal)';
info.valImages      = images(isVal)';
info.trainSeeds     = nnz(~valSeed);
info.valSeeds       = nnz(valSeed);
info.patchesPerEpoch = numel(trainIdx);
info.balanceByImage  = cfg.Patch.balanceByImage;

% Channel stack plus the seed heat-map. trainNet sizes the network from this
% rather than assuming 3, so cfg.Vessel.enabled needs no change there.
info.numChannels = size(chans{1}, 3) + 1;

% Patches drawn per training image this epoch, so a run's record shows how the
% epoch was actually spent rather than only how many patches it held.
trainImageIdx = unique(imgIdx(~valSeed))';
info.patchesPerImage = table( ...
    images(trainImageIdx), ...
    arrayfun(@(i) nnz(imgIdx(~valSeed) == i), trainImageIdx)', ...
    arrayfun(@(i) nnz(imgIdx(trainIdx) == i), trainImageIdx)', ...
    'VariableNames', {'Image', 'Seeds', 'Patches'});
info.patchesPerImage.Share = ...
    info.patchesPerImage.Patches / max(sum(info.patchesPerImage.Patches), 1);
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function ds = makeDatastore(rowIdx, imgIdx, seeds, chans, segMasks, cfg, augment)
ds = arrayDatastore(rowIdx, 'OutputType', 'same');
ds = transform(ds, @(k) makeSample(k, imgIdx, seeds, chans, segMasks, cfg, augment));
end


function out = makeSample(k, imgIdx, seeds, chans, segMasks, cfg, augment)
% One patch. k is the seed row index handed over by the arrayDatastore.
if iscell(k)
    k = k{1};
end

seed = seeds(k, :);
C    = chans{imgIdx(k)};
gt   = segMasks{k};              % this seed's own segment, not the whole image

if augment
    center = seed + randi([-cfg.Patch.jitter, cfg.Patch.jitter], 1, 2);
else
    center = seed;
end

[X, rows, cols] = cnngvf.seedPatch(C, seed, center, cfg);

% Crop the label with the geometry seedPatch just resolved, zero-padding the
% same way so X and T stay pixel-aligned.
p   = cfg.Patch.size;
fg  = false(p, p);
inR = rows >= 1 & rows <= size(gt, 1);
inC = cols >= 1 & cols <= size(gt, 2);
fg(inR, inC) = gt(rows(inR), cols(inC));

if augment
    % Dihedral group only. ../docs/spec.md warns that elastic warps distort
    % the thin fronds this network exists to find.
    if rand < 0.5, X = fliplr(X); fg = fliplr(fg); end
    if rand < 0.5, X = flipud(X); fg = flipud(fg); end
    r = randi(4) - 1;
    if r > 0, X = rot90(X, r); fg = rot90(fg, r); end
end

T = single(cat(3, ~fg, fg));   % one-hot [background, RNV]
out = {X, T};
end
