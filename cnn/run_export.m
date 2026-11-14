function outDir = run_export(cfg, outDir)
%RUN_EXPORT Write the bundle the PyTorch trainer (colab/) reads.
%
%  run_export                         % current configuration, its CV networks
%  run_export(cfg)                    % custom config
%  run_export(cfg, outDir)            % default <Output>/colab_bundle
%
%  Everything the PyTorch side needs, so it never reads datasets/ and cannot
%  diverge on loading, labelling or folds (docs/briefs/colab-parity.md):
%
%    bundle.mat          the cfg values training and inference use
%    images/<stem>.mat   chans (loadChannels), label (FilledGT labelmatrix),
%                        seeds, comp (cnngvf.seedLabels; 0 = dropped from
%                        training), negA and negB (cnngvf.negativeSeeds)
%    folds/fold<k>.mat   test and inner-validation images, the patch counts the
%                        MATLAB run recorded, and that fold network's weights
%                        and input mean
%    fixtures.mat        seedPatch crops, dihedral transforms, generalizedDice
%                        values, one network forward pass and one whole-image P
%                        (CPU), each produced by the MATLAB code itself
%
%  The recipe the trainer runs (encoder, multiScaleSeed, epochsCv, learnRate,
%  lrMin, miniBatch, the seedless network's seedless*, and Patch.negSeedFrac)
%  comes from cfg,
%  so a bundle exported before a recipe change trains the old one: re-export.
%
%  .mat v7, which scipy.io.loadmat reads. Strings are saved as cellstr for the
%  same reason. A few minutes; nothing trains.
%
%  See also CNNGVF.SEEDLABELS, RUN_CV, RUN_CACHE.

if nargin < 1
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);
end
if nargin < 2
    outDir = fullfile(cfg.Paths.Output, 'colab_bundle');
end
outDir = char(outDir);
for d = {outDir, fullfile(outDir, 'images'), fullfile(outDir, 'folds')}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

M = cnngvf.manifest(cfg);
[names, seeds] = cnngvf.loadPoints(cfg);
folds = cnngvf.cvFolds(cfg);
k = numel(folds);

fprintf('run_export | arm %s | %d images | %d folds -> %s\n', cfg.Arm, height(M), k, outDir);

% --- bundle.mat ------------------------------------------------------------
b.arm         = char(cfg.Arm);
b.images      = cellstr(M.Name);
b.datasets    = cellstr(M.Dataset);
b.k           = k;
b.patch       = cfg.Patch.size;
b.sigmaSeed   = cfg.Patch.sigmaSeed;
b.jitter      = cfg.Patch.jitter;
b.perSeed     = cfg.Patch.perSeed;
b.seedTolerance  = cfg.Patch.seedTolerance;
b.balanceByImage = cfg.Patch.balanceByImage;
b.negSeedFrac = cfg.Patch.negSeedFrac;
b.encoderDepth = cfg.Net.encoderDepth;
b.encoder     = char(cfg.Net.encoder);
b.multiScaleSeed = cfg.Net.multiScaleSeed;
b.epochsCv    = cfg.Net.epochsCv;
b.lrMin       = cfg.Net.lrMin;
b.miniBatch   = cfg.Net.miniBatch;
b.inferBatch  = cfg.Net.inferBatch;
b.learnRate   = cfg.Net.learnRate;
b.seedlessEpochs    = cfg.Net.seedlessEpochs;
b.seedlessMiniBatch = cfg.Net.seedlessMiniBatch;
b.seedlessCopies    = cfg.Net.seedlessCopies;
b.exported    = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
save(fullfile(outDir, 'bundle.mat'), '-struct', 'b', '-v7');

% --- images/ ---------------------------------------------------------------
for i = 1:height(M)
    [~, im.chans] = cnngvf.loadChannels(M.ImgPath(i), cfg);
    im.chans   = single(im.chans);
    im.seeds   = seeds(names == M.Name(i), :);
    gt = cnngvf.loadMask(M.GtPath(i));
    [im.comp, L] = cnngvf.seedLabels(gt, im.seeds, cfg.Patch.seedTolerance);
    [im.negA, im.negB] = cnngvf.negativeSeeds(gt, im.chans(:, :, 1), im.seeds);
    im.label   = uint16(L);
    im.name    = char(M.Name(i));
    im.dataset = char(M.Dataset(i));
    [~, stem] = fileparts(M.Name(i));
    save(fullfile(outDir, 'images', char(stem + ".mat")), '-struct', 'im', '-v7');
end

% --- folds/ ----------------------------------------------------------------
modelDir = fullfile(fileparts(cfg.Paths.Model), 'cv');
for f = 1:k
    s = load(fullfile(modelDir, sprintf('%s_fold%d.mat', cfg.Arm, f)), 'net', 'cfg', 'info');
    % The network must have been trained with this fold held out, or its
    % weights reproduce nothing here.
    assert(isempty(setxor(string(s.cfg.Split.Exclude), folds{f})), ...
        'run_export:staleFold', 'Fold %d network was trained on other folds than cvFolds gives.', f);

    fo.test  = cellstr(folds{f});
    fo.inner = cellstr(s.info.valImages);
    fo.patchesPerEpoch = s.info.patchesPerEpoch;
    fo.trainSeeds = s.info.trainSeeds;
    fo.valSeeds   = s.info.valSeeds;
    fo.layer  = cellstr(s.net.Learnables.Layer);
    fo.param  = cellstr(s.net.Learnables.Parameter);
    fo.values = cellfun(@(v) gather(extractdata(v)), s.net.Learnables.Value, 'UniformOutput', false);
    fo.mean   = single(s.net.Layers(1).Mean);
    save(fullfile(outDir, 'folds', sprintf('fold%d.mat', f)), '-struct', 'fo', '-v7');
end

% --- fixtures.mat ----------------------------------------------------------
rng(0);
fx = struct();

% Crops, including centres off every edge (zero padding) and jittered centres.
fx.patchChans  = rand(700, 800, 1, 'single');
fx.patchSeed   = [400 350; 10 10; 790 690; 3.4 600.6; 400.5 2.5; 250 300];
fx.patchCenter = fx.patchSeed + [0 0; 0 0; 0 0; 20 -31; -32 17; 31.5 -0.5];
n = size(fx.patchSeed, 1);
fx.patchX = zeros(cfg.Patch.size, cfg.Patch.size, 2, n, 'single');
fx.patchRows = zeros(n, cfg.Patch.size);
fx.patchCols = zeros(n, cfg.Patch.size);
for s = 1:n
    [fx.patchX(:, :, :, s), fx.patchRows(s, :), fx.patchCols(s, :)] = ...
        cnngvf.seedPatch(fx.patchChans, fx.patchSeed(s, :), fx.patchCenter(s, :), cfg);
end

% The dihedral operations prepareData applies.
fx.dihA      = reshape(1:12, 3, 4);
fx.dihFlipLR = fliplr(fx.dihA);
fx.dihFlipUD = flipud(fx.dihA);
fx.dihRot1   = rot90(fx.dihA, 1);
fx.dihRot2   = rot90(fx.dihA, 2);
fx.dihRot3   = rot90(fx.dihA, 3);

% generalizedDice per observation; observation 3 has no lesion.
Y  = rand(32, 32, 2, 3, 'single');
Y  = Y ./ sum(Y, 3);
fg = rand(32, 32, 1, 3) > 0.7;
fg(:, :, 1, 3) = false;
T  = single(cat(3, ~fg, fg));
fx.gdY = Y;
fx.gdT = T;
fx.gd  = extractdata(generalizedDice(dlarray(Y, 'SSCB'), dlarray(T, 'SSCB')));
fx.gdLoss = 1 - mean(fx.gd, 'all');

% The fold-1 network on CPU: one patch, and the whole P of the fold-1 test image
% with the fewest seeds. CPU, because MATLAB's GPU P is a few 1e-4 from its own
% CPU result (docs/results/2026-11-02-colab-parity-inference.md).
s   = load(fullfile(modelDir, sprintf('%s_fold1.mat', cfg.Arm)), 'net');
cnt = arrayfun(@(n) nnz(names == n), folds{1});
cnt(cnt == 0) = Inf;
[~, j] = min(cnt);
img = folds{1}(j);
row = find(M.Name == img, 1);
[~, chans] = cnngvf.loadChannels(M.ImgPath(row), cfg);
sd  = seeds(names == img, :);
fx.fwdImage = char(img);
fx.fwdX = cnngvf.seedPatch(chans, sd(1, :), sd(1, :), cfg);
fx.fwdY = gather(predict(s.net, fx.fwdX));
cpu = cfg;
cpu.UseGpu = false;
fx.fwdP = cnngvf.inferProbMap(chans, sd, s.net, cpu);

save(fullfile(outDir, 'fixtures.mat'), '-struct', 'fx', '-v7');
fprintf('run_export | done\n');
end
