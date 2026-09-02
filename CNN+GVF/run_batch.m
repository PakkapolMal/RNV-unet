function [perUnit, summary] = run_batch(cfg)
%RUN_BATCH Segment every seeded image in the domain and score the result.
%
%  run_batch                          % defaults; loads cfg.Paths.Model
%  run_batch("DRACTUH_unclustered")   % a named arm (see cnngvf.arm)
%  run_batch(cfg)                     % custom config
%  [perUnit, summary] = run_batch(...)
%
%  The domain spans every dataset in cfg.ActiveDatasets; cnngvf.manifest
%  resolves each image to its own dataset's mask, so DRAC22 and TUH are
%  segmented by one pass without either knowing about the other.
%
%  Output layout under cfg.Paths.Output:
%    combined/<stem>.png        union mask per image
%    segment/<stem>-<x>-<y>.png one mask per seed, named like the ground truth
%    overlay/<stem>.png         image + seeds + final contours
%    prob/<stem>.png            the CNN probability map
%    per_segment.csv            per-seed scores   (cfg.EvalMode "segment")
%    per_image.csv              per-image scores  (cfg.EvalMode "image")
%    run_summary.txt            aggregate scores and the config behind them
%
%  Scoring is governed by cfg.EvalMode; see cnngvf.evaluate. Images with no
%  ground truth are segmented and written but not scored.
%
%  This function segments AND scores. To re-score masks that already exist -
%  after changing cfg.EvalMode, or against corrected ground truth - use
%  run_eval, which reads the cached masks and skips the network entirely.
%
%  See also RUN_EVAL, CNNGVF.SEGMENTIMAGE, CNNGVF.EVALUATE, RUN_TRAIN.

if nargin < 1
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);   % run_batch("DRACTUH_unclustered")
end

if ~isfile(cfg.Paths.Model)
    error('cnngvf:runBatch:noModel', ...
        'No trained network at %s. Run run_train first.', cfg.Paths.Model);
end
loaded = load(cfg.Paths.Model, 'net');
net    = loaded.net;

% The domain, and the points that seed it. Both span every active dataset;
% loadPoints has already dropped points on images with no ground truth, so
% there is no on-disk existence check to do here.
M = cnngvf.manifest(cfg);
[names, seeds] = cnngvf.loadPoints(cfg);
images = unique(names, 'stable');

% Optional restriction to a subset - run_cv passes a fold's held-out images so
% the model is scored only on data it never saw. Empty means all.
if isfield(cfg, 'Images') && ~isempty(cfg.Images)
    wanted = ismember(images, string(cfg.Images));
    if ~any(wanted)
        error('cnngvf:runBatch:noneRequested', ...
            'None of cfg.Images (%s) carry points in the active datasets (%s).', ...
            strjoin(string(cfg.Images), ', '), strjoin(cfg.ActiveDatasets, ', '));
    end
    images = images(wanted);
end

dirs = struct('combined', fullfile(cfg.Paths.Output, 'combined'), ...
              'segment',  fullfile(cfg.Paths.Output, 'segment'), ...
              'overlay',  fullfile(cfg.Paths.Output, 'overlay'), ...
              'prob',     fullfile(cfg.Paths.Output, 'prob'));
% Clear before writing. Per-seed masks are named by seed coordinates, so when
% the localization CSV changes, the previous run's files survive under their
% old names and the directory ends up holding two generations at once. That is
% invisible to run_batch, which scores what it just computed, but run_eval
% recovers seeds from these filenames and would score both - it read 4 seeds
% for 573.png where the run produced 2. A run's output directory must contain
% exactly that run's output.
for d = string(struct2cell(dirs))'
    if exist(d, 'dir')
        old = dir(fullfile(d, '*.png'));
        for k = 1:numel(old)
            delete(fullfile(old(k).folder, old(k).name));
        end
    else
        mkdir(d);
    end
end

% Probability-map cache. Off unless a caller asks: cfg.Cache.Prob names the
% directory, cfg.Fold the fold whose held-out set this run scores. A cache
% written without a fold cannot be shown to be out-of-sample, and
% cnngvf.probCache refuses to score it later - which is the intended behaviour
% for a plain run_batch, whose model trained on most of the domain.
cacheDir = "";
if isfield(cfg, 'Cache') && isfield(cfg.Cache, 'Prob')
    cacheDir = string(cfg.Cache.Prob);
end
fold = NaN;
if isfield(cfg, 'Fold') && ~isempty(cfg.Fold)
    fold = cfg.Fold;
end
% Every score is tied to the localization CSVs in place when it was measured,
% and those live outside version control and have been revised repeatedly.
csvMtimes = csvStamps(cfg);

fprintf('run_batch | arm %s | %d images, %d seeds | device: %s\n', ...
    armLabel(cfg), numel(images), numel(names), cfg.GpuName);
if cacheDir ~= ""
    fprintf('Prob cache: %s (fold %d)\n', cacheDir, fold);
end
fprintf('Datasets: %s\n', strjoin(cfg.ActiveDatasets, ' + '));
fprintf('Output: %s\n\n', cfg.Paths.Output);

n = numel(images);
results(n) = struct('image', "", 'seeds', [], 'perSeed', {{}}, 'mask', []);
elapsed = zeros(n, 1);

for i = 1:n
    imgName   = images(i);
    imgSeeds  = seeds(names == imgName, :);
    [~, stem] = fileparts(imgName);

    tImg = tic;
    [mask, perSeed, contours, img, P] = cnngvf.segmentImage( ...
        M.ImgPath(M.Name == imgName), imgSeeds, net, cfg);
    elapsed(i) = toc(tImg);

    results(i).image   = imgName;
    results(i).seeds   = imgSeeds;
    results(i).perSeed = perSeed;
    results(i).mask    = mask;

    imwrite(mask,        fullfile(dirs.combined, char(stem + ".png")));
    imwrite(im2uint8(P), fullfile(dirs.prob,     char(stem + ".png")));

    % The full-precision P, when a caller asked for it. run_cv does, per fold,
    % so that every downstream post-processing variant can be scored off cached
    % P in seconds instead of paying for a fresh 18 h cross-validation. The
    % uint8 dump above is the human-viewable artefact and is not this.
    if cacheDir ~= ""
        cnngvf.probCache("write", cacheDir, imgName, P, struct( ...
            'fold',      fold, ...
            'arm',       cfg.Arm, ...
            'netFile',   string(cfg.Paths.Model), ...
            'csvMtimes', csvMtimes));
    end
    rnv.saveOverlay(fullfile(dirs.overlay, char(stem + ".png")), img, imgSeeds, contours);

    % Per-seed masks are written under the ground truth's own naming scheme so
    % a prediction and its target can be opened side by side.
    for s = 1:size(imgSeeds, 1)
        m = perSeed{s};
        if isempty(m)
            m = false(img.h, img.w);
        end
        imwrite(m, fullfile(dirs.segment, ...
            sprintf('%s-%d-%d.png', stem, round(imgSeeds(s, 1)), round(imgSeeds(s, 2)))));
    end

    fprintf('  [%3d/%3d] %-10s %d seed(s), %7d px, %5.1f s\n', ...
        i, n, imgName, size(imgSeeds, 1), nnz(mask), elapsed(i));
end

[perUnit, summary] = cnngvf.evaluate(results, cfg);

% Never lose a run's scores to a locked output file. Excel holds a .csv open
% for writing, which once threw away the scoring of a 2.5 h training run at
% the very last line - the masks were all on disk, but the numbers were gone.
csvName = writeScores(cfg, perUnit);
try
    cnngvf.writeSummary(fullfile(cfg.Paths.Output, 'run_summary.txt'), ...
        cfg, perUnit, summary, elapsed);
catch err
    warning('cnngvf:runBatch:noSummary', 'Could not write run_summary.txt: %s', err.message);
end

fprintf('\nDone in %.1f s. %d masks written to %s\n', sum(elapsed), n, dirs.combined);
if summary.nUnits > 0
    unitName = cfg.EvalMode + "s";
    fprintf('Scored %d %s: mean IoU %.4f (median %.4f), mean F1 %.4f (median %.4f)\n', ...
        summary.nUnits, unitName, summary.meanIou, summary.medianIou, ...
        summary.meanF1, summary.medianF1);
    fprintf('Precision %.4f, Recall %.4f', summary.meanPrecision, summary.meanRecall);
    if cfg.EvalMode == "segment"
        fprintf(' | %d GT px belong to lesions no seed claimed', summary.unclaimedGtPx);
    end
    fprintf('\n');
    if height(summary.perDataset) > 1
        fprintf('Per dataset:\n');
        for i = 1:height(summary.perDataset)
            fprintf('  %-8s n=%-3d IoU %.4f  F1 %.4f  prec %.4f  rec %.4f\n', ...
                summary.perDataset.Dataset(i), summary.perDataset.GroupCount(i), ...
                summary.perDataset.mean_Iou(i), summary.perDataset.mean_F1(i), ...
                summary.perDataset.mean_Precision(i), summary.perDataset.mean_Recall(i));
        end
    end
else
    fprintf('No ground truth found to score against.\n');
end
fprintf('Records: %s\n', fullfile(cfg.Paths.Output, csvName));
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function stamps = csvStamps(cfg)
%CSVSTAMPS Modification times of the localization CSVs behind this run, so a
%  cached P records which revision of them produced it.
stamps = struct();
for d = cfg.Datasets
    if ~ismember(d.Name, cfg.ActiveDatasets)
        continue;
    end
    if cfg.Points == "clustered"
        p = d.Clustered;
    else
        p = d.Unclustered;
    end
    info = dir(p);
    if isempty(info)
        stamps.(char(d.Name)) = "missing";
    else
        stamps.(char(d.Name)) = string(datetime(info.datenum, ...
            'ConvertFrom', 'datenum', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    end
end
end


function s = armLabel(cfg)
if strlength(cfg.Arm) > 0
    s = cfg.Arm;
else
    s = "(custom cfg)";
end
end


function csvName = writeScores(cfg, perUnit)
%WRITESCORES Write the score table, falling back to a timestamped name when
%  the usual one is locked - typically because it is open in Excel.
csvName = "per_" + cfg.EvalMode + ".csv";
try
    writetable(perUnit, fullfile(cfg.Paths.Output, csvName));
catch
    csvName = "per_" + cfg.EvalMode + "_" + ...
        string(datetime('now', 'Format', 'yyyyMMdd_HHmmss')) + ".csv";
    writetable(perUnit, fullfile(cfg.Paths.Output, csvName));
    warning('cnngvf:runBatch:csvLocked', ...
        'per_%s.csv is locked (open in Excel?); wrote %s instead.', ...
        cfg.EvalMode, csvName);
end
end
