function [perUnit, summary] = run_batch(cfg)
%RUN_BATCH Segment every image in the localization CSV and score each segment.
%
%  run_batch                          % defaults; loads cfg.Paths.Model
%  run_batch(cfg)                     % custom config
%  [perUnit, summary] = run_batch(...)
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
end

if ~isfile(cfg.Paths.Model)
    error('cnngvf:runBatch:noModel', ...
        'No trained network at %s. Run run_train first.', cfg.Paths.Model);
end
loaded = load(cfg.Paths.Model, 'net');
net    = loaded.net;

[names, seeds] = rnv.loadSeeds(cfg);
images = unique(names, 'stable');

% Optional restriction to a subset - run_cv passes a fold's held-out images so
% the model is scored only on data it never saw. Empty means all.
if isfield(cfg, 'Images') && ~isempty(cfg.Images)
    wanted = ismember(images, string(cfg.Images));
    if ~any(wanted)
        error('cnngvf:runBatch:noneRequested', ...
            'None of cfg.Images (%s) appear in %s', ...
            strjoin(string(cfg.Images), ', '), cfg.Paths.Localization);
    end
    images = images(wanted);
end

% The localization CSVs cover more images than RNV-DRAC22NV actually ships -
% the unclustered set names 41, of which 17 are present. Skip what is not on
% disk rather than dying on the first one.
onDisk = arrayfun(@(n) isfile(fullfile(cfg.Paths.Img, char(n))), images);
if ~all(onDisk)
    fprintf('Skipping %d of %d images named in the CSV but not present in %s\n', ...
        nnz(~onDisk), numel(images), cfg.Paths.Img);
    images = images(onDisk);
end
if isempty(images)
    error('cnngvf:runBatch:noImages', ...
        'None of the images in %s are present in %s', cfg.Paths.Localization, cfg.Paths.Img);
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

fprintf('run_batch | cnn+gvf | %d images, %d seeds | device: %s\n', ...
    numel(images), numel(names), cfg.GpuName);
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
        fullfile(cfg.Paths.Img, char(imgName)), imgSeeds, net, cfg);
    elapsed(i) = toc(tImg);

    results(i).image   = imgName;
    results(i).seeds   = imgSeeds;
    results(i).perSeed = perSeed;
    results(i).mask    = mask;

    imwrite(mask,        fullfile(dirs.combined, char(stem + ".png")));
    imwrite(im2uint8(P), fullfile(dirs.prob,     char(stem + ".png")));
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
else
    fprintf('No ground truth found to score against.\n');
end
fprintf('Records: %s\n', fullfile(cfg.Paths.Output, csvName));
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

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
