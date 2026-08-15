function [perImage, summary] = run_cv(arm, cfg)
%RUN_CV Cross-validated training and scoring: every image scored out-of-sample.
%
%  run_cv                       % arm "unclustered" (the proposed method)
%  run_cv("clustered")          % the baseline
%  run_cv(arm, cfg)             % custom config
%  [perImage, summary] = run_cv(...)
%
%  Why this exists: run_batch scores all 17 ground-truth images, but 13 of them
%  trained the model, so its headline is mostly in-sample. Balanced sampling
%  moved that headline +0.012 while moving the out-of-sample mean +0.068 - the
%  reported number was hiding the result. Cross-validation gives one
%  out-of-sample score per image instead of four.
%
%  Per fold:
%     test  = the fold's images        (cnngvf.cvFolds, area-stratified)
%     inner = cfg.CV.innerVal images   (best-checkpoint selection)
%     train = everything else
%  The test images are excluded from training AND from checkpoint selection, so
%  nothing about them influences the model that scores them.
%
%  arm : "unclustered" - train from DRAC_RNV_Points_Unclustered.csv (default)
%        "clustered"   - train from the clustered centres
%  Both arms use the SAME folds, so they can be compared paired by image.
%
%  Writes models/cv/<arm>_fold<k>.mat and result/cv/<arm>/fold<k>/, then
%  concatenates the folds into result/cv/<arm>/per_image.csv.
%
%  Runtime is cfg.CV.k training runs: ~2.4 h per fold at 30 epochs, so ~12 h.
%  Dry-run it first with cfg.Net.epochs = 1.
%
%  See also CNNGVF.CVFOLDS, RUN_BATCH, RUN_EVAL, CNNGVF.EVALUATE.

if nargin < 1 || strlength(arm) == 0
    arm = "unclustered";
end
arm = lower(string(arm));
if nargin < 2
    cfg = cnngvf.defaults();
end

% perSeed differs per arm so both arms see a comparable epoch, exactly as the
% non-CV runners do: the unclustered set has ~4.6x the seeds, so 4 crops each
% matches the clustered set's 16. Leaving it at the default 16 for both would
% give the unclustered arm 1232 patches per epoch instead of ~400 - six hours
% per fold rather than two, and an unfair comparison on top.
switch arm
    case "unclustered"
        seedCsv = fullfile(cfg.Paths.Img, 'DRAC_RNV_Points_Unclustered.csv');
        cfg.Patch.perSeed = 4;
    case "clustered"
        seedCsv = cfg.Paths.Localization;
        cfg.Patch.perSeed = 16;
    otherwise
        error('cnngvf:runCv:badArm', 'Unknown arm "%s". Use "unclustered" or "clustered".', arm);
end
if ~isfile(seedCsv)
    error('cnngvf:runCv:noCsv', 'Seed file not found: %s', seedCsv);
end

cfg.EvalMode = "image";   % the union mask vs FilledGT; the unit both arms share

[folds, foldInfo] = cnngvf.cvFolds(cfg);
k = numel(folds);

cvRoot   = fullfile(cfg.Paths.Output, 'cv', char(arm));
modelDir = fullfile(fileparts(cfg.Paths.Model), 'cv');
for d = string({cvRoot, modelDir})
    if ~exist(d, 'dir'), mkdir(d); end
end

fprintf('run_cv | arm=%s | %d folds | device: %s\n', arm, k, cfg.GpuName);
fprintf('Seeds  : %s\n', seedCsv);
fprintf('Output : %s\n\n', cvRoot);
disp(foldInfo);

tAll = tic;
for f = 1:k
    test = folds{f};
    rest = setdiff(foldInfo.Image, test, 'stable');
    inner = rest(1:min(cfg.CV.innerVal, numel(rest) - 1));

    fprintf('\n=== fold %d/%d ===\n', f, k);
    fprintf('  test  (%d): %s\n', numel(test),  strjoin(test',  ', '));
    fprintf('  inner (%d): %s\n', numel(inner), strjoin(inner', ', '));

    fc = cfg;
    fc.Paths.LocalizationTrain = seedCsv;
    fc.Split.Exclude = test;    % never seen by training or checkpoint selection
    fc.Split.Val     = inner;
    fc.Paths.Model   = fullfile(modelDir, sprintf('%s_fold%d.mat', arm, f));

    cnngvf.trainNet(fc);

    % Score this fold's model on this fold's held-out images only.
    bc = fc;
    bc.Paths.Localization = seedCsv;   % inference seeds match the arm
    bc.Images             = test;
    bc.Paths.Output       = fullfile(cvRoot, sprintf('fold%d', f));
    run_batch(bc);
end

[perImage, summary] = aggregate(cvRoot, cfg, k);

fprintf('\n=== %s: cross-validated over %d images, all out-of-sample ===\n', ...
    arm, height(perImage));
fprintf('mean IoU %.4f | median %.4f | F1 %.4f | precision %.4f | recall %.4f\n', ...
    summary.meanIou, summary.medianIou, summary.meanF1, ...
    summary.meanPrecision, summary.meanRecall);
fprintf('total runtime %.1f h\n', toc(tAll) / 3600);
fprintf('Records: %s\n', fullfile(cvRoot, 'per_image.csv'));
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [perImage, summary] = aggregate(cvRoot, cfg, k)
%AGGREGATE Concatenate the per-fold score tables into one out-of-sample table.
parts = {};
for f = 1:k
    p = fullfile(cvRoot, sprintf('fold%d', f), 'per_image.csv');
    if isfile(p)
        parts{end + 1} = readtable(p); %#ok<AGROW>
    else
        warning('cnngvf:runCv:missingFold', 'No scores for fold %d (%s)', f, p);
    end
end
if isempty(parts)
    error('cnngvf:runCv:noResults', 'No fold produced any scores.');
end

perImage = sortrows(vertcat(parts{:}), 'Image');
writetable(perImage, fullfile(cvRoot, 'per_image.csv'));

summary = struct( ...
    'mode',           "image", ...
    'nUnits',         height(perImage), ...
    'meanIou',        mean(perImage.Iou),  'medianIou', median(perImage.Iou), ...
    'meanF1',         mean(perImage.F1),   'medianF1',  median(perImage.F1), ...
    'meanPrecision',  mean(perImage.Precision), ...
    'meanRecall',     mean(perImage.Recall), ...
    'unclaimedGtPx',  0, ...
    'missingGtFiles', 0);

sc = cfg;
sc.Paths.Output = cvRoot;
sc.EvalMode     = "image";
cnngvf.writeSummary(fullfile(cvRoot, 'run_summary.txt'), sc, perImage, summary, []);
end
