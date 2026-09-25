function [perImage, summary] = run_cv(cfg)
%RUN_CV Cross-validated training and scoring: every image scored out-of-sample.
%
%  run_cv                             % the current configuration
%  run_cv("unclustered")              % a frozen arm (see cnngvf.arm)
%  run_cv(cfg)                        % custom config
%  [perImage, summary] = run_cv(...)
%
%  The only protocol whose numbers are quotable: run_batch scores images that
%  trained its model. Cross-validation gives one out-of-sample score per image.
%
%  Per fold:
%     test  = the fold's images        (cnngvf.cvFolds, area-stratified)
%     inner = cfg.CV.innerVal images   (best-checkpoint selection)
%     train = everything else
%  The test images are excluded from training AND from checkpoint selection, so
%  nothing about them influences the model that scores them.
%
%  Arms over the SAME domain use the SAME folds (cnngvf.cvFolds is
%  deterministic), so they compare paired by image. Arms over DIFFERENT
%  domains do not - "unclustered" is 41 DRAC22 images, the current
%  configuration is those 41 plus 27 TUH.
%
%  Writes models/cv/<arm>_fold<k>.mat and <Output>/cv/fold<k>/, then
%  concatenates the folds into <Output>/cv/per_image.csv. Output is already
%  arm-specific, so this cannot overwrite an earlier arm's record.
%
%  Also caches each fold's probability maps to <Output>/cv/probcache/, stamped
%  with the fold that held the image out. That is what lets a post-processing
%  change be scored off cached P in seconds instead of another 18 h here, and
%  cnngvf.probCache re-checks the stamp before anything is scored.
%
%  cfg.CV.skipTrain reuses existing fold networks instead of training them:
%
%     cfg = cnngvf.arm(cnngvf.defaults(), "unclustered_boundary");
%     cfg.CV.skipTrain = true;
%     run_cv(cfg)                             % minutes, not 18 h
%
%  For a change with no trainable parameter in it - a claiming rule, a
%  threshold, the whole reduced pipeline - retraining would destroy the pairing
%  and buy nothing. cfg.CV.ModelArm says whose networks to borrow; training
%  never writes to that path.
%
%  It is also how an interrupted run RESUMES. On an arm's own networks,
%  skipTrain reuses every fold already trained and trains only the missing ones,
%  so a 5-fold run killed during fold 4 costs 2 folds to finish rather than 5:
%
%     cfg = cnngvf.defaults(); cfg.CV.skipTrain = true;
%     run_cv(cfg)
%
%  Folds already scored are re-scored from their kept networks, which is cheap
%  and keeps every fold's output consistent with the code that aggregated it.
%
%  Folds are pooled across datasets and stratified by lesion area. per_image.csv
%  carries a Dataset column and the summary prints a per-dataset breakout, so a
%  pooled run can be inspected per device without being re-run.
%
%  Runtime is cfg.CV.k training runs. Dry-run it first with cfg.Net.epochs = 1.
%
%  See also CNNGVF.ARM, CNNGVF.CVFOLDS, RUN_BATCH, RUN_EVAL, CNNGVF.EVALUATE.

if nargin < 1
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);
end
arm = cfg.Arm;
cfg.EvalMode = "image";   % the union mask vs whole-image GT; the shared unit

% 20 epochs, not the 30 a single training run uses. Cross-validation pays for
% training k times. OutputNetwork="best-validation" keeps the best checkpoint,
% so stopping earlier costs accuracy at the margin rather than correctness -
% but it does mean a CV figure measures a slightly less-trained network than
% run_train deploys, and will if anything understate it.
cfg.Net.epochs = 20;

[folds, foldInfo] = cnngvf.cvFolds(cfg);
k = numel(folds);

% cfg.Paths.Output is already arm-specific, so 'cv' under it is enough.
cvRoot   = fullfile(cfg.Paths.Output, 'cv');
modelDir = fullfile(fileparts(cfg.Paths.Model), 'cv');
for d = string({cvRoot, modelDir})
    if ~exist(d, 'dir'), mkdir(d); end
end

fprintf('run_cv | arm=%s | %d folds | device: %s\n', arm, k, cfg.GpuName);
fprintf('Datasets : %s\n', strjoin(cfg.ActiveDatasets, ' + '));
fprintf('Points   : %s\n', cfg.Points);
fprintf('Output   : %s\n\n', cvRoot);
disp(groupsummary(foldInfo, {'Fold', 'Dataset'}));

tAll = tic;
for f = 1:k
    test = folds{f};
    rest = setdiff(foldInfo.Image, test, 'stable');

    % Spread the inner-validation picks across the area range. rest is sorted
    % by lesion area descending, so rest(1:innerVal) would always hand
    % best-validation checkpoint selection the two largest lesions - in the
    % 1-epoch dry run that made fold 3 keep its untrained network and predict
    % 0 px on all three held-out images. Interior evenly spaced positions give
    % a mid-range, representative pair instead.
    n   = numel(rest);
    nv  = min(cfg.CV.innerVal, n - 1);
    pos = round(linspace(1, n, nv + 2));
    inner = rest(pos(2:end-1));

    fprintf('\n=== fold %d/%d ===\n', f, k);
    fprintf('  test  (%d): %s\n', numel(test),  strjoin(test',  ', '));
    fprintf('  inner (%d): %s\n', numel(inner), strjoin(inner', ', '));

    % The arm already fixed datasets, point set and channels; a fold only
    % changes which images are held out and where the model goes.
    fc = cfg;
    fc.Split.Exclude = test;    % never seen by training or checkpoint selection
    fc.Split.Val     = inner;
    % Borrowing another arm's networks is read-only; training an arm's own is
    % not. Splitting on that is what makes skipTrain safe to RESUME with: a
    % 5-fold run that died in fold 4 keeps folds 1-3 and trains only what is
    % missing, instead of burning another 9 h reproducing them.
    borrowed       = cfg.CV.ModelArm ~= arm;
    fc.Paths.Model = fullfile(modelDir, sprintf('%s_fold%d.mat', cfg.CV.ModelArm, f));

    if cfg.CV.skipTrain && isfile(fc.Paths.Model)
        fprintf('  reusing %s\n', fc.Paths.Model);
    elseif borrowed
        % Never train into the path of an arm we are only borrowing from.
        error('cnngvf:runCv:noFoldModel', ...
            ['cfg.CV.ModelArm is "%s" but there is no network at %s.\n' ...
             'Training would overwrite that arm''s fold networks, so it is ' ...
             'refused. Run the borrowed arm''s own cross-validation first.'], ...
            cfg.CV.ModelArm, fc.Paths.Model);
    else
        fc.Paths.Model = fullfile(modelDir, sprintf('%s_fold%d.mat', arm, f));
        cnngvf.trainNet(fc);
    end

    % Score this fold's model on this fold's held-out images only.
    bc = fc;
    bc.Images       = test;
    bc.Paths.Output = fullfile(cvRoot, sprintf('fold%d', f));
    bc.Fold         = f;                                  % stamped into the P cache
    bc.Cache.Prob   = fullfile(cvRoot, 'probcache');
    run_batch(bc);
end

[perImage, summary] = aggregate(cvRoot, cfg, k);

fprintf('\n=== %s: cross-validated over %d images, all out-of-sample ===\n', ...
    arm, height(perImage));
fprintf('mean IoU %.4f | median %.4f | F1 %.4f | precision %.4f | recall %.4f\n', ...
    summary.meanIou, summary.medianIou, summary.meanF1, ...
    summary.meanPrecision, summary.meanRecall);
if height(summary.perDataset) > 1
    fprintf('\nPer dataset (the folds are pooled; this is a breakout, not a protocol):\n');
    for i = 1:height(summary.perDataset)
        fprintf('  %-8s n=%-3d IoU %.4f  F1 %.4f  prec %.4f  rec %.4f\n', ...
            summary.perDataset.Dataset(i), summary.perDataset.GroupCount(i), ...
            summary.perDataset.mean_Iou(i), summary.perDataset.mean_F1(i), ...
            summary.perDataset.mean_Precision(i), summary.perDataset.mean_Recall(i));
    end
end
fprintf('\ntotal runtime %.1f h\n', toc(tAll) / 3600);
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

perImage = vertcat(parts{:});

% readtable gives Image back as a cell of char, which fprintf's %s cannot
% format - cnngvf.evaluate builds it as a string array. Normalise before
% anything downstream touches it.
perImage.Image = string(perImage.Image);
if ismember('Dataset', perImage.Properties.VariableNames)
    perImage.Dataset = string(perImage.Dataset);
end
perImage = sortrows(perImage, 'Image');

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
summary.perDataset = groupsummary(perImage, 'Dataset', 'mean', ...
    {'Iou', 'F1', 'Precision', 'Recall'});

% The header names the thing that produced these numbers. It used to inherit
% cfg's default Model and Localization, so an unclustered CV run recorded
% itself as rnvNet.mat scored from the clustered CSV - wrong on both counts,
% in the file that exists to be the provenance record. k models produced this
% table, so say so.
sc = cfg;
sc.Paths.Output = cvRoot;
sc.EvalMode     = "image";
sc.Paths.Model  = sprintf('%d-fold CV, %s (see fold*/run_summary.txt)', ...
    k, fullfile(fileparts(cfg.Paths.Model), 'cv', char(cfg.Arm) + "_fold<k>.mat"));
cnngvf.writeSummary(fullfile(cvRoot, 'run_summary.txt'), sc, perImage, summary, []);
end
