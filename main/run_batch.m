function results = run_batch(varargin)
%RUN_BATCH Segment every RNV lesion in the localization CSV and save results.
%
%  run_batch                        % ray-casting pipeline
%  run_batch("adaptive")            % single-radius circle pipeline
%  run_batch(cfg)                   % fully custom config
%  results = run_batch(...)         % also return a per-image summary table
%
%  Replaces gvf_ray_v1.m and gvf_adaptive_v1.m, which were the same script
%  with a different initializer.
%
%  Output layout under cfg.Paths.Output:
%    combined/<stem>.png   union mask per image  <- Compare_DRAC22_IOU.m reads this
%    overlay/<stem>.png    image + seeds + final contours
%    evaluation.csv        per-image IoU/F1/precision/recall, area and runtime
%    run_summary.txt       aggregate scores + the config that produced them
%
%  Images are scored as they are produced, against cfg.Paths.GT; metrics are
%  NaN for images with no ground truth. Writing the union directly makes the
%  separate Combine_RNV_Masks.m pass unnecessary.
%
%  See also RNV.SEGMENTIMAGE, RNV.EVALUATE, RUN_TUNING.

if nargin == 0
    cfg = rnv.defaults();
elseif isstruct(varargin{1})
    cfg = varargin{1};
else
    cfg = rnv.defaults(varargin{1});
end

[names, seeds] = rnv.loadSeeds(cfg);
images = unique(names, 'stable');

combinedDir = fullfile(cfg.Paths.Output, 'combined');
overlayDir  = fullfile(cfg.Paths.Output, 'overlay');
for d = [string(combinedDir), string(overlayDir)]
    if ~exist(d, 'dir')
        mkdir(d);
    end
end

fprintf('run_batch | init=%s postproc=%s | %d images, %d seeds\n', ...
    cfg.Init, cfg.PostProc, numel(images), numel(names));
fprintf('Output: %s\n\n', cfg.Paths.Output);

n       = numel(images);
nSeeds  = zeros(n, 1);
areaPx  = zeros(n, 1);
gtPx    = nan(n, 1);
elapsed = zeros(n, 1);
iou     = nan(n, 1);
f1      = nan(n, 1);
prec    = nan(n, 1);
rec     = nan(n, 1);

for i = 1:n
    imgName  = images(i);
    imgSeeds = seeds(names == imgName, :);
    [~, stem] = fileparts(imgName);

    tImg = tic;
    [mask, contours, img] = rnv.segmentImage( ...
        fullfile(cfg.Paths.Img, char(imgName)), imgSeeds, cfg);
    elapsed(i) = toc(tImg);

    nSeeds(i) = size(imgSeeds, 1);
    areaPx(i) = nnz(mask);

    imwrite(mask, fullfile(combinedDir, char(stem + ".png")));
    rnv.saveOverlay(fullfile(overlayDir, char(stem + ".png")), img, imgSeeds, contours);

    % Score against ground truth where it exists. The mask is already in
    % hand, so this costs a file read rather than a second segmentation.
    gt = rnv.loadMask(fullfile(cfg.Paths.GT, char(imgName)));
    if isempty(gt)
        score = '   n/a';
    else
        m       = rnv.metrics(mask, gt);
        gtPx(i) = nnz(gt);
        iou(i)  = m.iou;
        f1(i)   = m.f1;
        prec(i) = m.precision;
        rec(i)  = m.recall;
        score   = sprintf('%6.4f', m.iou);
    end

    fprintf('  [%3d/%3d] %-10s %d seed(s), %7d px, %5.1f s, IoU %s\n', ...
        i, n, imgName, nSeeds(i), areaPx(i), elapsed(i), score);
end

results = table(images, nSeeds, areaPx, gtPx, iou, f1, prec, rec, elapsed, ...
    'VariableNames', {'Image', 'Seeds', 'AreaPx', 'GtPx', ...
                      'Iou', 'F1', 'Precision', 'Recall', 'Seconds'});

writetable(results, fullfile(cfg.Paths.Output, 'evaluation.csv'));
writeSummary(fullfile(cfg.Paths.Output, 'run_summary.txt'), cfg, results);

scored = ~isnan(iou);
fprintf('\nDone in %.1f s. %d masks written to %s\n', sum(elapsed), n, combinedDir);
if any(scored)
    fprintf('Scored %d/%d images against GT: mean IoU = %.4f, mean F1 = %.4f\n', ...
        nnz(scored), n, mean(iou(scored)), mean(f1(scored)));
else
    fprintf('No ground truth found in %s; evaluation columns are NaN.\n', cfg.Paths.GT);
end
fprintf('Records: %s\n', fullfile(cfg.Paths.Output, 'evaluation.csv'));
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function writeSummary(txtPath, cfg, results)
%WRITESUMMARY Aggregate scores plus the config that produced them, so a run
%  can be reproduced from its own record.

fid = fopen(txtPath, 'w');
if fid < 0
    warning('rnv:runBatch:noWrite', 'Could not write %s', txtPath);
    return;
end
cleanup = onCleanup(@() fclose(fid));

scored = ~isnan(results.Iou);

fprintf(fid, 'RNV segmentation run\n');
fprintf(fid, '====================\n');
fprintf(fid, 'Timestamp : %s\n', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
fprintf(fid, 'Init      : %s\n', cfg.Init);
fprintf(fid, 'PostProc  : %s\n', cfg.PostProc);
fprintf(fid, 'Images    : %d (%d seeds)\n', height(results), sum(results.Seeds));
fprintf(fid, 'Runtime   : %.1f s\n\n', sum(results.Seconds));

if any(scored)
    fprintf(fid, 'Scored against GT: %d/%d images\n', nnz(scored), height(results));
    fprintf(fid, '  mean IoU       : %.6f\n', mean(results.Iou(scored)));
    fprintf(fid, '  mean F1        : %.6f\n', mean(results.F1(scored)));
    fprintf(fid, '  mean Precision : %.6f\n', mean(results.Precision(scored)));
    fprintf(fid, '  mean Recall    : %.6f\n', mean(results.Recall(scored)));
    fprintf(fid, '  median IoU     : %.6f\n', median(results.Iou(scored)));
else
    fprintf(fid, 'No ground truth found in %s.\n', cfg.Paths.GT);
end

fprintf(fid, '\nSnake2D parameters\n------------------\n');
snakeFields = fieldnames(cfg.Snake);
for k = 1:numel(snakeFields)
    v = cfg.Snake.(snakeFields{k});
    if islogical(v)
        v = double(v);
    end
    fprintf(fid, '  %-12s %g\n', snakeFields{k}, v);
end

fprintf(fid, '\nInitialization parameters\n-------------------------\n');
radiusFields = fieldnames(cfg.Radius);
for k = 1:numel(radiusFields)
    fprintf(fid, '  %-18s %g\n', radiusFields{k}, cfg.Radius.(radiusFields{k}));
end

fprintf(fid, '\n  %-18s %g\n', 'GvfMargin', cfg.GvfMargin);
fprintf(fid, '  %-18s %g\n', 'MinArea', cfg.Post.MinArea);
fprintf(fid, '  %-18s %g\n', 'EdgeSigma', cfg.EdgeSigma);
fprintf(fid, '  %-18s %s\n', 'Denoise', mat2str(cfg.Denoise));

fprintf(fid, '\nPer-image\n---------\n');
fprintf(fid, '  %-12s %6s %6s %6s %6s %9s\n', 'Image', 'IoU', 'F1', 'Prec', 'Rec', 'AreaPx');
for i = 1:height(results)
    fprintf(fid, '  %-12s %6.4f %6.4f %6.4f %6.4f %9d\n', ...
        results.Image(i), results.Iou(i), results.F1(i), ...
        results.Precision(i), results.Recall(i), results.AreaPx(i));
end
end
