function [perUnit, summary] = evaluate(results, cfg)
%CNNGVF.EVALUATE Score predictions against ground truth.
%
%  [perUnit, summary] = cnngvf.evaluate(results, cfg)
%
%  results : struct array from run_batch, one entry per image, with fields
%              .image   "458.png"
%              .seeds   S x 2 [cx cy]
%              .perSeed S x 1 cell of per-seed predicted masks
%              .mask    union of those masks
%
%  One row per image: the union of every seed's mask against the whole
%  ground-truth mask. Several detections land on one lesion, so scoring them
%  one at a time would count that lesion once per detection; whole-image
%  scoring also charges the prediction for lesions it missed entirely.
%
%  perUnit : table of per-image metrics
%  summary : struct of aggregates
%
%  Metrics come from cnngvf.metrics (iou / f1 / precision / recall).
%
%  See also CNNGVF.METRICS, RUN_BATCH.

M = cnngvf.manifest(cfg);

[perUnit, missing] = scoreImages(results, cfg, M);

summary = struct( ...
    'nUnits',         height(perUnit), ...
    'meanIou',        mean(perUnit.Iou),   'medianIou', median(perUnit.Iou), ...
    'meanF1',         mean(perUnit.F1),    'medianF1',  median(perUnit.F1), ...
    'meanPrecision',  mean(perUnit.Precision), ...
    'meanRecall',     mean(perUnit.Recall), ...
    'missingGtFiles', numel(missing));

% Per-dataset breakout. The protocol pools datasets deliberately, but a pooled
% mean over two devices is not interpretable on its own - and this costs one
% groupsummary on a table that already exists, so there is no reason to make
% anyone re-run a job to get it.
summary.perDataset = groupsummary(perUnit, 'Dataset', 'mean', ...
    {'Iou', 'F1', 'Precision', 'Recall'});

if ~isempty(missing)
    warning('cnngvf:evaluate:missingGt', ...
        '%d ground-truth file(s) missing and not scored. First: %s', ...
        numel(missing), missing(1));
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [T, missing] = scoreImages(results, ~, M)
rows = {};
missing = strings(0, 1);

for i = 1:numel(results)
    r = results(i);
    k = find(M.Name == r.image, 1);
    if isempty(k)
        missing(end + 1, 1) = r.image; %#ok<AGROW>
        continue;
    end
    gt = cnngvf.loadMask(M.GtPath(k));
    if isempty(gt)
        missing(end + 1, 1) = M.GtPath(k); %#ok<AGROW>
        continue;
    end

    pred = r.mask;
    if isempty(pred)
        pred = false(size(gt));
    end
    m = cnngvf.metrics(pred, gt);

    rows(end + 1, :) = {r.image, M.Dataset(k), size(r.seeds, 1), ...
        m.iou, m.f1, m.precision, m.recall, nnz(pred), nnz(gt)}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', ...
    {'Image', 'Dataset', 'Seeds', 'Iou', 'F1', 'Precision', 'Recall', 'PredPx', 'GtPx'});
T.Image   = string(T.Image);
T.Dataset = string(T.Dataset);
end

