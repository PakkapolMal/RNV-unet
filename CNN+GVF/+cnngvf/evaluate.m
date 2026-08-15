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
%  cfg.EvalMode selects the unit of evaluation:
%
%   "segment"  One row per seed. The seed's own mask is compared with
%              SegmentFilledGT/<stem>-<x>-<y>.png, the ground-truth component
%              that seed falls in. Only valid for the clustered centres, since
%              the ground-truth files are keyed by their coordinates.
%              Ground-truth components no seed claims have no file and are not
%              scored; they are reported as unclaimedGtPx rather than folded
%              into recall, because they reflect the localization stage.
%
%   "image"    One row per image. The union of every seed's mask is compared
%              with the whole FilledGT/<stem>.png. This is the correct unit for
%              the unclustered points: they have no per-point ground truth, and
%              several of them land on one lesion, so scoring them one at a
%              time would count the same lesion repeatedly and weight big
%              lesions by how many detections they attracted. Whole-image
%              scoring also charges the prediction for lesions it missed
%              entirely, which per-segment scoring cannot see.
%
%  perUnit : table of per-row metrics; the columns differ by mode (a segment
%            row carries the seed's X,Y, an image row does not)
%  summary : struct of aggregates, same field names in both modes
%
%  Metrics come from rnv.metrics (iou / f1 / precision / recall), reused from
%  RNV-Claude/main so both pipelines are scored by identical code.
%
%  See also RNV.METRICS, CNNGVF.SEGGTFILE, RUN_BATCH.

switch cfg.EvalMode
    case "segment"
        [perUnit, missing] = scoreSegments(results, cfg);
        unclaimed = unclaimedGtPixels(results, cfg);

    case "image"
        [perUnit, missing] = scoreImages(results, cfg);
        unclaimed = 0;   % the whole mask is the target; nothing is excluded

    otherwise
        error('cnngvf:evaluate:badMode', ...
            'Unknown EvalMode "%s". Use "segment" or "image".', cfg.EvalMode);
end

summary = struct( ...
    'mode',           cfg.EvalMode, ...
    'nUnits',         height(perUnit), ...
    'meanIou',        mean(perUnit.Iou),   'medianIou', median(perUnit.Iou), ...
    'meanF1',         mean(perUnit.F1),    'medianF1',  median(perUnit.F1), ...
    'meanPrecision',  mean(perUnit.Precision), ...
    'meanRecall',     mean(perUnit.Recall), ...
    'unclaimedGtPx',  unclaimed, ...
    'missingGtFiles', numel(missing));

if ~isempty(missing)
    warning('cnngvf:evaluate:missingGt', ...
        '%d ground-truth file(s) missing and not scored. First: %s', ...
        numel(missing), missing(1));
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [T, missing] = scoreSegments(results, cfg)
rows = {};
missing = strings(0, 1);

for i = 1:numel(results)
    r = results(i);
    for s = 1:size(r.seeds, 1)
        gtPath = cnngvf.segGtFile(cfg, r.image, r.seeds(s, :));
        gt = rnv.loadMask(gtPath);
        if isempty(gt)
            missing(end + 1, 1) = string(gtPath); %#ok<AGROW>
            continue;
        end

        pred = r.perSeed{s};
        if isempty(pred)
            pred = false(size(gt));
        end
        m = rnv.metrics(pred, gt);

        rows(end + 1, :) = {r.image, r.seeds(s, 1), r.seeds(s, 2), ...
            m.iou, m.f1, m.precision, m.recall, nnz(pred), nnz(gt)}; %#ok<AGROW>
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'Image', 'X', 'Y', 'Iou', 'F1', 'Precision', 'Recall', 'PredPx', 'GtPx'});
T.Image = string(T.Image);
end


function [T, missing] = scoreImages(results, cfg)
rows = {};
missing = strings(0, 1);

for i = 1:numel(results)
    r = results(i);
    gtPath = fullfile(cfg.Paths.GT, char(r.image));
    gt = rnv.loadMask(gtPath);
    if isempty(gt)
        missing(end + 1, 1) = string(gtPath); %#ok<AGROW>
        continue;
    end

    pred = r.mask;
    if isempty(pred)
        pred = false(size(gt));
    end
    m = rnv.metrics(pred, gt);

    rows(end + 1, :) = {r.image, size(r.seeds, 1), ...
        m.iou, m.f1, m.precision, m.recall, nnz(pred), nnz(gt)}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', ...
    {'Image', 'Seeds', 'Iou', 'F1', 'Precision', 'Recall', 'PredPx', 'GtPx'});
T.Image = string(T.Image);
end


function px = unclaimedGtPixels(results, cfg)
% Ground-truth pixels in FilledGT that no seed's segment file covers, i.e.
% lesions the localization stage never pointed at. Reported so a good
% per-segment score is not mistaken for full coverage of the image.
px = 0;
for i = 1:numel(results)
    filled = rnv.loadMask(fullfile(cfg.Paths.GT, char(results(i).image)));
    if isempty(filled)
        continue;
    end
    claimed = false(size(filled));
    for s = 1:size(results(i).seeds, 1)
        g = rnv.loadMask(cnngvf.segGtFile(cfg, results(i).image, results(i).seeds(s, :)));
        if ~isempty(g)
            claimed = claimed | g;
        end
    end
    px = px + nnz(filled & ~claimed);
end
end
