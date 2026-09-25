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
%  Metrics come from cnngvf.metrics (iou / f1 / precision / recall).
%
%  See also CNNGVF.METRICS, CNNGVF.SEGGTFILE, RUN_BATCH.

M = cnngvf.manifest(cfg);

switch cfg.EvalMode
    case "segment"
        requireSegmentable(cfg, M);
        [perUnit, missing] = scoreSegments(results, cfg, M);
        unclaimed = unclaimedGtPixels(results, cfg, M);

    case "image"
        [perUnit, missing] = scoreImages(results, cfg, M);
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

function requireSegmentable(cfg, M)
%REQUIRESEGMENTABLE EvalMode "segment" needs per-seed masks, which only DRAC22
%  has - and only for the clustered coordinates they are named after. Fail with
%  the reason rather than scoring every TUH seed against a missing file and
%  reporting it as a warning at the end.
noSeg = unique(M.Dataset(strlength(M.SegGtDir) == 0));
if ~isempty(noSeg)
    error('cnngvf:evaluate:notSegmentable', ...
        ['EvalMode "segment" needs per-seed ground truth, which dataset(s) %s ' ...
         'do not have.\nUse EvalMode "image", or restrict the domain with ' ...
         'cfg.ActiveDatasets = "DRAC22".'], strjoin(noSeg', ', '));
end
if cfg.Points ~= "clustered"
    error('cnngvf:evaluate:notSegmentable', ...
        ['EvalMode "segment" is only valid for the clustered centres: the ' ...
         'SegmentFilledGT files are keyed to those coordinates. This run uses ' ...
         'the "%s" points, several of which land on one lesion, so per-segment ' ...
         'scoring would count that lesion once per detection.'], cfg.Points);
end
end


function [T, missing] = scoreSegments(results, cfg, M)
rows = {};
missing = strings(0, 1);

for i = 1:numel(results)
    r = results(i);
    ds = datasetOf(M, r.image);
    for s = 1:size(r.seeds, 1)
        gtPath = cnngvf.segGtFile(cfg, r.image, r.seeds(s, :));
        gt = cnngvf.loadMask(gtPath);
        if isempty(gt)
            missing(end + 1, 1) = string(gtPath); %#ok<AGROW>
            continue;
        end

        pred = r.perSeed{s};
        if isempty(pred)
            pred = false(size(gt));
        end
        m = cnngvf.metrics(pred, gt);

        rows(end + 1, :) = {r.image, ds, r.seeds(s, 1), r.seeds(s, 2), ...
            m.iou, m.f1, m.precision, m.recall, nnz(pred), nnz(gt)}; %#ok<AGROW>
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'Image', 'Dataset', 'X', 'Y', 'Iou', 'F1', 'Precision', 'Recall', 'PredPx', 'GtPx'});
T.Image   = string(T.Image);
T.Dataset = string(T.Dataset);
end


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


function ds = datasetOf(M, imgName)
k = find(M.Name == imgName, 1);
if isempty(k)
    ds = "unknown";
else
    ds = M.Dataset(k);
end
end


function px = unclaimedGtPixels(results, cfg, M)
% Ground-truth pixels in FilledGT that no seed's segment file covers, i.e.
% lesions the localization stage never pointed at. Reported so a good
% per-segment score is not mistaken for full coverage of the image.
px = 0;
for i = 1:numel(results)
    k = find(M.Name == results(i).image, 1);
    if isempty(k)
        continue;
    end
    filled = cnngvf.loadMask(M.GtPath(k));
    if isempty(filled)
        continue;
    end
    claimed = false(size(filled));
    for s = 1:size(results(i).seeds, 1)
        g = cnngvf.loadMask(cnngvf.segGtFile(cfg, results(i).image, results(i).seeds(s, :)));
        if ~isempty(g)
            claimed = claimed | g;
        end
    end
    px = px + nnz(filled & ~claimed);
end
end
