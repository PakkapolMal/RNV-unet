function out = crossFit(grid, scoreFn, folds, dset, tieTol)
%CNNGVF.CROSSFIT Choose a parameter on the other folds, score it on this one.
%
%  out = cnngvf.crossFit(grid, scoreFn, folds, dset)
%  out = cnngvf.crossFit(grid, scoreFn, folds, dset, tieTol)
%
%  grid    : table, one row per parameter setting, ORDERED MOST CONSERVATIVE
%            FIRST - the row closest to the feature being off
%  scoreFn : @(gridRow) -> per-image table (Image, Dataset, Fold, GtPx, PredPx,
%            TP, FP, FN, Iou, F1, Precision, Recall), same image order every call
%  folds   : n x 1 fold index per image, from the cache metadata (cnngvf.cvFolds)
%  dset    : n x 1 dataset name per image, for the cross-device check
%  tieTol  : rows within tieTol mean IoU of the best count as tied, and the first
%            (most conservative) of them wins. Default 0.001 (../docs/REFINE.md).
%            0 is a plain maximum - what run_confirm_relax did.
%
%  out.perImage    : the cross-fitted per-image table, plus ChosenRow
%  out.chosen      : table Fold, GridRow, TrainIou, TestIou, then the chosen grid row
%  out.surface     : every grid row's full-domain metrics - a SURFACE, never a result
%  out.crossDevice : table Train, Test, GridRow, TestIou - choose on one device, score the other
%  out.rows        : n-grid cell of the per-image tables, so callers reuse them
%
%  Every grid row is scored once, on all images; the fold loop only selects.
%  Generalizes the logic in run_confirm_relax.
%
%  See also RUN_REFINE, RUN_CONFIRM_RELAX, CNNGVF.REFINESCORE.

if nargin < 5 || isempty(tieTol)
    tieTol = 0.001;
end
folds = folds(:);
dset  = string(dset(:));
G = height(grid);

rows = cell(G, 1);
iou  = zeros(numel(folds), G);
for g = 1:G
    rows{g}   = scoreFn(grid(g, :));
    iou(:, g) = rows{g}.Iou;
end

perImage = rows{1};
chosenRow = zeros(numel(folds), 1);
fs = unique(folds)';
chosen = table();
for f = fs
    te = folds == f;
    g = pick(mean(iou(~te, :), 1), tieTol);
    perImage(te, :) = rows{g}(te, :);
    chosenRow(te) = g;
    chosen = [chosen; [table(f, g, mean(iou(~te, g)), mean(iou(te, g)), ...
        'VariableNames', {'Fold', 'GridRow', 'TrainIou', 'TestIou'}), grid(g, :)]]; %#ok<AGROW>
end
perImage.ChosenRow = chosenRow;

surface = table();
for g = 1:G
    surface = [surface; [grid(g, :), cnngvf.refineScore(rows{g})]]; %#ok<AGROW>
end

crossDevice = table();
for ds = unique(dset)'
    te = dset == ds;
    g = pick(mean(iou(~te, :), 1), tieTol);
    crossDevice = [crossDevice; table(strjoin(unique(dset(~te))', '+'), ds, g, mean(iou(te, g)), ...
        'VariableNames', {'Train', 'Test', 'GridRow', 'TestIou'})]; %#ok<AGROW>
end

out = struct('perImage', perImage, 'chosen', chosen, 'surface', surface, ...
    'crossDevice', crossDevice);
out.rows = rows;
end


function g = pick(m, tieTol)
g = find(m >= max(m) - tieTol, 1, 'first');
end
