function out = run_relax_grid(outDir)
%RUN_RELAX_GRID relaxWindow x relaxFloor, scored off the CV cache and cross-fitted.
%
%  out = run_relax_grid                  % writes to <Output>/relax_grid
%  out = run_relax_grid(outDir)
%
%  Every cell goes through run_cache, so every P is out-of-sample. The cell is
%  chosen per fold on the other four folds and scored on the held-out one;
%  the full surface is tuned on the images it scores and is not a result.
%  Pre-registration: docs/results/2026-09-30-relax-grid.md.

cfg0 = cnngvf.defaults();
if nargin < 1, outDir = fullfile(cfg0.Paths.Output, 'relax_grid'); end
cacheDir = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
cfg0.Paths.Output = outDir;   % run_cache writes its per-cell csv here, not over the real one

% Default first: it wins ties (within tieTol).
[W, F] = ndgrid([128 64 256 512], [0.20 0.05 0.10 0.15 0.25 0.30]);
grid = table(W(:), F(:), 'VariableNames', {'Window', 'Floor'});
grid = [grid(1, :); sortrows(grid(2:end, :))];
tieTol = 0.001;

G = height(grid);
rows = cell(G, 1);
for g = 1:G
    cfg = cfg0;
    cfg.Post.relaxWindow = grid.Window(g);
    cfg.Post.relaxFloor  = grid.Floor(g);
    rows{g} = run_cache(cfg, cacheDir);
end
iou = cell2mat(cellfun(@(t) t.Iou, rows', 'UniformOutput', false));
[~, info] = cnngvf.cvFolds(cfg0);
[~, loc] = ismember(rows{1}.Image, info.Image);
fold = info.Fold(loc);
ref = readtable(cfg0.CV.reference, 'TextType', 'string');
[~, r] = ismember(rows{1}.Image, ref.Image);
assert(all(loc > 0) && isequal(fold, ref.Fold(r)), 'cvFolds disagrees with the reference folds');
dset = string(rows{1}.Dataset);

surface = grid;
for m = ["Iou" "F1" "Precision" "Recall"]
    surface.(m) = cellfun(@(t) mean(t.(m)), rows);
end
for ds = unique(dset)'
    surface.("Iou_" + ds) = mean(iou(dset == ds, :), 1)';
end

% Cross-fit: choose on the other folds, score on this one.
cf = rows{1};
chosen = table();
for f = unique(fold)'
    te = fold == f;
    g = pick(mean(iou(~te, :), 1), tieTol);
    cf(te, :) = rows{g}(te, :);
    chosen = [chosen; table(f, grid.Window(g), grid.Floor(g), mean(iou(~te, g)), ...
        mean(iou(te, g)), 'VariableNames', {'Fold', 'Window', 'Floor', 'TrainIou', 'TestIou'})]; %#ok<AGROW>
end

% Cross-device: choose on one dataset, score the other.
crossDevice = table();
for ds = unique(dset)'
    te = dset == ds;
    g = pick(mean(iou(~te, :), 1), tieTol);
    crossDevice = [crossDevice; table(ds, grid.Window(g), grid.Floor(g), mean(iou(te, g)), ...
        mean(iou(te, 1)), 'VariableNames', {'Test', 'Window', 'Floor', 'TestIou', 'DefaultIou'})]; %#ok<AGROW>
end

d = cf.Iou - rows{1}.Iou;
out = struct('surface', surface, 'chosen', chosen, 'crossDevice', crossDevice, ...
    'crossFit', cf, 'dMean', mean(d), 'dUp', nnz(d > 1e-12), 'dDown', nnz(d < -1e-12), ...
    'dCi', bootCi(d));

if ~exist(outDir, 'dir'), mkdir(outDir); end
writetable(surface, fullfile(outDir, 'surface.csv'));
writetable(chosen, fullfile(outDir, 'chosen.csv'));
writetable(crossDevice, fullfile(outDir, 'cross_device.csv'));
writetable(cf, fullfile(outDir, 'crossfit_per_image.csv'));

disp(surface); disp(chosen); disp(crossDevice);
fprintf(['cross-fitted: IoU %.4f | median %.4f | F1 %.4f | precision %.4f | recall %.4f\n' ...
    'vs default (128, 0.20) %.4f: dIoU %+.4f, %d up / %d down, 95%% CI [%+.4f, %+.4f]\n'], ...
    mean(cf.Iou), median(cf.Iou), mean(cf.F1), mean(cf.Precision), mean(cf.Recall), ...
    mean(rows{1}.Iou), out.dMean, out.dUp, out.dDown, out.dCi);
for ds = unique(dset)'
    fprintf('  %s: cross-fitted %.4f, default %.4f\n', ds, ...
        mean(cf.Iou(dset == ds)), mean(rows{1}.Iou(dset == ds)));
end
end


function g = pick(m, tieTol)
g = find(m >= max(m) - tieTol, 1, 'first');
end


function ci = bootCi(d)
% Percentile bootstrap on the mean, seeded as the 2026-09-17 record's.
old = rng(20260917);
restore = onCleanup(@() rng(old));
means = sort(mean(d(randi(numel(d), numel(d), 10000)), 1));
ci = means([250 9751]);
end
