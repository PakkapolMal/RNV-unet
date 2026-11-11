function out = run_postfit_grid(cacheDir, outDir)
%RUN_POSTFIT_GRID Post.tau x Post.relaxFloor scored off a CV P cache, cross-fitted per fold.
%
%  out = run_postfit_grid                        % the reference cache
%  out = run_postfit_grid(cacheDir, outDir)
%
%  Every cell goes through run_cache, so every P is out-of-sample. A cell is
%  chosen per fold on the other four folds and scored on the held-out one; the
%  fixed-cell surface is tuned on the images it scores and is not a result.
%  Pre-registration: docs/briefs/tau-floor-refit.md.

cfg0 = cnngvf.defaults();
if nargin < 1 || strlength(cacheDir) == 0
    cacheDir = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
end
if nargin < 2
    outDir = fullfile(cfg0.Paths.Output, 'postfit_grid');
end
cfg0.Paths.Output = outDir;   % run_cache writes here, not over the real one

% Nearest the default first, so the default wins ties and then the nearest cell.
taus   = [0.50 0.45 0.55 0.40 0.60 0.65 0.70];
floors = [0.20 0.15 0.25 0.10 0.30];
[F, T] = meshgrid(floors, taus);
cells  = [reshape(T', [], 1), reshape(F', [], 1)];   % tau-major: (0.50, 0.20) first
assert(isequal(cells(1, :), [cfg0.Post.tau cfg0.Post.relaxFloor]), 'the default must be cell 1');
tieTol = 0.001;

G = size(cells, 1);
rows = cell(G, 1);
for g = 1:G
    cfg = cfg0;
    cfg.Post.tau = cells(g, 1);
    cfg.Post.relaxFloor = cells(g, 2);
    fprintf('cell %d/%d: tau %.2f, relaxFloor %.2f\n', g, G, cells(g, :));
    rows{g} = run_cache(cfg, cacheDir);
end
base = rows{1};
for g = 2:G
    assert(isequal(rows{g}.Image, base.Image), 'image order differs between runs');
end
metric = @(m) cell2mat(cellfun(@(t) t.(m), rows', 'UniformOutput', false));
iou = metric("Iou");
prec = metric("Precision");
rec = metric("Recall");
f1 = metric("F1");

[~, info] = cnngvf.cvFolds(cfg0);
[~, loc] = ismember(base.Image, info.Image);
assert(all(loc > 0), 'an image is in no fold');
fold = info.Fold(loc);
dset = string(base.Dataset);

% GT-area quintiles: Q1 smallest, 14/14/14/14/13 on 69 images.
[~, ord] = sort(base.GtPx);
rk = zeros(height(base), 1);
rk(ord) = 1:height(base);
quint = floor(5 * (rk - 1) / height(base)) + 1;

surface = table(cells(:, 1), cells(:, 2), mean(iou)', mean(f1)', mean(prec)', mean(rec)', ...
    'VariableNames', {'Tau', 'Floor', 'Iou', 'F1', 'Precision', 'Recall'});
for ds = unique(dset)'
    surface.("Iou_" + ds) = mean(iou(dset == ds, :), 1)';
end
surface = sortrows(surface, {'Tau', 'Floor'});

% Cross-fit: choose on the other folds, score on this one.
pickIdx = zeros(height(base), 1);
chosen = table();
for f = unique(fold)'
    te = fold == f;
    g = pick(mean(iou(~te, :), 1), tieTol);
    pickIdx(te) = g;
    chosen = [chosen; table(f, cells(g, 1), cells(g, 2), mean(iou(~te, g)), mean(iou(te, g)), ...
        mean(iou(te, 1)), 'VariableNames', {'Fold', 'Tau', 'Floor', 'TrainIou', 'TestIou', 'TestIouDefault'})]; %#ok<AGROW>
end
at = sub2ind(size(iou), (1:height(base))', pickIdx);
cf = struct('Iou', iou(at), 'Precision', prec(at), 'Recall', rec(at), 'F1', f1(at));

byQuint = table((1:5)', accumarray(quint, 1), ...
    accumarray(quint, prec(:, 1), [], @mean), accumarray(quint, cf.Precision, [], @mean), ...
    accumarray(quint, rec(:, 1), [], @mean), accumarray(quint, cf.Recall, [], @mean), ...
    accumarray(quint, iou(:, 1), [], @mean), accumarray(quint, cf.Iou, [], @mean), ...
    'VariableNames', {'Q', 'N', 'PrecDefault', 'PrecCf', 'RecDefault', 'RecCf', 'IouDefault', 'IouCf'});
q13 = quint <= 3;
q45 = quint >= 4;
precGuard = mean(cf.Precision(q13)) - mean(prec(q13, 1));
recGuard  = mean(cf.Recall(q45)) - mean(rec(q45, 1));

% Cross-device: choose on one dataset, score the other.
crossDevice = table();
for ds = unique(dset)'
    te = dset == ds;
    g = pick(mean(iou(~te, :), 1), tieTol);
    crossDevice = [crossDevice; table(ds, cells(g, 1), cells(g, 2), mean(iou(te, g)), mean(iou(te, 1)), ...
        'VariableNames', {'Test', 'Tau', 'Floor', 'TestIou', 'TestIouDefault'})]; %#ok<AGROW>
end

% The cell chosen on all 69 images: what would ship (Reporting rule 6).
gAll = pick(mean(iou, 1), tieTol);

d = cf.Iou - iou(:, 1);
out = struct('surface', surface, 'chosen', chosen, 'byQuint', byQuint, 'crossDevice', crossDevice, ...
    'cfIou', mean(cf.Iou), 'cfPrecision', mean(cf.Precision), 'cfRecall', mean(cf.Recall), ...
    'cfF1', mean(cf.F1), 'baseIou', mean(iou(:, 1)), 'dMean', mean(d), ...
    'dUp', nnz(d > 1e-12), 'dDown', nnz(d < -1e-12), 'dCi', bootCi(d), ...
    'precGuard', precGuard, 'recGuard', recGuard, 'allImagesCell', cells(gAll, :));

if ~exist(outDir, 'dir'), mkdir(outDir); end
writetable(surface, fullfile(outDir, 'surface.csv'));
writetable(chosen, fullfile(outDir, 'chosen.csv'));
writetable(byQuint, fullfile(outDir, 'by_quintile.csv'));
writetable(crossDevice, fullfile(outDir, 'cross_device.csv'));
writetable(table(base.Image, base.Dataset, fold, quint, cells(pickIdx, 1), cells(pickIdx, 2), ...
    iou(:, 1), cf.Iou, d, cf.Precision, cf.Recall, ...
    'VariableNames', {'Image', 'Dataset', 'Fold', 'Quintile', 'Tau', 'Floor', 'IouDefault', 'IouCf', ...
    'dIou', 'PrecisionCf', 'RecallCf'}), fullfile(outDir, 'crossfit_per_image.csv'));

fprintf('\n=== tau x relaxFloor grid | %s ===\n', cacheDir);
disp(surface); disp(chosen); disp(byQuint); disp(crossDevice);
fprintf(['cross-fitted: IoU %.4f | F1 %.4f | precision %.4f | recall %.4f\n' ...
    'vs default %.4f: dIoU %+.4f, %d up / %d down, 95%% CI [%+.4f, %+.4f]\n' ...
    'guards: Q1-Q3 precision %+.4f, Q4-Q5 recall %+.4f\n' ...
    'chosen on all 69 images: tau %.2f, relaxFloor %.2f\n'], ...
    out.cfIou, out.cfF1, out.cfPrecision, out.cfRecall, out.baseIou, out.dMean, ...
    out.dUp, out.dDown, out.dCi, precGuard, recGuard, out.allImagesCell);
for ds = unique(dset)'
    fprintf('  %s: cross-fitted %.4f, default %.4f\n', ds, mean(cf.Iou(dset == ds)), mean(iou(dset == ds, 1)));
end
end


function g = pick(m, tieTol)
g = find(m >= max(m) - tieTol, 1, 'first');
end


function ci = bootCi(d)
% Percentile bootstrap on the mean, seeded as the earlier cross-fit records.
old = rng(20260917);
restore = onCleanup(@() rng(old));
means = sort(mean(d(randi(numel(d), numel(d), 10000)), 1));
ci = means([250 9751]);
end
