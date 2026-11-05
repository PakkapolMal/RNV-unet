function out = run_tau_grid(cacheDir, outDir)
%RUN_TAU_GRID Post.tau scored off a CV P cache and cross-fitted per fold.
%
%  out = run_tau_grid                        % the reference cache
%  out = run_tau_grid(cacheDir, outDir)
%
%  Every value goes through run_cache, so every P is out-of-sample. tau is
%  chosen per fold on the other four folds and scored on the held-out one; the
%  fixed-tau curve is tuned on the images it scores and is not a result.
%  Pre-registration: docs/briefs/tau-crossfit.md.

cfg0 = cnngvf.defaults();
if nargin < 1 || strlength(cacheDir) == 0
    cacheDir = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
end
if nargin < 2
    outDir = fullfile(cfg0.Paths.Output, 'tau_grid');
end
cfg0.Paths.Output = outDir;   % run_cache writes here, not over the real one

taus   = [0.50 0.40 0.45 0.55 0.60 0.65 0.70];   % default first: it wins ties
tieTol = 0.001;

G = numel(taus);
rows = cell(G, 1);
for g = 1:G
    cfg = cfg0;
    cfg.Post.tau = taus(g);
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

curve = table(taus(:), mean(iou)', median(iou)', mean(metric("F1"))', mean(prec)', mean(rec)', ...
    'VariableNames', {'Tau', 'Iou', 'Median', 'F1', 'Precision', 'Recall'});
for ds = unique(dset)'
    curve.("Iou_" + ds) = mean(iou(dset == ds, :), 1)';
end
curve = sortrows(curve, 'Tau');

% Cross-fit: choose on the other folds, score on this one.
pickIdx = zeros(height(base), 1);
chosen = table();
for f = unique(fold)'
    te = fold == f;
    g = pick(mean(iou(~te, :), 1), tieTol);
    pickIdx(te) = g;
    chosen = [chosen; table(f, taus(g), mean(iou(~te, g)), mean(iou(te, g)), mean(iou(te, 1)), ...
        'VariableNames', {'Fold', 'Tau', 'TrainIou', 'TestIou', 'TestIouAt050'})]; %#ok<AGROW>
end
at = sub2ind(size(iou), (1:height(base))', pickIdx);
cf = struct('Iou', iou(at), 'Precision', prec(at), 'Recall', rec(at), ...
    'F1', cellfun(@(g, i) rows{g}.F1(i), num2cell(pickIdx), num2cell((1:height(base))')));

% Precision and recall by GT-area quintile, cross-fitted vs tau 0.50.
byQuint = table((1:5)', accumarray(quint, 1), ...
    accumarray(quint, prec(:, 1), [], @mean), accumarray(quint, cf.Precision, [], @mean), ...
    accumarray(quint, rec(:, 1), [], @mean), accumarray(quint, cf.Recall, [], @mean), ...
    accumarray(quint, iou(:, 1), [], @mean), accumarray(quint, cf.Iou, [], @mean), ...
    'VariableNames', {'Q', 'N', 'Prec050', 'PrecCf', 'Rec050', 'RecCf', 'Iou050', 'IouCf'});
q45 = quint >= 4;
recallGuard = mean(cf.Recall(q45)) - mean(rec(q45, 1));

% Cross-device: choose on one dataset, score the other.
crossDevice = table();
for ds = unique(dset)'
    te = dset == ds;
    g = pick(mean(iou(~te, :), 1), tieTol);
    crossDevice = [crossDevice; table(ds, taus(g), mean(iou(te, g)), mean(iou(te, 1)), ...
        'VariableNames', {'Test', 'Tau', 'TestIou', 'TestIouAt050'})]; %#ok<AGROW>
end

d = cf.Iou - iou(:, 1);
out = struct('curve', curve, 'chosen', chosen, 'byQuint', byQuint, 'crossDevice', crossDevice, ...
    'cfIou', mean(cf.Iou), 'cfPrecision', mean(cf.Precision), 'cfRecall', mean(cf.Recall), ...
    'cfF1', mean(cf.F1), 'baseIou', mean(iou(:, 1)), 'dMean', mean(d), ...
    'dUp', nnz(d > 1e-12), 'dDown', nnz(d < -1e-12), 'dCi', bootCi(d), 'recallGuard', recallGuard);

if ~exist(outDir, 'dir'), mkdir(outDir); end
writetable(curve, fullfile(outDir, 'curve.csv'));
writetable(chosen, fullfile(outDir, 'chosen.csv'));
writetable(byQuint, fullfile(outDir, 'by_quintile.csv'));
writetable(crossDevice, fullfile(outDir, 'cross_device.csv'));
writetable(table(base.Image, base.Dataset, fold, quint, taus(pickIdx)', iou(:, 1), cf.Iou, d, ...
    'VariableNames', {'Image', 'Dataset', 'Fold', 'Quintile', 'Tau', 'Iou050', 'IouCf', 'dIou'}), ...
    fullfile(outDir, 'crossfit_per_image.csv'));

fprintf('\n=== tau grid | %s ===\n', cacheDir);
disp(curve); disp(chosen); disp(byQuint); disp(crossDevice);
fprintf(['cross-fitted: IoU %.4f | F1 %.4f | precision %.4f | recall %.4f\n' ...
    'vs tau 0.50 %.4f: dIoU %+.4f, %d up / %d down, 95%% CI [%+.4f, %+.4f]\n' ...
    'recall guard (Q4-Q5 recall change): %+.4f\n'], ...
    out.cfIou, out.cfF1, out.cfPrecision, out.cfRecall, out.baseIou, out.dMean, ...
    out.dUp, out.dDown, out.dCi, recallGuard);
for ds = unique(dset)'
    fprintf('  %s: cross-fitted %.4f, tau 0.50 %.4f\n', ds, ...
        mean(cf.Iou(dset == ds)), mean(iou(dset == ds, 1)));
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
