function out = run_fusion_grid(outDir)
%RUN_FUSION_GRID Cross-fit what weak seeds get: relax, seedless first, or max then relax.
%
%  out = run_fusion_grid               % outDir: <Output>/fusion_grid
%  out = run_fusion_grid(outDir)
%
%  Every cell goes through run_cache on the adopted seeded and seedless CV
%  caches, so every P is out-of-sample and both pass the leakage guard. A cell
%  is chosen per fold on the other four folds and scored on the held-out one;
%  the fixed-cell surface is tuned on the images it scores and is not a result.
%  Pre-registration: docs/briefs/fusion.md.

cfg0 = cnngvf.defaults();
if nargin < 1
    outDir = fullfile(cfg0.Paths.Output, 'fusion_grid');
end
seededCache = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
cfg0.Paths.Output = outDir;   % run_cache writes here, not over the real one

% The default first, then the conservative order ties fall to (docs/briefs/fusion.md).
cells = table(["relax"; "seedlessFirst"; "maxRelax"; "seedlessFirst"; "maxRelax"], ...
    [NaN; 0.9; 0.9; 0.5; 0.5], 'VariableNames', {'Variant', 'TauS'});
assert(cells.Variant(1) == cfg0.Post.weakSeed, 'the default must be cell 1');
tieTol = 0.001;
G = height(cells);

rows = cell(G, 1);
for g = 1:G
    cfg = cellCfg(cfg0, cells(g, :));
    fprintf('cell %d/%d: %s, tau_s %.1f\n', g, G, cells.Variant(g), cells.TauS(g));
    rows{g} = run_cache(cfg, seededCache);
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

% What each weak seed got, per cell: seedless evidence, relax, or the disk.
counts = weakSeedCounts(cfg0, cells, base.Image, seededCache);

surface = [cells, table(mean(iou)', mean(f1)', mean(prec)', mean(rec)', ...
    'VariableNames', {'Iou', 'F1', 'Precision', 'Recall'}), counts];
for ds = unique(dset)'
    surface.("Iou_" + ds) = mean(iou(dset == ds, :), 1)';
end

% Cross-fit: choose on the other folds, score on this one.
pickIdx = zeros(height(base), 1);
chosen = table();
for f = unique(fold)'
    te = fold == f;
    g = pick(mean(iou(~te, :), 1), tieTol);
    pickIdx(te) = g;
    chosen = [chosen; table(f, cells.Variant(g), cells.TauS(g), mean(iou(~te, g)), mean(iou(te, g)), ...
        mean(iou(te, 1)), 'VariableNames', {'Fold', 'Variant', 'TauS', 'TrainIou', 'TestIou', 'TestIouDefault'})]; %#ok<AGROW>
end
at = sub2ind(size(iou), (1:height(base))', pickIdx);
cf = struct('Iou', iou(at), 'Precision', prec(at), 'Recall', rec(at), 'F1', f1(at));

byQuint = table((1:5)', accumarray(quint, 1), ...
    accumarray(quint, prec(:, 1), [], @mean), accumarray(quint, cf.Precision, [], @mean), ...
    accumarray(quint, rec(:, 1), [], @mean), accumarray(quint, cf.Recall, [], @mean), ...
    accumarray(quint, iou(:, 1), [], @mean), accumarray(quint, cf.Iou, [], @mean), ...
    'VariableNames', {'Q', 'N', 'PrecDefault', 'PrecCf', 'RecDefault', 'RecCf', 'IouDefault', 'IouCf'});
precGuard = mean(cf.Precision(quint <= 3)) - mean(prec(quint <= 3, 1));
recQ45 = mean(cf.Recall(quint >= 4)) - mean(rec(quint >= 4, 1));

gAll = pick(mean(iou, 1), tieTol);   % what would ship (Reporting rule 6)
d = cf.Iou - iou(:, 1);
[~, o] = sort(d, 'descend');
moves = table(base.Image(o), dset(o), d(o), iou(o, 1), cf.Iou(o), ...
    'VariableNames', {'Image', 'Dataset', 'dIou', 'IouDefault', 'IouCf'});

out = struct('surface', surface, 'chosen', chosen, 'byQuint', byQuint, ...
    'cfIou', mean(cf.Iou), 'cfPrecision', mean(cf.Precision), 'cfRecall', mean(cf.Recall), ...
    'cfF1', mean(cf.F1), 'baseIou', mean(iou(:, 1)), 'dMean', mean(d), ...
    'dUp', nnz(d > 1e-12), 'dDown', nnz(d < -1e-12), 'dCi', bootCi(d), ...
    'precGuard', precGuard, 'recQ45', recQ45, 'allImagesCell', cells(gAll, :));

if ~exist(outDir, 'dir'), mkdir(outDir); end
writetable(surface, fullfile(outDir, 'surface.csv'));
writetable(chosen, fullfile(outDir, 'chosen.csv'));
writetable(byQuint, fullfile(outDir, 'by_quintile.csv'));
writetable(table(base.Image, dset, fold, quint, cells.Variant(pickIdx), cells.TauS(pickIdx), ...
    iou(:, 1), cf.Iou, d, cf.Precision, cf.Recall, 'VariableNames', {'Image', 'Dataset', 'Fold', ...
    'Quintile', 'Variant', 'TauS', 'IouDefault', 'IouCf', 'dIou', 'PrecisionCf', 'RecallCf'}), ...
    fullfile(outDir, 'crossfit_per_image.csv'));
save(fullfile(outDir, 'out.mat'), 'out');

fprintf('\n=== weak-seed fusion grid ===\n');
disp(surface); disp(chosen); disp(byQuint);
disp(moves([1:6, end-5:end], :));
fprintf(['cross-fitted: IoU %.4f | F1 %.4f | precision %.4f | recall %.4f\n' ...
    'vs default %.4f: dIoU %+.4f, %d up / %d down, 95%% CI [%+.4f, %+.4f]\n' ...
    'guard: Q1-Q3 precision %+.4f (limit -0.01); Q4-Q5 recall %+.4f\n' ...
    'chosen on all 69 images: %s, tau_s %.1f\n'], ...
    out.cfIou, out.cfF1, out.cfPrecision, out.cfRecall, out.baseIou, out.dMean, ...
    out.dUp, out.dDown, out.dCi, precGuard, recQ45, cells.Variant(gAll), cells.TauS(gAll));
for ds = unique(dset)'
    fprintf('  %s: cross-fitted %.4f, default %.4f\n', ds, mean(cf.Iou(dset == ds)), mean(iou(dset == ds, 1)));
end
end


function cfg = cellCfg(cfg, c)
cfg.Post.weakSeed = c.Variant;
if ~isnan(c.TauS)
    cfg.Post.weakSeedTau = c.TauS;
end
end


function counts = weakSeedCounts(cfg0, cells, images, seededCache)
% Per cell: weak seeds given seedless evidence, left to relax, left to the disk.
% Relax's test is replayed on the fused P (cnngvf.seedFallback).
[names, seeds] = cnngvf.loadPoints(cfg0);
G = height(cells);
n = zeros(G, 3);
for i = 1:numel(images)
    s = seeds(names == images(i), :);
    if isempty(s), continue; end
    Pk = double(cnngvf.probCache("read", seededCache, images(i), cfg0).P);
    Ps = double(cnngvf.probCache("read", cfg0.Post.noPointCache, images(i), cfg0).P);
    for g = 1:G
        cfg = cellCfg(cfg0, cells(g, :));
        [Pf, st] = cnngvf.fuseWeakSeeds(Pk, Ps, s, cfg);
        weak = find(st > 0 | (cfg.Post.weakSeed == "relax" & ~seedOn(Pk, s, cfg)));
        for k = weak'
            if st(k) == 1
                n(g, 1) = n(g, 1) + 1;
            elseif relaxFinds(Pf, s(k, :), cfg)
                n(g, 2) = n(g, 2) + 1;
            else
                n(g, 3) = n(g, 3) + 1;
            end
        end
    end
end
counts = array2table(n, 'VariableNames', {'WeakSeedless', 'WeakRelax', 'WeakDisk'});
end


function on = seedOn(P, s, cfg)
[h, w] = size(P);
r = min(max(round(s(:, 2)), 1), h);
c = min(max(round(s(:, 1)), 1), w);
on = P(sub2ind([h w], r, c)) > cfg.Post.tau;
end


function ok = relaxFinds(P, seed, cfg)
[h, w] = size(P);
row = min(max(round(seed(2)), 1), h);
col = min(max(round(seed(1)), 1), w);
win = cfg.Post.relaxWindow;
r0 = max(row - win, 1); r1 = min(row + win, h);
c0 = max(col - win, 1); c1 = min(col + win, w);
sub = P(r0:r1, c0:c1) > cfg.Post.relaxFloor;
ok = sub(row - r0 + 1, col - c0 + 1) && ...
    nnz(bwselect(sub, col - c0 + 1, row - r0 + 1, 8)) >= cfg.Post.MinArea;
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
