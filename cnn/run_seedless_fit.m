function out = run_seedless_fit(seedlessCache, outDir)
%RUN_SEEDLESS_FIT Cross-fit the no-point rule on the images with 1-3 points, score it on those with none.
%
%  out = run_seedless_fit(seedlessCache)            % outDir: <Output>/seedless_fit
%  out = run_seedless_fit(seedlessCache, outDir)
%
%  seedlessCache holds the seedless network's CV P (colab/rnvtorch.py
%  train-seedless). Every read goes through cnngvf.probCache's leakage guard.
%
%  Grid: Post.noPointTau x Post.noPointKeep. The proxy is the images with 1-3
%  localization points, scored with cnngvf.noPointMask as if they had none:
%  chosen by point count, which is known at inference, so no GT selects it.
%  A target (zero points) in fold f is scored with the cell best on the proxy
%  images outside fold f; the targets never choose. Then run_cache with the
%  rule on checks the 66 images with points are unchanged.
%  Pre-registration: docs/briefs/seedless-fallback.md.

cfg0 = cnngvf.defaults();
if nargin < 2
    outDir = fullfile(cfg0.Paths.Output, 'seedless_fit');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

% Nearest the default first, so ties go to the default, then tau, then "all".
taus  = [0.50 0.40 0.60 0.30 0.70 0.80 0.90];
keeps = ["all" "top1"];
[K, T] = meshgrid(1:numel(keeps), taus);
cells = table(reshape(T', [], 1), reshape(keeps(K'), [], 1), 'VariableNames', {'Tau', 'Keep'});
assert(cells.Tau(1) == cfg0.Post.noPointTau && cells.Keep(1) == cfg0.Post.noPointKeep, ...
    'the default must be cell 1');
tieTol = 0.001;
G = height(cells);

M = cnngvf.manifest(cfg0);
[names, ~] = cnngvf.loadPoints(cfg0);
nPts = arrayfun(@(n) nnz(names == n), M.Name);
[~, info] = cnngvf.cvFolds(cfg0);
[~, loc] = ismember(M.Name, info.Image);
fold = info.Fold(loc);
target = nPts == 0;
proxy  = nPts >= 1 & nPts <= 3;
fprintf('run_seedless_fit | %d targets, %d proxy images | %s\n', nnz(target), nnz(proxy), seedlessCache);

% Every cell on every image: the proxy for choosing, the targets for the gate,
% all 69 for the diagnostic.
n = height(M);
[iou, prec, rec, px] = deal(zeros(n, G));
agree = nan(n, 4);   % seedless vs seeded P > 0.5: in GT both, seedless only; outside GT likewise
seededCache = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
for i = 1:n
    c = cnngvf.probCache("read", seedlessCache, M.Name(i), cfg0);
    P = double(c.P);
    gt = cnngvf.loadMask(M.GtPath(i));
    for g = 1:G
        cfg = cfg0;
        cfg.Post.noPointTau = cells.Tau(g);
        cfg.Post.noPointKeep = cells.Keep(g);
        mask = cnngvf.noPointMask(P, cfg);
        m = cnngvf.metrics(mask, gt);
        [iou(i, g), prec(i, g), rec(i, g), px(i, g)] = deal(m.iou, m.precision, m.recall, nnz(mask));
    end
    if nPts(i) > 0
        s = cnngvf.probCache("read", seededCache, M.Name(i), cfg0);
        a = P > 0.5;
        b = double(s.P) > 0.5;
        agree(i, :) = [mean(a(gt) & b(gt)), mean(a(gt) & ~b(gt)), mean(a(~gt) & b(~gt)), mean(a(~gt) & ~b(~gt))];
    end
end

surface = [cells, table(mean(iou(proxy, :))', mean(prec(proxy, :))', mean(rec(proxy, :))', ...
    mean(iou(target, :))', mean(iou)', 'VariableNames', ...
    {'ProxyIou', 'ProxyPrecision', 'ProxyRecall', 'TargetIou', 'AllIou'})];

% Cross-fit: each fold's targets and proxy images get the cell chosen on the
% other folds' proxy images.
pickIdx = zeros(n, 1);
chosen = table();
for f = unique(fold)'
    te = fold == f;
    g = pick(mean(iou(proxy & ~te, :), 1), tieTol);
    pickIdx(te) = g;
    chosen = [chosen; table(f, cells.Tau(g), cells.Keep(g), nnz(proxy & ~te), mean(iou(proxy & ~te, g)), ...
        'VariableNames', {'Fold', 'Tau', 'Keep', 'NProxyOther', 'ProxyIouOther'})]; %#ok<AGROW>
end
at = sub2ind(size(iou), (1:n)', pickIdx);
cfIou = iou(at);
gAll = pick(mean(iou(proxy, :), 1), tieTol);   % what ships (Reporting rule 6)

tgt = table(M.Name(target), M.Dataset(target), fold(target), M.GtPx(target), cells.Tau(pickIdx(target)), ...
    cells.Keep(pickIdx(target)), px(at(target)), cfIou(target), prec(at(target)), rec(at(target)), ...
    'VariableNames', {'Image', 'Dataset', 'Fold', 'GtPx', 'Tau', 'Keep', 'PredPx', 'Iou', 'Precision', 'Recall'});

% Isolation: run_cache with the rule on, at the cell that would ship.
cfg = cfg0;
cfg.Paths.Output = outDir;
cfg.Post.noPointCache = string(seedlessCache);
cfg.Post.noPointTau = cells.Tau(gAll);
cfg.Post.noPointKeep = cells.Keep(gAll);
perImage = run_cache(cfg, seededCache);   % Output now points at outDir
ref = readtable(cfg0.CV.reference, 'TextType', 'string');
[~, ia, ib] = intersect(perImage.Image, ref.Image);
withPts = perImage.Seeds(ia) > 0;
isolation = max(abs(perImage.Iou(ia(withPts)) - ref.Iou(ib(withPts))));

out = struct('surface', surface, 'chosen', chosen, 'targets', tgt, ...
    'targetMeanIou', mean(cfIou(target)), 'proxyCfIou', mean(cfIou(proxy)), ...
    'proxyCfPrecision', mean(prec(at(proxy))), 'proxyCfRecall', mean(rec(at(proxy))), ...
    'headlineGain', sum(cfIou(target)) / n, 'isolation', isolation, ...
    'shipCell', cells(gAll, :), 'shipHeadline', mean(perImage.Iou), ...
    'agreement', array2table(mean(agree, 1, 'omitnan'), 'VariableNames', ...
        {'InGtBoth', 'InGtSeedlessOnly', 'OutGtBoth', 'OutGtSeedlessOnly'}));
for ds = unique(M.Dataset)'
    s = M.Dataset == ds;
    out.diagnostic.(ds) = table(mean(iou(s, 1)), mean(iou(s, gAll)), mean(prec(s, gAll)), mean(rec(s, gAll)), ...
        'VariableNames', {'IouDefaultCell', 'IouShipCell', 'PrecisionShipCell', 'RecallShipCell'});
end

writetable(surface, fullfile(outDir, 'surface.csv'));
writetable(chosen, fullfile(outDir, 'chosen.csv'));
writetable(tgt, fullfile(outDir, 'targets.csv'));
writetable(table(M.Name, M.Dataset, fold, nPts, M.GtPx, iou(:, 1), cfIou, cells.Tau(pickIdx), cells.Keep(pickIdx), ...
    'VariableNames', {'Image', 'Dataset', 'Fold', 'Points', 'GtPx', 'IouDefaultCell', 'IouCf', 'Tau', 'Keep'}), ...
    fullfile(outDir, 'per_image.csv'));
save(fullfile(outDir, 'out.mat'), 'out');

fprintf('\n=== seedless no-point rule | %s ===\n', seedlessCache);
disp(surface); disp(chosen); disp(tgt);
fprintf(['gate 1  target mean IoU %.4f (>= 0.115), headline %+.4f\n' ...
    'gate 2  proxy cross-fitted IoU %.4f (>= 0.20); precision %.4f, recall %.4f\n' ...
    'gate 3  max |dIoU| on the %d images with points %.3g (~1e-15)\n' ...
    'ships   tau %.2f, keep %s; headline with the rule on %.4f\n'], ...
    out.targetMeanIou, out.headlineGain, out.proxyCfIou, out.proxyCfPrecision, out.proxyCfRecall, ...
    nnz(withPts), isolation, cells.Tau(gAll), cells.Keep(gAll), out.shipHeadline);
disp(out.agreement);
for ds = unique(M.Dataset)'
    fprintf('diagnostic, rule on every %s image:\n', ds); disp(out.diagnostic.(ds));
end
end


function g = pick(m, tieTol)
g = find(m >= max(m) - tieTol, 1, 'first');
end
