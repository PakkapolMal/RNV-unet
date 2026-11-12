function out = run_fusion_u_grid(outDir)
%RUN_FUSION_U_GRID Cross-fit fusing the seedless P on images whose seeded P is unsure.
%
%  out = run_fusion_u_grid             % outDir: <Output>/fusion_u_grid
%  out = run_fusion_u_grid(outDir)
%
%  Each image with points is scored three ways, off and fused at tau_s 0.9 and
%  0.5 (cnngvf.fuseUncertain, then cnngvf.maskFromProb unchanged); images with no
%  points keep stage 9. A cell (fraction fused q, tau_s) fuses the images whose U
%  is in the top q; for held-out fold f the cut-off is set on the images with
%  points outside f, and the cell is chosen there too. One cell is re-run through
%  run_cache as a consistency check. Pre-registration:
%  docs/briefs/fusion-uncertainty.md.

cfg0 = cnngvf.defaults();
if nargin < 1
    outDir = fullfile(cfg0.Paths.Output, 'fusion_u_grid');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
seededCache = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
taus = [0.9 0.5];

% Off first, then the conservative order ties fall to: smaller q, then tau_s 0.9.
[T, Q] = meshgrid(taus, [0.10 0.25 0.50 1.00]);
cells = [table(0, NaN, 'VariableNames', {'Q', 'TauS'}); ...
    table(reshape(Q', [], 1), reshape(T', [], 1), 'VariableNames', {'Q', 'TauS'})];
G = height(cells);
tieTol = 0.001;

M = cnngvf.manifest(cfg0);
[names, seeds] = cnngvf.loadPoints(cfg0);
[~, info] = cnngvf.cvFolds(cfg0);
[~, loc] = ismember(M.Name, info.Image);
fold = info.Fold(loc);
n = height(M);
hasPts = false(n, 1);
U = nan(n, 1);
[iou, prec, rec] = deal(zeros(n, 3));   % columns: off, fused tau_s 0.9, fused tau_s 0.5
added = zeros(n, 2);                    % px fusion added to the final mask, per tau_s
for i = 1:n
    Pk = double(cnngvf.probCache("read", seededCache, M.Name(i), cfg0).P);
    Ps = double(cnngvf.probCache("read", cfg0.Post.noPointCache, M.Name(i), cfg0).P);
    gt = cnngvf.loadMask(M.GtPath(i));
    s = seeds(names == M.Name(i), :);
    hasPts(i) = ~isempty(s);
    if ~hasPts(i)
        m = cnngvf.metrics(cnngvf.noPointMask(Ps, cfg0), gt);   % stage 9, the same in every cell
        [iou(i, :), prec(i, :), rec(i, :)] = deal(m.iou, m.precision, m.recall);
        continue;
    end
    U(i) = cnngvf.seededUncertainty(Pk);
    off = cnngvf.maskFromProb(Pk, s, cfg0);
    m = cnngvf.metrics(off, gt);
    [iou(i, 1), prec(i, 1), rec(i, 1)] = deal(m.iou, m.precision, m.recall);
    for t = 1:2
        cfg = cfg0;
        cfg.Post.fuseCut = -Inf;
        cfg.Post.fuseTauS = taus(t);
        mask = cnngvf.maskFromProb(cnngvf.fuseUncertain(Pk, Ps, cfg), s, cfg);
        m = cnngvf.metrics(mask, gt);
        [iou(i, t + 1), prec(i, t + 1), rec(i, t + 1)] = deal(m.iou, m.precision, m.recall);
        added(i, t) = nnz(mask & ~off);
    end
end

% Sanity: off reproduces the adopted reference.
ref = readtable(cfg0.CV.reference, 'TextType', 'string');
[~, ia, ib] = intersect(M.Name, ref.Image);
sanity = max(abs(iou(ia, 1) - ref.Iou(ib)));

% Which column each image takes under a cell with cut-off c.
col = @(g, c) 1 + (hasPts & U >= c & g > 1) .* (1 + (cells.TauS(max(g, 1)) == 0.5));
cutOn = @(q, set) topCut(U(set & hasPts), q);

% Fixed-cell surface: cut-off set on all 66, scored on all 69 (description only).
surface = cells;
[surface.Cut, surface.NFused, surface.Iou, surface.Precision, surface.Recall, ...
    surface.Iou_DRAC22, surface.Iou_TUH] = deal(zeros(G, 1));
pickAll = zeros(n, G);
for g = 1:G
    c = Inf;
    if g > 1, c = cutOn(cells.Q(g), true(n, 1)); end
    pickAll(:, g) = col(g, c);
    at = sub2ind(size(iou), (1:n)', pickAll(:, g));
    surface.Cut(g) = c;
    surface.NFused(g) = nnz(pickAll(:, g) > 1);
    surface.Iou(g) = mean(iou(at));
    surface.Precision(g) = mean(prec(at));
    surface.Recall(g) = mean(rec(at));
    surface.Iou_DRAC22(g) = mean(iou(at(M.Dataset == "DRAC22")));
    surface.Iou_TUH(g) = mean(iou(at(M.Dataset == "TUH")));
end

% Cross-fit: cut-off and cell from the other folds, scored on this one.
pick = ones(n, 1);
chosen = table();
for f = unique(fold)'
    te = fold == f;
    trainIou = zeros(1, G);
    cutF = Inf(1, G);
    for g = 2:G
        cutF(g) = cutOn(cells.Q(g), ~te);
        c = col(g, cutF(g));
        at = sub2ind(size(iou), find(~te), c(~te));
        trainIou(g) = mean(iou(at));
    end
    trainIou(1) = mean(iou(~te, 1));
    g = find(trainIou >= max(trainIou) - tieTol, 1, 'first');
    c = col(g, cutF(g));
    pick(te) = c(te);
    chosen = [chosen; table(f, cells.Q(g), cells.TauS(g), cutF(g), nnz(te & c > 1), trainIou(g), ...
        mean(iou(sub2ind(size(iou), find(te), c(te)))), mean(iou(te, 1)), 'VariableNames', ...
        {'Fold', 'Q', 'TauS', 'Cut', 'NFused', 'TrainIou', 'TestIou', 'TestIouOff'})]; %#ok<AGROW>
end
at = sub2ind(size(iou), (1:n)', pick);
cf = struct('Iou', iou(at), 'Precision', prec(at), 'Recall', rec(at));
d = cf.Iou - iou(:, 1);

[~, ord] = sort(M.GtPx);
rk = zeros(n, 1);
rk(ord) = 1:n;
quint = floor(5 * (rk - 1) / n) + 1;
byQuint = table((1:5)', accumarray(quint, prec(:, 1), [], @mean), accumarray(quint, cf.Precision, [], @mean), ...
    accumarray(quint, rec(:, 1), [], @mean), accumarray(quint, cf.Recall, [], @mean), ...
    accumarray(quint, iou(:, 1), [], @mean), accumarray(quint, cf.Iou, [], @mean), ...
    'VariableNames', {'Q', 'PrecOff', 'PrecCf', 'RecOff', 'RecCf', 'IouOff', 'IouCf'});
precGuard = mean(cf.Precision(quint <= 3)) - mean(prec(quint <= 3, 1));
recQ45 = mean(cf.Recall(quint >= 4)) - mean(rec(quint >= 4, 1));

% Consistency: the (all, 0.9) cell through run_cache equals column 2 here.
cfg = cfg0;
cfg.Paths.Output = outDir;
cfg.Post.fuseCut = -Inf;
cfg.Post.fuseTauS = 0.9;
rc = run_cache(cfg, seededCache);
[~, ia, ib] = intersect(M.Name, rc.Image);
consistency = max(abs(iou(ia, 2) - rc.Iou(ib)));

% Diagnostic (uses GT): does U rank the poor seeded images?
p = hasPts;
rho = spearman(U(p), iou(p, 1));
gAll = find(surface.Iou >= max(surface.Iou) - tieTol, 1, 'first');
[~, o] = sort(d, 'descend');
moves = table(M.Name(o), M.Dataset(o), U(o), d(o), iou(o, 1), cf.Iou(o), ...
    'VariableNames', {'Image', 'Dataset', 'U', 'dIou', 'IouOff', 'IouCf'});

out = struct('surface', surface, 'chosen', chosen, 'byQuint', byQuint, ...
    'cfIou', mean(cf.Iou), 'cfPrecision', mean(cf.Precision), 'cfRecall', mean(cf.Recall), ...
    'baseIou', mean(iou(:, 1)), 'dMean', mean(d), 'dUp', nnz(d > 1e-12), 'dDown', nnz(d < -1e-12), ...
    'dCi', bootCi(d), 'precGuard', precGuard, 'recQ45', recQ45, 'sanity', sanity, ...
    'consistency', consistency, 'rhoUIou', rho, 'shipCell', surface(gAll, :), ...
    'addedMedian', [median(added(p, 1)) median(added(p, 2))], 'U', table(M.Name(p), U(p), ...
    iou(p, 1), 'VariableNames', {'Image', 'U', 'IouOff'}));
writetable(surface, fullfile(outDir, 'surface.csv'));
writetable(chosen, fullfile(outDir, 'chosen.csv'));
writetable(byQuint, fullfile(outDir, 'by_quintile.csv'));
writetable(moves, fullfile(outDir, 'crossfit_per_image.csv'));
save(fullfile(outDir, 'out.mat'), 'out');

fprintf('\n=== uncertainty fusion grid ===\n');
fprintf('sanity: off vs reference max |dIoU| %.3g; consistency vs run_cache %.3g\n', sanity, consistency);
disp(surface); disp(chosen); disp(byQuint);
disp(moves([1:6, end-5:end], :));
fprintf(['cross-fitted: IoU %.4f | precision %.4f | recall %.4f\n' ...
    'vs off %.4f: dIoU %+.4f, %d up / %d down, 95%% CI [%+.4f, %+.4f]\n' ...
    'guard: Q1-Q3 precision %+.4f (limit -0.01); Q4-Q5 recall %+.4f\n' ...
    'Spearman(U, seeded IoU) over %d images: %+.3f; median px added when fused: %d (0.9), %d (0.5)\n'], ...
    out.cfIou, out.cfPrecision, out.cfRecall, out.baseIou, out.dMean, out.dUp, out.dDown, out.dCi, ...
    precGuard, recQ45, nnz(p), rho, out.addedMedian);
for ds = ["DRAC22" "TUH"]
    s = M.Dataset == ds;
    fprintf('  %s: cross-fitted %.4f, off %.4f\n', ds, mean(cf.Iou(s)), mean(iou(s, 1)));
end
end


function c = topCut(u, q)
% The cut-off that fuses the top q of u: its k-th largest value, k = round(q n).
if q >= 1
    c = -Inf;
    return;
end
u = sort(u, 'descend');
c = u(max(1, round(q * numel(u))));
end


function r = spearman(x, y)
r = corrcoef(ranks(x), ranks(y));
r = r(1, 2);
end


function r = ranks(x)
% Ranks with ties averaged.
[s, i] = sort(x(:));
r = zeros(size(s));
k = 1;
while k <= numel(s)
    j = k;
    while j < numel(s) && s(j + 1) == s(k), j = j + 1; end
    r(i(k:j)) = (k + j) / 2;
    k = j + 1;
end
end


function ci = bootCi(d)
% Percentile bootstrap on the mean, seeded as the earlier cross-fit records.
old = rng(20260917);
restore = onCleanup(@() rng(old));
means = sort(mean(d(randi(numel(d), numel(d), 10000)), 1));
ci = means([250 9751]);
end
