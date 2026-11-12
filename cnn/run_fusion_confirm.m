function out = run_fusion_confirm(seedlessCache, outDir)
%RUN_FUSION_CONFIRM Score the two fixed fusion cells on a seedless CV cache.
%
%  out = run_fusion_confirm(seedlessCache)              % outDir: <Output>/fusion_confirm
%  out = run_fusion_confirm(seedlessCache, outDir)
%
%  Cells, fixed before the run (docs/briefs/fusion-confirm.md), on the images
%  with points; images with no points keep stage 9 on the adopted seedless cache:
%    off  the current pipeline
%    (a)  max(seeded P, seedless P > 0.9) over the whole image, maskFromProb
%    (b)  (a), keeping only the pieces (a) adds of at least MinArea px
%  Nothing is selected, so every score is out-of-sample as it stands.

cfg0 = cnngvf.defaults();
if nargin < 2
    outDir = fullfile(cfg0.Paths.Output, 'fusion_confirm');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
seededCache = fullfile(cfg0.Paths.Output, 'cv', 'probcache');
cfgA = cfg0;
cfgA.Post.fuseCut = -Inf;
cfgA.Post.fuseTauS = 0.9;

M = cnngvf.manifest(cfg0);
[names, seeds] = cnngvf.loadPoints(cfg0);
n = height(M);
[iou, prec, rec] = deal(zeros(n, 3));   % off, (a), (b)
target = nan(n, 1);                     % stage 9 on this seedless map, for images with no points
for i = 1:n
    gt = cnngvf.loadMask(M.GtPath(i));
    Ps = double(cnngvf.probCache("read", seedlessCache, M.Name(i), cfg0).P);
    s = seeds(names == M.Name(i), :);
    if isempty(s)
        adopted = double(cnngvf.probCache("read", cfg0.Post.noPointCache, M.Name(i), cfg0).P);
        m = cnngvf.metrics(cnngvf.noPointMask(adopted, cfg0), gt);
        [iou(i, :), prec(i, :), rec(i, :)] = deal(m.iou, m.precision, m.recall);
        target(i) = cnngvf.metrics(cnngvf.noPointMask(Ps, cfg0), gt).iou;
        continue;
    end
    Pk = double(cnngvf.probCache("read", seededCache, M.Name(i), cfg0).P);
    off = cnngvf.maskFromProb(Pk, s, cfg0);
    a = cnngvf.maskFromProb(cnngvf.fuseUncertain(Pk, Ps, cfgA), s, cfgA);
    b = (off & a) | bwareaopen(a & ~off, cfg0.Post.MinArea, 8);
    masks = {off, a, b};
    for k = 1:3
        m = cnngvf.metrics(masks{k}, gt);
        [iou(i, k), prec(i, k), rec(i, k)] = deal(m.iou, m.precision, m.recall);
    end
end

ref = readtable(cfg0.CV.reference, 'TextType', 'string');
[~, ia, ib] = intersect(M.Name, ref.Image);
sanity = max(abs(iou(ia, 1) - ref.Iou(ib)));

[~, ord] = sort(M.GtPx);
rk = zeros(n, 1);
rk(ord) = 1:n;
quint = floor(5 * (rk - 1) / n) + 1;
q13 = quint <= 3;
pairs = [2 1; 3 1; 3 2];
cmp = table(["(a) - off"; "(b) - off"; "(b) - (a)"], zeros(3, 1), zeros(3, 2), zeros(3, 1), zeros(3, 1), ...
    zeros(3, 1), 'VariableNames', {'Comparison', 'dIou', 'Ci', 'Up', 'Down', 'dPrecQ13'});
for p = 1:3
    d = iou(:, pairs(p, 1)) - iou(:, pairs(p, 2));
    cmp.dIou(p) = mean(d);
    cmp.Ci(p, :) = bootCi(d);
    cmp.Up(p) = nnz(d > 1e-12);
    cmp.Down(p) = nnz(d < -1e-12);
    cmp.dPrecQ13(p) = mean(prec(q13, pairs(p, 1))) - mean(prec(q13, pairs(p, 2)));
end
summary = table(["off"; "(a)"; "(b)"], mean(iou)', mean(prec)', mean(rec)', ...
    mean(iou(M.Dataset == "DRAC22", :))', mean(iou(M.Dataset == "TUH", :))', ...
    'VariableNames', {'Cell', 'Iou', 'Precision', 'Recall', 'Iou_DRAC22', 'Iou_TUH'});
byQuint = table((1:5)', accumarray(quint, iou(:, 1), [], @mean), accumarray(quint, iou(:, 2), [], @mean), ...
    accumarray(quint, iou(:, 3), [], @mean), accumarray(quint, prec(:, 1), [], @mean), ...
    accumarray(quint, prec(:, 2), [], @mean), accumarray(quint, prec(:, 3), [], @mean), ...
    accumarray(quint, rec(:, 1), [], @mean), accumarray(quint, rec(:, 2), [], @mean), ...
    'VariableNames', {'Q', 'IouOff', 'IouA', 'IouB', 'PrecOff', 'PrecA', 'PrecB', 'RecOff', 'RecA'});
per = table(M.Name, M.Dataset, M.GtPx, iou(:, 1), iou(:, 2), iou(:, 3), iou(:, 2) - iou(:, 1), ...
    iou(:, 3) - iou(:, 1), 'VariableNames', {'Image', 'Dataset', 'GtPx', 'IouOff', 'IouA', 'IouB', 'dA', 'dB'});
t = ~isnan(target);
out = struct('summary', summary, 'cmp', cmp, 'byQuint', byQuint, 'perImage', per, 'sanity', sanity, ...
    'targets', table(M.Name(t), target(t), iou(t, 1), 'VariableNames', {'Image', 'IouThisMap', 'IouAdopted'}));
writetable(per, fullfile(outDir, 'per_image.csv'));
save(fullfile(outDir, 'out.mat'), 'out');

fprintf('\n=== fusion confirmation | %s ===\nsanity: off vs reference max |dIoU| %.3g\n', seedlessCache, sanity);
disp(summary); disp(cmp); disp(byQuint);
per = sortrows(per, 'dA');
disp(per([1:6, end-3:end], :));
disp(out.targets);
end


function ci = bootCi(d)
% Percentile bootstrap on the mean, seeded as the earlier records.
old = rng(20260917);
restore = onCleanup(@() rng(old));
means = sort(mean(d(randi(numel(d), numel(d), 10000)), 1));
ci = means([250 9751]);
end
