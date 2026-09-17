function row = refineScore(T, Q)
%CNNGVF.REFINESCORE Every aggregate ../docs/REFINE.md reports, as one table row.
%
%  row = cnngvf.refineScore(T)
%  row = cnngvf.refineScore(T, Q)
%
%  T : per-image table with Image, Dataset, Fold, GtPx, PredPx, TP, FP, FN, Iou,
%      F1, Precision, Recall - one row per image, as run_refine builds it
%  Q : table Image, Q (1..5), the frozen size quintiles (refine/quintiles.mat).
%      Omit to skip the quintile columns.
%
%  row : 1-row table, so rows for several arms vertcat into summary.csv.
%        Mean*/MedianIou  per image, then averaged
%        Pooled*          TP, FP, FN summed over all images, then computed once
%        Iou_<dataset>    mean IoU per dataset
%        Iou_fold<k>      mean IoU per fold
%        Iou_Q<q>         mean IoU per GT-area quintile
%        PredGt_Q<q>      sum(PredPx) / sum(GtPx) in that quintile (pooled ratio,
%                         not a mean of per-image ratios, which tiny lesions dominate)
%
%  See also RUN_REFINE, CNNGVF.CROSSFIT.

tp = sum(T.TP); fp = sum(T.FP); fn = sum(T.FN);
pp = tp / max(tp + fp, eps);
pr = tp / max(tp + fn, eps);

row = table(height(T), mean(T.Iou), median(T.Iou), mean(T.F1), mean(T.Precision), mean(T.Recall), ...
    tp / max(tp + fp + fn, eps), 2 * pp * pr / max(pp + pr, eps), pp, pr, ...
    'VariableNames', {'N', 'MeanIou', 'MedianIou', 'MeanF1', 'MeanPrec', 'MeanRec', ...
    'PooledIou', 'PooledF1', 'PooledPrec', 'PooledRec'});

for ds = unique(T.Dataset)'
    row.("Iou_" + ds) = mean(T.Iou(T.Dataset == ds));
end
for f = unique(T.Fold)'
    row.("Iou_fold" + f) = mean(T.Iou(T.Fold == f));
end

if nargin >= 2
    [ok, loc] = ismember(T.Image, Q.Image);
    assert(all(ok), 'cnngvf:refineScore:noQuintile', 'An image has no frozen quintile.');
    q = Q.Q(loc);
    for k = 1:5
        s = q == k;
        row.("Iou_Q" + k)    = mean(T.Iou(s));
        row.("PredGt_Q" + k) = sum(T.PredPx(s)) / max(sum(T.GtPx(s)), eps);
    end
end
end
