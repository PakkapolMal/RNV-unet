function m = metrics(pred, gt)
%RNV.METRICS Overlap metrics between a predicted mask and ground truth.
%
%  m = rnv.metrics(pred, gt)
%
%  Returns a struct with .iou, .f1, .precision, .recall. Both inputs are
%  coerced to logical; a size mismatch resizes gt (nearest neighbour) and
%  warns rather than failing silently.
%
%  Empty prediction and empty GT scores 1 (a correct "nothing here"), which
%  is the convention used when evaluating images with no lesion.
%
%  See also RNV.EVALUATE.

pred = logical(pred);
gt   = logical(gt);

if ~isequal(size(pred), size(gt))
    warning('rnv:metrics:sizeMismatch', ...
        'GT %s resized to prediction size %s.', mat2str(size(gt)), mat2str(size(pred)));
    gt = imresize(gt, size(pred), 'nearest');
end

tp = nnz(pred & gt);
fp = nnz(pred) - tp;
fn = nnz(gt) - tp;

if tp + fp + fn == 0
    m = struct('iou', 1, 'f1', 1, 'precision', 1, 'recall', 1);
    return;
end

m.iou       = tp / (tp + fp + fn);
m.precision = tp / max(tp + fp, eps);
m.recall    = tp / max(tp + fn, eps);
m.f1        = 2 * m.precision * m.recall / max(m.precision + m.recall, eps);
end
