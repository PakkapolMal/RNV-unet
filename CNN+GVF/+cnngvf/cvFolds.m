function [folds, info] = cvFolds(cfg)
%CNNGVF.CVFOLDS Area-stratified cross-validation folds over the GT images.
%
%  [folds, info] = cnngvf.cvFolds(cfg)
%
%  folds : cfg.CV.k x 1 cell, each a string array of image names to hold out
%  info  : table of Image / GtPx / Fold, for checking the stratification
%
%  Deterministic: the same images always land in the same folds, so two
%  configurations can be compared paired image-by-image rather than each
%  against its own random split.
%
%  Stratified by ground-truth lesion area, not shuffled. Areas span 2 016 px
%  (573.png) to 239 227 px (1115.png) over 17 images, and performance tracks
%  size strongly - the largest lesions are the ones a single patch cannot
%  cover. A random split could put several giants in one fold and make that
%  fold unrepresentative of the whole. Dealing the area-sorted list round-robin
%  spreads them one per fold.
%
%  Every ground-truth image appears in exactly one fold, so concatenating the
%  per-fold results gives one out-of-sample score per image.
%
%  See also RUN_CV, CNNGVF.PREPAREDATA.

d = dir(fullfile(cfg.Paths.GT, '*.png'));
if isempty(d)
    error('cnngvf:cvFolds:noGt', 'No ground-truth masks in %s', cfg.Paths.GT);
end

images = string({d.name})';
area   = zeros(numel(images), 1);
for i = 1:numel(images)
    area(i) = nnz(rnv.loadMask(fullfile(cfg.Paths.GT, char(images(i)))));
end

% Deal the area-sorted list round-robin: largest to fold 1, next to fold 2, ...
[area, ord] = sort(area, 'descend');
images = images(ord);

k = cfg.CV.k;
if k < 2 || k > numel(images)
    error('cnngvf:cvFolds:badK', ...
        'cfg.CV.k = %d is not usable with %d ground-truth images.', k, numel(images));
end

assign = mod((0:numel(images) - 1)', k) + 1;

folds = cell(k, 1);
for f = 1:k
    folds{f} = images(assign == f);
end

info = table(images, area, assign, 'VariableNames', {'Image', 'GtPx', 'Fold'});
end
