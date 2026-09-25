function [folds, info] = cvFolds(cfg)
%CNNGVF.CVFOLDS Area-stratified cross-validation folds over the image domain.
%
%  [folds, info] = cnngvf.cvFolds(cfg)
%
%  folds : cfg.CV.k x 1 cell, each a string array of image names to hold out
%  info  : table of Image / Dataset / GtPx / Fold, for checking stratification
%
%  Deterministic: the same images always land in the same folds, so two
%  configurations can be compared paired image-by-image rather than each
%  against its own random split.
%
%  Stratified by ground-truth lesion area, not shuffled. Areas span 1 106 px
%  (TUH gt (4)) to 432 533 px (TUH gt (21)) over the 68 domain images - a 391x
%  spread - and performance tracks size strongly, because the largest lesions
%  are the ones a single 512 px patch cannot cover. A random split could put
%  several giants in one fold and make that fold unrepresentative. Dealing the
%  area-sorted list round-robin spreads them one per fold.
%
%  Folds are POOLED across datasets, not stratified by dataset. The research
%  question is RNV segmentation rather than per-device performance, so a fold
%  is not restricted to one device. Area-sorted round-robin dealing interleaves
%  the datasets in practice anyway, since their area ranges overlap heavily.
%  info carries the Dataset column so the actual balance can be inspected, and
%  cnngvf.evaluate writes it into per_image.csv so a pooled run can be broken
%  out per dataset afterwards without being re-run.
%
%  Every domain image appears in exactly one fold, so concatenating the
%  per-fold results gives one out-of-sample score per image.
%
%  See also RUN_CV, CNNGVF.MANIFEST, CNNGVF.PREPAREDATA.

M = cnngvf.manifest(cfg);

% Deal the area-sorted list round-robin: largest to fold 1, next to fold 2, ...
M = sortrows(M, 'GtPx', 'descend');

k = cfg.CV.k;
if k < 2 || k > height(M)
    error('cnngvf:cvFolds:badK', ...
        'cfg.CV.k = %d is not usable with %d domain images.', k, height(M));
end

assign = mod((0:height(M) - 1)', k) + 1;

folds = cell(k, 1);
for f = 1:k
    folds{f} = M.Name(assign == f);
end

info = table(M.Name, M.Dataset, M.GtPx, assign, ...
    'VariableNames', {'Image', 'Dataset', 'GtPx', 'Fold'});
end
