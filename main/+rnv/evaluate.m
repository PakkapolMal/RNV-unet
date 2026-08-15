function [summary, perImage] = evaluate(cfg, valImages)
%RNV.EVALUATE Score the pipeline against the ground-truth image set.
%
%  [summary, perImage] = rnv.evaluate(cfg)
%  [summary, perImage] = rnv.evaluate(cfg, valImages)
%
%  valImages : optional string array of GT file names; defaults to every
%              *.png in cfg.Paths.GT.
%
%  summary  : struct with .meanIou, .meanF1, .nScored, .runtimeSec
%  perImage : table (Image, Iou, F1, Seeds), NaN where the image had no
%             seeds or no GT.
%
%  All seeds of an image are segmented and OR'd before scoring, matching
%  the per-image ground truth.
%
%  See also RNV.SEGMENTIMAGE, RNV.METRICS.

if nargin < 2 || isempty(valImages)
    gtFiles   = dir(fullfile(cfg.Paths.GT, '*.png'));
    valImages = sort(string({gtFiles.name}));
end
valImages = valImages(:);

if isempty(valImages)
    error('rnv:evaluate:noGt', 'No GT images found in %s', cfg.Paths.GT);
end

[names, seeds] = rnv.loadSeeds(cfg);

n   = numel(valImages);
iou = nan(n, 1);
f1  = nan(n, 1);
ns  = zeros(n, 1);

tStart = tic;
for i = 1:n
    imgName  = valImages(i);
    imgSeeds = seeds(names == imgName, :);
    ns(i)    = size(imgSeeds, 1);

    if ns(i) == 0
        warning('rnv:evaluate:noSeeds', 'No seeds for %s; skipped.', imgName);
        continue;
    end

    gt = rnv.loadMask(fullfile(cfg.Paths.GT, char(imgName)));
    if isempty(gt)
        warning('rnv:evaluate:noGtFile', 'GT missing for %s; skipped.', imgName);
        continue;
    end

    pred = rnv.segmentImage(fullfile(cfg.Paths.Img, char(imgName)), imgSeeds, cfg);

    m      = rnv.metrics(pred, gt);
    iou(i) = m.iou;
    f1(i)  = m.f1;
end
runtimeSec = toc(tStart);

perImage = table(valImages, iou, f1, ns, ...
    'VariableNames', {'Image', 'Iou', 'F1', 'Seeds'});

scored = ~isnan(iou);
summary = struct( ...
    'meanIou',    mean(iou(scored)), ...
    'meanF1',     mean(f1(scored)), ...
    'nScored',    nnz(scored), ...
    'runtimeSec', runtimeSec);
end
