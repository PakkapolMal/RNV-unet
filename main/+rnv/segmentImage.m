function [mask, contours, img] = segmentImage(src, seeds, cfg)
%RNV.SEGMENTIMAGE Segment every RNV lesion in one image.
%
%  [mask, contours, img] = rnv.segmentImage(imgPath, seeds, cfg)
%
%  seeds    : S x 2 array of [cx cy] localization points for this image
%  mask     : union of all per-seed lesion masks
%  contours : S x 1 cell of final contours ([row col]); empty where the seed
%             produced nothing usable
%  img      : the loaded image struct, so callers can draw overlays without
%             re-reading from disk
%
%  Seeds are segmented independently and OR'd, matching how the results are
%  scored against the per-image ground truth.
%
%  See also RNV.SEGMENTSEED, RNV.METRICS.

img = rnv.loadImage(src, cfg);

nSeeds   = size(seeds, 1);
mask     = false(img.h, img.w);
contours = cell(nSeeds, 1);

for s = 1:nSeeds
    [m, O] = rnv.segmentSeed(seeds(s, :), img, cfg);
    if isempty(m)
        continue;
    end
    contours{s} = O;
    mask = mask | m;
end
end
