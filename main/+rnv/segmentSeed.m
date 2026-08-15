function [mask, O, P] = segmentSeed(seed, img, cfg)
%RNV.SEGMENTSEED Segment one RNV lesion from one localization seed.
%
%  [mask, O, P] = rnv.segmentSeed(seed, img, cfg)
%
%  seed : [cx cy] localization point (column, row)
%  img  : struct from rnv.loadImage
%  cfg  : struct from rnv.defaults
%
%  mask : logical lesion mask, or [] if post-processing rejected everything
%  O    : final contour, N x 2 as [row col]
%  P    : initial contour, same layout (useful for debugging initialization)
%
%  Pipeline: initialize -> restrict the GVF field to an ROI around the
%  initial contour -> evolve the snake -> clean up.
%
%  See also RNV.INITCONTOUR, RNV.POSTPROCESS, RNV.SEGMENTIMAGE.

P = rnv.initContour(seed, img, cfg);

I = img.I;
if cfg.GvfMargin > 0
    % Zeroing outside a margin around the initial contour keeps distant
    % vasculature from bleeding into the diffused GVF field.
    roi = poly2mask(P(:, 2), P(:, 1), img.h, img.w);
    roi = imdilate(roi, strel('disk', cfg.GvfMargin));
    I   = I .* double(roi);
end

[O, J] = Snake2D(I, P, cfg.Snake);

[mask, O] = rnv.postprocess(O, J, seed, img, cfg);
end
