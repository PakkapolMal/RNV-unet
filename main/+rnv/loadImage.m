function img = loadImage(src, cfg)
%RNV.LOADIMAGE Load an OCTA image and derive its normalized edge map.
%
%  img = rnv.loadImage(imgPath, cfg)
%  img = rnv.loadImage(matrix,  cfg)   % already-loaded image (used by tests)
%
%  Returns a struct:
%    .I    double grayscale in [0,1], median-filtered if cfg.Denoise is set
%    .raw  double grayscale before denoising
%    .edge gradient magnitude of a Gaussian-blurred .I, scaled to max 1
%    .h,.w image size
%
%  The edge map is what the initializers probe along rays; .I is what the
%  snake's own external-force machinery consumes.
%
%  See also RNV.INITCONTOUR, RNV.SEGMENTIMAGE.

if ischar(src) || isstring(src)
    src = imread(char(src));
end

I = im2double(src);
if size(I, 3) == 3
    I = rgb2gray(I);
end

img.raw = I;
if ~isempty(cfg.Denoise)
    I = medfilt2(I, cfg.Denoise);
end
img.I = I;

[img.h, img.w] = size(I);

[Gx, Gy] = gradient(imgaussfilt(I, cfg.EdgeSigma));
E = hypot(Gx, Gy);
peak = max(E(:));
if peak > 0
    E = E ./ peak;
end
img.edge = E;
end
