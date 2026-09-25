function mask = loadMask(maskPath)
%CNNGVF.LOADMASK Read a binary mask PNG as logical.
%
%  mask = cnngvf.loadMask(maskPath)
%
%  Returns [] when the file does not exist, so callers can decide whether a
%  missing ground truth is a warning or a skip.
%
%  See also CNNGVF.EVALUATE, CNNGVF.METRICS.

maskPath = char(maskPath);
if ~isfile(maskPath)
    mask = [];
    return;
end

m = imread(maskPath);
if size(m, 3) == 3
    m = rgb2gray(m);
end
mask = im2double(m) > 0;
end
