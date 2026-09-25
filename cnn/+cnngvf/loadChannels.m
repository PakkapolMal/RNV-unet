function [img, chans] = loadChannels(src, cfg)
%CNNGVF.LOADCHANNELS Load an OCTA image as the CNN's input channel stack.
%
%  [img, chans] = cnngvf.loadChannels(imgPath, cfg)
%  [img, chans] = cnngvf.loadChannels(matrix,  cfg)
%
%  img   : struct
%            .raw  double grayscale in [0,1]
%            .I    .raw, median-filtered when cfg.Denoise is set
%            .h,.w image size
%  chans : h x w x C, [I] or [I, Frangi vesselness]
%
%  C is 1 unless cfg.Vessel.enabled. The seed heat-map is appended later, per
%  patch, by cnngvf.seedPatch, so the network sees C+1 inputs. Every caller
%  derives the count from size(chans,3), and cnngvf.inferProbMap errors if it
%  disagrees with the network.
%
%  See also CNNGVF.SEEDPATCH, CNNGVF.PREPAREDATA, CNNGVF.SEGMENTIMAGE.

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

chans = img.I;
if cfg.Vessel.enabled
    V = fibermetric(img.I, cfg.Vessel.scales, ObjectPolarity="bright");
    chans = cat(3, chans, V);
end
end
