function [img, chans] = loadChannels(src, cfg)
%CNNGVF.LOADCHANNELS Load an OCTA image as the CNN's input channel stack.
%
%  [img, chans] = cnngvf.loadChannels(imgPath, cfg)
%  [img, chans] = cnngvf.loadChannels(matrix,  cfg)
%
%  img   : the struct from rnv.loadImage (.I denoised grayscale, .raw, .edge,
%          .h, .w) - reused as-is so training and the classical pipeline in
%          RNV-Claude/main see identical pixels
%  chans : h x w x 2, [intensity, Frangi vesselness]
%
%  The vesselness channel is the front-end ../docs/CLAUDE.md credits for
%  suppressing the background texture a raw gradient latches onto. The seed
%  heat-map is appended later, per patch, by cnngvf.seedPatch.
%
%  See also CNNGVF.SEEDPATCH, CNNGVF.PREPAREDATA, CNNGVF.SEGMENTIMAGE.

img = rnv.loadImage(src, cfg);
V   = fibermetric(img.I, cfg.Vessel.scales, ObjectPolarity="bright");
chans = cat(3, img.I, V);
end
