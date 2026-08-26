function [img, chans] = loadChannels(src, cfg)
%CNNGVF.LOADCHANNELS Load an OCTA image as the CNN's input channel stack.
%
%  [img, chans] = cnngvf.loadChannels(imgPath, cfg)
%  [img, chans] = cnngvf.loadChannels(matrix,  cfg)
%
%  img   : the struct from rnv.loadImage (.I grayscale, .raw, .edge, .h, .w) -
%          reused as-is so training and the classical pipeline in
%          RNV-Claude/main see identical pixels
%  chans : h x w x C, [intensity] or [intensity, Frangi vesselness]
%
%  Two switches change C, and every caller derives the count from size(chans,3)
%  rather than assuming it:
%
%    cfg.Vessel.enabled  false drops the vesselness channel (C = 1)
%    cfg.Denoise         []    skips the median filter on .I
%
%  The vesselness channel is the front-end ../docs/CLAUDE.md credits for
%  suppressing the background texture a raw gradient latches onto; dropping it
%  is an ablation, not a speed-up. The seed heat-map is appended later, per
%  patch, by cnngvf.seedPatch, so the network sees C+1 inputs.
%
%  A network is tied to the C it was trained with. cnngvf.inferProbMap checks
%  and errors rather than producing quiet nonsense.
%
%  See also CNNGVF.SEEDPATCH, CNNGVF.PREPAREDATA, CNNGVF.SEGMENTIMAGE.

img = rnv.loadImage(src, cfg);

chans = img.I;
if cfg.Vessel.enabled
    V = fibermetric(img.I, cfg.Vessel.scales, ObjectPolarity="bright");
    chans = cat(3, chans, V);
end
end
