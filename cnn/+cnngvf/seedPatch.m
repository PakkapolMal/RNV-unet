function [X, rows, cols] = seedPatch(chans, seed, center, cfg)
%CNNGVF.SEEDPATCH Crop a seed-conditioned CNN patch out of a channel stack.
%
%  [X, rows, cols] = cnngvf.seedPatch(chans, seed, center, cfg)
%
%  chans  : h x w x C image channel stack (double or single)
%  seed   : [cx cy] the localization point, where the heat-map peaks
%  center : [cx cy] the crop centre. Equal to seed at inference; jittered
%           away from it during training so the network learns that the
%           heat-map - not the patch centre - marks the lesion.
%  cfg    : cnngvf.defaults()
%
%  X    : p x p x (C+1) single. The appended last channel is the Gaussian
%         seed heat-map. Crops running off the image are zero-padded.
%  rows : 1 x p image row indices covered by the crop (may fall outside
%         1..h; those columns of X are the zero padding)
%  cols : 1 x p image column indices, likewise
%
%  rows/cols let cnngvf.inferProbMap paste a prediction back where it came
%  from without recomputing the geometry.
%
%  See also CNNGVF.PREPAREDATA, CNNGVF.INFERPROBMAP.

p    = cfg.Patch.size;
half = floor(p / 2);
[h, w, C] = size(chans);

% Top-left corner in image coordinates. +1 so that for even p the seed lands
% at in-patch index half+1, which is what the heat-map below assumes.
r0 = round(center(2)) - half;
c0 = round(center(1)) - half;

rows = r0:(r0 + p - 1);
cols = c0:(c0 + p - 1);

X = zeros(p, p, C + 1, 'single');

% Copy only the part of the crop that actually overlaps the image.
inR = rows >= 1 & rows <= h;
inC = cols >= 1 & cols <= w;
X(inR, inC, 1:C) = single(chans(rows(inR), cols(inC), :));

% Gaussian heat-map at the seed's position within the patch. The seed is
% always inside the patch (jitter < half), so no clamping is needed.
sx = seed(1) - c0 + 1;
sy = seed(2) - r0 + 1;
[xx, yy] = meshgrid(1:p, 1:p);
X(:, :, C + 1) = single(exp(-((xx - sx).^2 + (yy - sy).^2) ./ (2 * cfg.Patch.sigmaSeed^2)));
end
