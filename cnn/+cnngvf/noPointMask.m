function mask = noPointMask(P, cfg)
%CNNGVF.NOPOINTMASK The mask of an image with no localization points, from seedless P.
%
%  mask = cnngvf.noPointMask(P, cfg)
%
%  P : the seedless network's whole-frame probability map
%
%     P > Post.noPointTau -> imclose -> imfill -> drop below MinArea
%
%  No seed claims components, so every component is kept: keeping only the one
%  with the most P mass lost on every fold. +0.024 IoU on the 69-image headline;
%  docs/results/2026-11-12-seedless-fallback.md.
%
%  See also CNNGVF.MASKFROMPROB, RUN_CACHE.

BW = P > cfg.Post.noPointTau;
BW = imclose(BW, cfg.Post.MorphElement);
BW = imfill(BW, 'holes');
mask = bwareaopen(BW, cfg.Post.MinArea);
end
