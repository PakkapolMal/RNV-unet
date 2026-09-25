function D = seedDisks(sz, seeds, BW, cfg)
%CNNGVF.SEEDDISKS A disk at every seed the threshold left with nothing.
%
%  D = cnngvf.seedDisks(sz, seeds, BW, cfg)
%
%  sz    : [h w] of the image
%  seeds : S x 2 [cx cy]
%  BW    : the thresholded foreground; a seed already standing on it is skipped
%  cfg   : reads cfg.Post.seedRadius
%
%  D : logical, the union of the planted disks. A recall floor, not a shape:
%      ~707 px at r = 15, against lesions of 1 106 - 432 533 px.
%
%  See also CNNGVF.SEEDFALLBACK.

D = false(sz);
if isempty(seeds)
    return;
end

r = cfg.Post.seedRadius;
[xx, yy] = meshgrid(1:sz(2), 1:sz(1));
for s = 1:size(seeds, 1)
    row = min(max(round(seeds(s, 2)), 1), sz(1));
    col = min(max(round(seeds(s, 1)), 1), sz(2));
    if BW(row, col)
        continue;   % already covered by a thresholded component
    end
    D = D | ((xx - seeds(s, 1)).^2 + (yy - seeds(s, 2)).^2 <= r^2);
end
end
