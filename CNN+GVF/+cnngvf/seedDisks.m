function D = seedDisks(sz, seeds, BW, cfg)
%CNNGVF.SEEDDISKS A disk at every seed the threshold left with nothing.
%
%  D = cnngvf.seedDisks(sz, seeds, BW, cfg)
%
%  sz    : [h w] of the image
%  seeds : S x 2 [cx cy]
%  BW    : the thresholded foreground; a seed already standing on it is skipped
%  cfg   : reads cfg.Drlse.seedRadius
%
%  D : logical, the union of the planted disks
%
%  Exists so the radius and the trigger have ONE definition. cnngvf.initPhi
%  plants these to give the balloon and GVF terms something to grow from when
%  the CNN missed a lesion; cnngvf.maskFromProb plants the same disks so that
%  cfg.Post.mode is the only difference between the two paths; and run_ablate
%  needs the same again to build the pre-morphology mask the imclose cells
%  branch off. Three copies of a trigger condition is how two of them end up
%  disagreeing.
%
%  The trigger is the seed's own pixel, not its neighbourhood: a seed standing
%  on foreground needs no help, and one that is not does.
%
%  See also CNNGVF.INITPHI, CNNGVF.MASKFROMPROB, RUN_ABLATE.

D = false(sz);
if isempty(seeds)
    return;
end

r = cfg.Drlse.seedRadius;
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
