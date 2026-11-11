function mask = noPointMask(P, cfg)
%CNNGVF.NOPOINTMASK The mask of an image with no localization points, from seedless P.
%
%  mask = cnngvf.noPointMask(P, cfg)
%
%  P : the seedless network's whole-frame probability map
%
%     P > Post.noPointTau -> imclose -> imfill -> drop below MinArea
%       -> Post.noPointKeep: "all" components, or "top1", the one with the
%          largest P mass (sum of P over it)
%
%  With no seeds there is nothing to claim components, so top-1 stands in for
%  claiming. It relies on every image in the domain containing RNV.
%  Experiment: docs/briefs/seedless-fallback.md.
%
%  See also CNNGVF.MASKFROMPROB, RUN_CACHE.

BW = P > cfg.Post.noPointTau;
BW = imclose(BW, cfg.Post.MorphElement);
BW = imfill(BW, 'holes');
BW = bwareaopen(BW, cfg.Post.MinArea);

switch cfg.Post.noPointKeep
    case "all"
        mask = BW;
    case "top1"
        cc = bwconncomp(BW);
        mask = false(size(BW));
        if cc.NumObjects > 0
            [~, k] = max(cellfun(@(ix) sum(P(ix)), cc.PixelIdxList));
            mask(cc.PixelIdxList{k}) = true;
        end
    otherwise
        error('cnngvf:noPointMask:keep', 'Unknown Post.noPointKeep "%s": use "all" or "top1".', ...
            cfg.Post.noPointKeep);
end
end
