function [mask, perSeed, contours, img, P] = segmentImage(src, seeds, net, cfg)
%CNNGVF.SEGMENTIMAGE Segment every RNV lesion in one image (Approach 1).
%
%  [mask, perSeed, contours, img, P] = cnngvf.segmentImage(imgPath, seeds, net, cfg)
%
%  seeds    : S x 2 [cx cy] localization points for this image
%  net      : dlnetwork from cnngvf.trainNet
%
%  mask     : logical h x w, the union of every claimed lesion
%  perSeed  : S x 1 cell, one mask per seed. This is the unit the evaluation
%             scores, against RNV-DRAC22GT/SegmentFilledGT/<stem>-<x>-<y>.png
%  contours : cell of N x 2 [row col] boundaries, ready for rnv.saveOverlay
%  img      : struct from rnv.loadImage, so callers can draw without re-reading
%  P        : the CNN probability map, returned for debugging the force field
%
%  Stages, per ../docs/spec.md Approach 1, when cfg.Post.mode is "levelset":
%     channels -> P -> f -> GVF field -> phi0 -> DRLSE -> cleanup
%
%  When cfg.Post.mode is "threshold" the middle four are replaced by
%  cnngvf.maskFromProb - P > tau directly:
%     channels -> P -> threshold -> cleanup
%  Nothing above the branch changes, so the two modes can be measured paired on
%  identical P. See ../docs/REDUCE.md.
%
%  Unlike RNV-Claude/main, seeds are not segmented independently and OR'd. One
%  level set evolves over the whole image and the seeds decide afterwards
%  which components survive, which is what lets a lesion split into fronds.
%
%  See also CNNGVF.INFERPROBMAP, CNNGVF.DRLSE, RUN_BATCH.

[img, chans] = cnngvf.loadChannels(src, cfg);

P = cnngvf.inferProbMap(chans, seeds, net, cfg);

% The branch sits as high as it can: everything above it - channels, patch
% extraction, the seed heat-map, inference, the raised-cosine stitching - is
% shared, and no trainable parameter is involved below it. That is what makes
% the two modes comparable on identical P.
switch cfg.Post.mode
    case "levelset"
        % Post.tau exists so it can be swept in threshold mode. If it were
        % allowed to drift from the threshold that actually seeds phi0, a sweep
        % would silently move this arm too and the pairing would be gone.
        assert(cfg.Post.tau == cfg.Drlse.tau, ...
            'cnngvf:segmentImage:tauMismatch', ...
            ['Post.tau (%g) and Drlse.tau (%g) disagree in level-set mode. ' ...
             'Drlse.tau is what seeds phi0; sweep Post.tau only in ' ...
             '"threshold" mode.'], cfg.Post.tau, cfg.Drlse.tau);

        f       = cnngvf.buildForce(P, img.I, cfg);
        [u, v]  = cnngvf.gvfField(f, cfg);
        phi0    = cnngvf.initPhi(P, seeds, cfg);
        BW      = cnngvf.drlse(P, phi0, u, v, cfg);

        if cfg.Post.imclose
            BW = imclose(BW, cfg.Post.MorphElement);
        end
        BW = imfill(BW, 'holes');  % RNV-DRAC22GT/FilledGT has its lesions filled

        [perSeed, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);

    case "threshold"
        [mask, perSeed] = cnngvf.maskFromProb(P, seeds, cfg);

    otherwise
        error('cnngvf:segmentImage:badPostMode', ...
            'Unknown Post.mode "%s". Use "levelset" or "threshold".', cfg.Post.mode);
end

contours = bwboundaries(mask, 'noholes');
end
