function f = buildForce(P, I, cfg)
%CNNGVF.BUILDFORCE Turn the CNN probability map into the GVF energy field.
%
%  f = cnngvf.buildForce(P, I, cfg)
%
%  P : h x w probability map from cnngvf.inferProbMap
%  I : h x w grayscale image (only read by the "blend" source)
%
%  f : h x w in [0,1]. Xu & Prince pre-normalise every energy field to [0,1]
%      and the GVF diffusion assumes it, so that is the last step here.
%
%  cfg.Force.source
%    "probGrad" f = |grad P|. Classic edge-map behaviour on the CNN boundary:
%               the GVF of a ridge points at that ridge from both sides, which
%               is what makes the level set settle on it.
%    "blend"    w*|grad P| + (1-w)*|grad I|. Degrades gracefully towards the
%               raw image where the CNN is uncertain.
%
%  ../docs/spec.md also lists f = P itself. That is deliberately not offered:
%  the GVF of P points at lesion *cores*, not at lesion *boundaries*, so under
%  the advection sign in cnngvf.drlse it collapses the front to a point instead
%  of stopping it on the edge. Both sources here are boundary maps, so the two
%  terms of "blend" pull the same way and w means what it looks like.
%
%  See also CNNGVF.GVFFIELD, CNNGVF.DRLSE, CNNGVF.SEGMENTIMAGE.

switch cfg.Force.source
    case "probGrad"
        f = gradMag(P, cfg.Force.sigma);

    case "blend"
        f = cfg.Force.w       * normalize01(gradMag(P, cfg.Force.sigma)) ...
          + (1 - cfg.Force.w) * normalize01(gradMag(I, cfg.Force.sigma));

    otherwise
        error('cnngvf:buildForce:badSource', ...
            'Unknown Force.source "%s". Use "probGrad" or "blend".', ...
            cfg.Force.source);
end

f = normalize01(f);
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function g = gradMag(A, sigma)
[Ax, Ay] = gradient(imgaussfilt(A, sigma));
g = hypot(Ax, Ay);
end


function A = normalize01(A)
% mat2gray, but defined for the constant field that an all-zero P produces
% (an image whose seeds all landed on background).
lo = min(A(:));
hi = max(A(:));
if hi > lo
    A = (A - lo) ./ (hi - lo);
else
    A = zeros(size(A));
end
end
