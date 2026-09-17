function [P, V] = inferProbMap(chans, seeds, net, cfg)
%CNNGVF.INFERPROBMAP Predict the full-image RNV probability map from seeds.
%
%  P = cnngvf.inferProbMap(chans, seeds, net, cfg)
%  [P, V] = cnngvf.inferProbMap(chans, seeds, net, cfg)
%
%  chans : h x w x 2 stack from cnngvf.loadChannels
%  seeds : S x 2 [cx cy] localization points for this image
%  net   : dlnetwork from cnngvf.trainNet
%
%  P : h x w double in [0,1]. Seeds are the only patch centres, so regions no
%      seed reaches stay 0 - the seed-conditioned design of ../docs/spec.md.
%  V : h x w double, the across-transform variance of the foreground
%      probability, stitched like P. [] when cfg.Infer.tta is "none".
%
%  cfg.Infer.tta (../docs/REFINE.md step 6)
%    "none"      one prediction per patch. The original, and the default.
%    "d4"        the 8 dihedral transforms (cnngvf.d4): both channels are
%                transformed together, predicted, the foreground probability
%                inverse-transformed, and the 8 PROBABILITIES averaged per patch
%                before the usual raised-cosine stitch.
%    "identity"  the TTA code path with a 1-element group. Exists so a test can
%                show the averaging adds nothing on its own.
%
%  All of one image's patches go through predict as a single batch, and the
%  patches are blended back with a raised-cosine window so overlapping seeds
%  (458.png has 7) do not leave seams at the patch borders.
%
%  See also CNNGVF.SEEDPATCH, CNNGVF.SEGMENTIMAGE.

p = cfg.Patch.size;
n = size(seeds, 1);
[h, w, ~] = size(chans);

P = zeros(h, w);
V = [];
if n == 0
    return;
end

% C+1: the channel stack plus the seed heat-map seedPatch appends. Derived,
% not assumed, so dropping the vesselness channel needs no change here.
nIn  = size(chans, 3) + 1;
X    = zeros(p, p, nIn, n, 'single');
rows = zeros(n, p);
cols = zeros(n, p);

% A network is tied to the channel count it trained with. Catch the mismatch
% here with a readable message instead of letting predict fail obscurely.
expected = net.Layers(1).InputSize(3);
if expected ~= nIn
    error('cnngvf:inferProbMap:channelMismatch', ...
        ['The network expects %d input channels but this cfg builds %d.\n' ...
         'cfg.Vessel.enabled is %d - it must match the setting the network ' ...
         'was trained with.'], expected, nIn, cfg.Vessel.enabled);
end
for s = 1:n
    [X(:, :, :, s), rows(s, :), cols(s, :)] = ...
        cnngvf.seedPatch(chans, seeds(s, :), seeds(s, :), cfg);
end

tta = "none";
if isfield(cfg, 'Infer') && isfield(cfg.Infer, 'tta')
    tta = string(cfg.Infer.tta);
end
switch tta
    case "none"
        Pn = predictPatches(net, X, cfg);
        Vn = [];
    case {"d4", "identity"}
        group = 0:7;
        if tta == "identity", group = 0; end
        acc  = zeros(p, p, n, 'single');
        acc2 = zeros(p, p, n, 'single');
        for t = group
            Yt = cnngvf.d4(predictPatches(net, cnngvf.d4(X, t), cfg), t, true);
            acc  = acc + Yt;
            acc2 = acc2 + Yt.^2;
        end
        m  = numel(group);
        Pn = acc / m;
        Vn = max(acc2 / m - Pn.^2, 0);
    otherwise
        error('cnngvf:inferProbMap:badTta', 'Unknown Infer.tta "%s". Use "none", "d4" or "identity".', tta);
end

% Raised cosine, floored so that a pixel covered by exactly one patch still
% gets that patch's value at the very patch border instead of 0/0.
hann = 0.5 - 0.5 * cos(2 * pi * (0:p-1)' / (p - 1));
win  = max(hann * hann', 1e-3);

W = zeros(h, w);
if ~isempty(Vn), V = zeros(h, w); end
for s = 1:n
    inR = rows(s, :) >= 1 & rows(s, :) <= h;
    inC = cols(s, :) >= 1 & cols(s, :) <= w;
    r   = rows(s, inR);
    c   = cols(s, inC);

    P(r, c) = P(r, c) + double(Pn(inR, inC, s)) .* win(inR, inC);
    W(r, c) = W(r, c) + win(inR, inC);
    if ~isempty(Vn)
        V(r, c) = V(r, c) + double(Vn(inR, inC, s)) .* win(inR, inC);
    end
end

P = P ./ max(W, eps);
if ~isempty(Vn)
    V = V ./ max(W, eps);
end
end


function Pn = predictPatches(net, X, cfg)
% Predict in chunks. One batch per image was fine for the clustered centres
% (7 seeds at most) but the unclustered points put 62 on 1115.png, and 62
% 512x512 patches at once exhausts an 8 GB card.
p = size(X, 1);
n = size(X, 4);
Pn = zeros(p, p, n, 'single');
step = max(1, cfg.Net.inferBatch);
for a = 1:step:n
    b  = min(a + step - 1, n);
    Xb = X(:, :, :, a:b);
    if cfg.UseGpu
        Xb = gpuArray(Xb);
    end
    Y = gather(predict(net, Xb));
    Pn(:, :, a:b) = Y(:, :, 2, :);   % channel 2 = RNV (see prepareData)
end
end
