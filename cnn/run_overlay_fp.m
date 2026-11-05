function out = run_overlay_fp(cacheDir, outDir)
%RUN_OVERLAY_FP Are TUH false positives enriched on the acquisition crosshair?
%
%  out = run_overlay_fp                      % the reference cache
%  out = run_overlay_fp(cacheDir, outDir)
%
%  Scores each TUH image's cached out-of-fold P through cnngvf.maskFromProb, as
%  run_cache does, and compares the false-positive rate in a fixed band around
%  the crosshair with the rest of the seed-covered region, against a placebo
%  band 64 px away. Diagnostic only. Pre-registration: docs/briefs/overlay-fp.md.

cfg = cnngvf.defaults();
if nargin < 1 || strlength(cacheDir) == 0
    cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
end
if nargin < 2
    outDir = fullfile(cfg.Paths.Output, 'overlay_fp');
end

M = cnngvf.manifest(cfg);
M = M(M.Dataset == "TUH", :);
[names, seeds] = cnngvf.loadPoints(cfg);
ref = readtable(cfg.CV.reference, 'TextType', 'string');

% Crosshair at rows/cols 510-513 in every TUH image (brief, "The band").
bands = struct('name', {"band", "wide", "placebo"}, 'idx', {507:516, 502:521, 571:580});
n = height(M);
p = cfg.Patch.size;
half = floor(p / 2);

cnt = zeros(n, numel(bands), 4);   % FP in band, non-GT in band, FP outside, non-GT outside
iou = zeros(n, 3);                 % as scored, oracle (in-band FP removed), GT-free band removal
for i = 1:n
    c  = cnngvf.probCache("read", cacheDir, M.Name(i), cfg);
    s  = seeds(names == M.Name(i), :);
    mask = cnngvf.maskFromProb(double(c.P), s, cfg);
    gt = cnngvf.loadMask(M.GtPath(i));
    [h, w] = size(gt);

    C = false(h, w);                       % union of the seed patches: where P exists
    for k = 1:size(s, 1)
        r0 = round(s(k, 2)) - half;
        c0 = round(s(k, 1)) - half;
        C(max(r0, 1):min(r0 + p - 1, h), max(c0, 1):min(c0 + p - 1, w)) = true;
    end
    N  = C & ~gt;
    FP = mask & N;
    for b = 1:numel(bands)
        B = false(h, w);
        B(bands(b).idx, :) = true;
        B(:, bands(b).idx) = true;
        cnt(i, b, :) = [nnz(FP & B), nnz(N & B), nnz(FP & ~B), nnz(N & ~B)];
        if b == 1
            Bmain = B;
        end
    end

    iou(i, 1) = cnngvf.metrics(mask, gt).iou;
    iou(i, 2) = cnngvf.metrics(mask & ~(Bmain & ~gt), gt).iou;
    iou(i, 3) = cnngvf.metrics(mask & ~Bmain, gt).iou;
end

[~, r] = ismember(M.Name, ref.Image);
sanity = max(abs(iou(:, 1) - ref.Iou(r)));

ratio = @(k, b) (sum(k(:, b, 1)) / sum(k(:, b, 2))) / (sum(k(:, b, 3)) / sum(k(:, b, 4)));
est = struct('band', ratio(cnt, 1), 'wide', ratio(cnt, 2), 'placebo', ratio(cnt, 3));
est.diff = est.band - est.placebo;
est.fpShareInBand = sum(cnt(:, 1, 1)) / (sum(cnt(:, 1, 1)) + sum(cnt(:, 1, 3)));
est.oracle = mean(iou(:, 2)) - mean(iou(:, 1));
est.gtFree = mean(iou(:, 3)) - mean(iou(:, 1));

old = rng(20260917);
restore = onCleanup(@() rng(old));
B = 10000;
boot = zeros(B, 5);
for t = 1:B
    j = randi(n, n, 1);
    k = cnt(j, :, :);
    boot(t, :) = [ratio(k, 1), ratio(k, 2), ratio(k, 3), ratio(k, 1) - ratio(k, 3), ...
        mean(iou(j, 2)) - mean(iou(j, 1))];
end
ci = prctile(boot, [2.5 97.5]);

perImage = table(M.Name, iou(:, 1), iou(:, 2), iou(:, 3), cnt(:, 1, 1), cnt(:, 1, 2), ...
    cnt(:, 1, 3), cnt(:, 1, 4), cnt(:, 3, 1), cnt(:, 3, 2), ...
    'VariableNames', {'Image', 'Iou', 'IouOracle', 'IouGtFree', 'FpBand', 'NBand', ...
    'FpOut', 'NOut', 'FpPlacebo', 'NPlacebo'});
if ~exist(outDir, 'dir'), mkdir(outDir); end
writetable(perImage, fullfile(outDir, 'per_image.csv'));

out = struct('est', est, 'ci', ci, 'sanity', sanity, 'perImage', perImage, ...
    'tuhIou', mean(iou(:, 1)));
fprintf('\n=== overlay FP | %s | %d TUH images ===\n', cacheDir, n);
fprintf('sanity: max |IoU - reference| %.3g\n', sanity);
fprintf('TUH mean IoU %.4f\n', mean(iou(:, 1)));
fprintf('FP rate: band %.4f | outside %.4f\n', sum(cnt(:, 1, 1)) / sum(cnt(:, 1, 2)), ...
    sum(cnt(:, 1, 3)) / sum(cnt(:, 1, 4)));
fprintf('enrichment band   %.3f  CI [%.3f, %.3f]\n', est.band, ci(:, 1));
fprintf('enrichment wide   %.3f  CI [%.3f, %.3f]\n', est.wide, ci(:, 2));
fprintf('enrichment placebo %.3f CI [%.3f, %.3f]\n', est.placebo, ci(:, 3));
fprintf('band - placebo    %+.3f  CI [%+.3f, %+.3f]\n', est.diff, ci(:, 4));
fprintf('share of covered FP pixels in band %.4f\n', est.fpShareInBand);
fprintf('oracle ceiling    %+.4f  CI [%+.4f, %+.4f]\n', est.oracle, ci(:, 5));
fprintf('GT-free band removal %+.4f\n', est.gtFree);
end
