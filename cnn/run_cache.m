function [perImage, summary, maxDiff] = run_cache(cfg, cacheDir)
%RUN_CACHE Score cached cross-validated P through the current post-processing.
%
%  run_cache                          % the current configuration, its own cache
%  run_cache(cfg)                     % custom config
%  run_cache(cfg, cacheDir)           % another cache directory
%  [perImage, summary, maxDiff] = run_cache(...)
%
%  Reads the per-fold P that run_cv cached (cfg.CV.probCache, default
%  <Output>/cv/probcache), runs cnngvf.maskFromProb on each image, and scores
%  it. No network, no GPU: seconds. Every P is fold-checked by
%  cnngvf.probCache, so the result is out-of-sample.
%
%  This is how a change after P is verified: with cfg.CV.reference set, the
%  per-image IoU is compared with that record and the largest difference is
%  printed and returned as maxDiff. A refactor must give maxDiff ~ 1e-15; any
%  real difference is a result, to be recorded, not a refactor.
%
%  Writes <Output>/cache/per_image.csv and run_summary.txt.
%
%  See also RUN_CV, CNNGVF.PROBCACHE, CNNGVF.MASKFROMPROB.

if nargin < 1
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);
end
if nargin < 2 || strlength(cacheDir) == 0
    cacheDir = cfg.CV.probCache;
    if strlength(cacheDir) == 0
        cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
    end
end

images = cnngvf.probCache("list", cacheDir);
if isempty(images)
    error('cnngvf:runCache:empty', 'No cached P in %s. Run run_cv first.', cacheDir);
end
[names, seeds] = cnngvf.loadPoints(cfg);

fprintf('run_cache | arm %s | %d images | %s\n', cfg.Arm, numel(images), cacheDir);

n = numel(images);
results(n) = struct('image', "", 'seeds', [], 'perSeed', {{}}, 'mask', []);
elapsed = zeros(n, 1);
for i = 1:n
    t = tic;
    c = cnngvf.probCache("read", cacheDir, images(i), cfg);
    s = seeds(names == images(i), :);
    [mask, perSeed] = cnngvf.maskFromProb(double(c.P), s, cfg);
    results(i) = struct('image', images(i), 'seeds', s, 'perSeed', {perSeed}, 'mask', mask);
    elapsed(i) = toc(t);
end

[perImage, summary] = cnngvf.evaluate(results, cfg);

outDir = fullfile(cfg.Paths.Output, 'cache');
if ~exist(outDir, 'dir'), mkdir(outDir); end
writetable(perImage, fullfile(outDir, 'per_image.csv'));
cnngvf.writeSummary(fullfile(outDir, 'run_summary.txt'), cfg, perImage, summary, elapsed);

fprintf('mean IoU %.6f | median %.4f | F1 %.4f | precision %.4f | recall %.4f\n', ...
    summary.meanIou, summary.medianIou, summary.meanF1, ...
    summary.meanPrecision, summary.meanRecall);

maxDiff = NaN;
if strlength(cfg.CV.reference) > 0 && isfile(cfg.CV.reference)
    ref = readtable(cfg.CV.reference, 'TextType', 'string');
    [common, ia, ib] = intersect(perImage.Image, ref.Image);
    d = perImage.Iou(ia) - ref.Iou(ib);
    maxDiff = max(abs(d));
    fprintf('vs reference %s: %d of %d images compared, reference mean %.6f, max |dIoU| %.3g\n', ...
        cfg.CV.reference, numel(common), height(ref), mean(ref.Iou(ib)), maxDiff);
end
end
