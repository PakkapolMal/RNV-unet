function [cells, perImage] = run_ablate(arm, mode)
%RUN_ABLATE Score post-processing variants off cached P. No GPU, no retraining.
%
%  run_ablate                              % "unclustered_boundary", 2^3 cells
%  run_ablate("DRACTUH_unclustered")
%  run_ablate(arm, "tau")                  % the tau sensitivity curve instead
%  [cells, perImage] = run_ablate(...)
%
%  Reads <Output>/cv/probcache/, written by run_cv. Every cell sees byte-
%  identical P, so training variance between cells is exactly ZERO and the
%  paired tests below are unusually powerful: a small consistent delta is real.
%  Nothing here adds or removes a trainable parameter, which is why no retrain
%  is needed - and why retraining would be actively wrong, since it would
%  destroy the pairing.
%
%  The 2^3 factorial (../docs/REDUCE.md):
%
%     mode        "levelset" | "threshold"
%     diskFallback  plant a disk at a seed the threshold missed
%     imclose       imclose(disk 2) before the fill
%
%  Run together, all eight cells, NOT sequentially. They interact: imclose was
%  justified as bridging the gaps the LEVEL SET leaves, and the disk fallback as
%  giving a missed lesion something to GROW from - both claims are about the
%  level set specifically, so a sequential ablation would attribute their
%  interaction to whichever ran first.
%
%  Cell 1 (levelset / true / true) is the current pipeline and the reference
%  every delta is measured against.
%
%  Paired statistics against cell 1: exact sign test, percentile bootstrap 95%
%  CI, and Wilcoxon signed-rank. This MATLAB has no Statistics and Machine
%  Learning Toolbox, so all three are implemented below; the sign test and the
%  bootstrap are exact, but the WILCOXON IS THE NORMAL APPROXIMATION with tie
%  correction, not the exact test. That is stated wherever it is reported.
%
%  tau mode sweeps Post.tau in threshold mode and reports mean F1 AS A CURVE.
%  Quoting its maximum would be leakage - the same class of error as quoting
%  Split.Val as held-out - so the headline stays the pre-registered tau = 0.5.
%
%  Writes <Output>/ablate/{cells.csv, per_image.csv, ablate_summary.txt}, or
%  <Output>/ablate/{tau_curve.csv, tau_summary.txt}.
%
%  See also RUN_CV, CNNGVF.PROBCACHE, CNNGVF.MASKFROMPROB, CNNGVF.SEGMENTIMAGE.

if nargin < 1 || isempty(arm) || strlength(arm) == 0
    arm = "unclustered_boundary";
end
if nargin < 2 || isempty(mode)
    mode = "factorial";
end

cfg = cnngvf.arm(cnngvf.defaults(), string(arm));
cfg.EvalMode = "image";

% "blend" reads the raw image, which the cache does not carry. Every current arm
% is "probGrad", where buildForce ignores its I argument entirely.
assert(cfg.Force.source == "probGrad", 'cnngvf:runAblate:needsProbGrad', ...
    ['Force.source is "%s"; run_ablate scores from cached P alone and only ' ...
     '"probGrad" is computable that way.'], cfg.Force.source);

cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
if ~isfolder(cacheDir)
    error('cnngvf:runAblate:noCache', ...
        ['No probability cache at %s.\nRun run_cv("%s") first - with ' ...
         'cfg.CV.skipTrain = true if its fold networks already exist.'], ...
        cacheDir, arm);
end

M      = cnngvf.manifest(cfg);
images = cnngvf.probCache("list", cacheDir);
images = images(ismember(images, M.Name));
assert(~isempty(images), 'cnngvf:runAblate:emptyCache', 'The cache holds no domain image.');

[names, seedRows] = cnngvf.loadPoints(cfg);

outDir = fullfile(cfg.Paths.Output, 'ablate');
if ~exist(outDir, 'dir'), mkdir(outDir); end

fprintf('run_ablate | arm %s | %d images | %s\n', arm, numel(images), mode);
fprintf('Cache : %s\n', cacheDir);
fprintf('Output: %s\n\n', outDir);

switch string(mode)
    case "factorial"
        [cells, perImage] = runFactorial(cfg, M, images, names, seedRows, cacheDir, outDir);
    case "tau"
        [cells, perImage] = tauCurve(cfg, M, images, names, seedRows, cacheDir, outDir);
    otherwise
        error('cnngvf:runAblate:badMode', ...
            'Unknown mode "%s". Use "factorial" or "tau".', mode);
end
end


% =========================================================================
% THE FACTORIAL
% =========================================================================

function [cells, perImage] = runFactorial(cfg, M, images, names, seedRows, cacheDir, outDir)
% Cell order puts the reference first, so every delta below is against row 1.
spec = [ ...
    struct('mode', "levelset",  'disk', true,  'close', true)  ...
    struct('mode', "levelset",  'disk', true,  'close', false) ...
    struct('mode', "levelset",  'disk', false, 'close', true)  ...
    struct('mode', "levelset",  'disk', false, 'close', false) ...
    struct('mode', "threshold", 'disk', true,  'close', true)  ...
    struct('mode', "threshold", 'disk', true,  'close', false) ...
    struct('mode', "threshold", 'disk', false, 'close', true)  ...
    struct('mode', "threshold", 'disk', false, 'close', false)];
nCells = numel(spec);
nImg   = numel(images);

iou  = nan(nImg, nCells);
f1   = nan(nImg, nCells);
prec = nan(nImg, nCells);
rec  = nan(nImg, nCells);
fold = nan(nImg, 1);
dset = strings(nImg, 1);

tAll = tic;
for i = 1:nImg
    imgName = images(i);
    k       = find(M.Name == imgName, 1);
    seeds   = seedRows(names == imgName, :);

    got  = cnngvf.probCache("read", cacheDir, imgName, cfg);   % leakage-guarded
    P    = double(got.P);
    fold(i) = got.meta.fold;
    dset(i) = M.Dataset(k);
    gt   = rnv.loadMask(M.GtPath(k));

    % The GVF field depends only on P, so it is computed ONCE per image and
    % shared by every level-set cell - the expensive stage runs 1x, not 4x.
    f      = cnngvf.buildForce(P, P, cfg);
    [u, v] = cnngvf.gvfField(f, cfg);

    % Likewise the raw per-mode masks depend only on (mode, diskFallback), so
    % each is built once and the imclose variants branch off it.
    raw = containers.Map();
    for d = [true false]
        c = cfg; c.Post.diskFallback = d;
        raw(sprintf('levelset_%d', d))  = cnngvf.drlse(P, cnngvf.initPhi(P, seeds, c), u, v, c);
        raw(sprintf('threshold_%d', d)) = thresholdRaw(P, seeds, c);
    end

    for j = 1:nCells
        s = spec(j);
        c = cfg;
        c.Post.mode = s.mode; c.Post.diskFallback = s.disk; c.Post.imclose = s.close;

        BW = raw(sprintf('%s_%d', s.mode, s.disk));
        if s.close
            BW = imclose(BW, c.Post.MorphElement);
        end
        BW = imfill(BW, 'holes');
        [~, mask] = cnngvf.seedMasks(BW, seeds, c.Post.MinArea, c.Post.ClaimRule);

        m = rnv.metrics(mask, gt);
        iou(i, j) = m.iou; f1(i, j) = m.f1;
        prec(i, j) = m.precision; rec(i, j) = m.recall;
    end

    fprintf('  [%2d/%2d] %-10s fold %d | cell1 IoU %.4f | cell5 IoU %.4f\n', ...
        i, nImg, imgName, fold(i), iou(i, 1), iou(i, 5));
end

% The harness check: cells 1 and 3 differ only in a flag the level set uses at
% initialization. If they came out identical, initPhi is ignoring it.
if isequal(iou(:, 1), iou(:, 3))
    warning('cnngvf:runAblate:diskInert', ...
        'diskFallback changed nothing in level-set mode on any image.');
end

perImage = table(images, dset, fold, 'VariableNames', {'Image', 'Dataset', 'Fold'});
for j = 1:nCells
    perImage.(sprintf('Iou_cell%d', j)) = iou(:, j);
    perImage.(sprintf('F1_cell%d',  j)) = f1(:, j);
end

cells = summarise(spec, iou, f1, prec, rec);
writetable(cells,    fullfile(outDir, 'cells.csv'));
writetable(perImage, fullfile(outDir, 'per_image.csv'));
report(fullfile(outDir, 'ablate_summary.txt'), cfg, spec, cells, iou, f1, fold, dset, images);

fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('%d images, %d cells, %.1f s. Reference is cell 1 (levelset/disk/close).\n', ...
    nImg, nCells, toc(tAll));
disp(cells);
fprintf('Records: %s\n', fullfile(outDir, 'cells.csv'));
end


function BW = thresholdRaw(P, seeds, cfg)
% cnngvf.maskFromProb up to but excluding the morphology and the claiming, so
% the two imclose cells branch off one shared result. Deliberately NOT
% maskFromProb with minArea 0: that would claim components here and again after
% the morphology, and claiming twice is precisely what the reduced pipeline
% exists to stop doing. The disks come from the same cnngvf.seedDisks
% maskFromProb calls, so there is no second copy of the geometry to drift.
BW = P > cfg.Post.tau;
if cfg.Post.diskFallback
    BW = BW | cnngvf.seedDisks(size(P), seeds, BW, cfg);
end
end


% =========================================================================
% THE TAU CURVE
% =========================================================================

function [curve, perImage] = tauCurve(cfg, M, images, names, seedRows, cacheDir, outDir)
taus  = 0.30:0.025:0.80;
nImg  = numel(images);
iou   = nan(nImg, numel(taus));
f1    = nan(nImg, numel(taus));

cfg.Post.mode = "threshold";

for i = 1:nImg
    imgName = images(i);
    k       = find(M.Name == imgName, 1);
    seeds   = seedRows(names == imgName, :);
    got     = cnngvf.probCache("read", cacheDir, imgName, cfg);
    P       = double(got.P);
    gt      = rnv.loadMask(M.GtPath(k));

    for t = 1:numel(taus)
        c = cfg; c.Post.tau = taus(t);
        [mask, ~] = cnngvf.maskFromProb(P, seeds, c);
        m = rnv.metrics(mask, gt);
        iou(i, t) = m.iou; f1(i, t) = m.f1;
    end
    fprintf('  [%2d/%2d] %-10s\n', i, nImg, imgName);
end

curve = table(taus', mean(iou, 1)', median(iou, 1)', mean(f1, 1)', ...
    'VariableNames', {'Tau', 'MeanIou', 'MedianIou', 'MeanF1'});

perImage = array2table(iou, 'VariableNames', compose("Iou_tau%03d", round(taus * 1000)));
perImage = [table(images, 'VariableNames', {'Image'}), perImage];

% The ceiling, NOT quotable: picking tau per image uses the answer to choose the
% question. Reported only to bound how much a better-calibrated P could buy.
oracle = max(iou, [], 2);

writetable(curve,    fullfile(outDir, 'tau_curve.csv'));
writetable(perImage, fullfile(outDir, 'tau_per_image.csv'));

fid = fopen(fullfile(outDir, 'tau_summary.txt'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'tau sensitivity - %s, %d images\n', cfg.Arm, nImg);
fprintf(fid, '================================\n\n');
fprintf(fid, 'REPORT THIS AS A CURVE. Quoting its maximum is leakage: tau would\n');
fprintf(fid, 'have been chosen on the images being scored.\n\n');
fprintf(fid, '   tau   meanIoU  medIoU   meanF1\n');
for t = 1:numel(taus)
    star = "";
    if abs(taus(t) - 0.5) < 1e-9
        star = "  <- pre-registered, the headline";
    end
    fprintf(fid, '  %.3f   %.4f  %.4f   %.4f%s\n', taus(t), ...
        curve.MeanIou(t), curve.MedianIou(t), curve.MeanF1(t), star);
end
flat = max(curve.MeanIou(taus >= 0.40 & taus <= 0.65)) ...
     - min(curve.MeanIou(taus >= 0.40 & taus <= 0.65));
fprintf(fid, '\nSpread of mean IoU over tau 0.40-0.65: %.4f\n', flat);
fprintf(fid, ['A flat curve here means a fixed tau is safe. A sharp one is a\n' ...
              'CALIBRATION problem in P, not a threshold problem - record it and\n' ...
              'stop, do not patch it with a tuned tau.\n']);
fprintf(fid, '\nOracle per-image tau (NOT QUOTABLE, upper bound only): %.4f\n', mean(oracle));
fprintf(fid, 'Pre-registered tau = 0.5                              : %.4f\n', ...
    curve.MeanIou(abs(taus - 0.5) < 1e-9));

fprintf('\nmean IoU at the pre-registered tau = 0.5: %.4f\n', ...
    curve.MeanIou(abs(taus - 0.5) < 1e-9));
fprintf('spread over tau 0.40-0.65               : %.4f\n', flat);
fprintf('oracle per-image tau (not quotable)     : %.4f\n', mean(oracle));
fprintf('Records: %s\n', fullfile(outDir, 'tau_curve.csv'));
end


% =========================================================================
% SUMMARY AND REPORT
% =========================================================================

function T = summarise(spec, iou, f1, prec, rec)
n = numel(spec);
rows = cell(n, 1);
for j = 1:n
    d = iou(:, j) - iou(:, 1);
    [p, up, dn] = signTest(d);
    ci = bootstrapCI(d);
    rows{j} = {j, spec(j).mode, spec(j).disk, spec(j).close, ...
        mean(iou(:, j)), median(iou(:, j)), mean(f1(:, j)), ...
        mean(prec(:, j)), mean(rec(:, j)), ...
        mean(d), up, dn, p, wilcoxonP(d), ci(1), ci(2)};
end
T = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'Cell', 'Mode', 'DiskFallback', 'Imclose', 'MeanIou', 'MedianIou', ...
     'MeanF1', 'MeanPrecision', 'MeanRecall', 'DeltaIou', 'Better', 'Worse', ...
     'SignP', 'WilcoxonP', 'CiLo', 'CiHi'});
T.Mode = string(T.Mode);
end


function report(txtPath, cfg, ~, cells, iou, f1, fold, dset, images)
fid = fopen(txtPath, 'w');
if fid < 0
    warning('cnngvf:runAblate:noSummary', 'Could not write %s', txtPath);
    return;
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Post-processing ablation - %s\n', cfg.Arm);
fprintf(fid, '==========================================\n');
fprintf(fid, 'Timestamp : %s\n', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
fprintf(fid, 'Domain    : %d images, %s\n', numel(images), strjoin(cfg.ActiveDatasets, ' + '));
fprintf(fid, 'Points    : %s | claim rule: %s | tau: %g\n', ...
    cfg.Points, cfg.Post.ClaimRule, cfg.Post.tau);
fprintf(fid, 'Protocol  : %d-fold CV, every image out-of-sample, EvalMode "%s"\n', ...
    cfg.CV.k, cfg.EvalMode);
fprintf(fid, 'Source    : cached P, identical across all cells (training variance = 0)\n\n');

fprintf(fid, 'STATE THE DOMAIN WITH EVERY NUMBER. %d-image figures are not\n', numel(images));
fprintf(fid, 'comparable with the 17-, 39- or 68-image ones.\n\n');

fprintf(fid, 'Cells (reference = cell 1, the current pipeline)\n');
fprintf(fid, '-----------------------------------------------\n');
fprintf(fid, '  # mode       disk close |    IoU  median      F1    prec     rec |   dIoU  +/-   signP  wilcoxP   95%% CI\n');
for j = 1:height(cells)
    fprintf(fid, '  %d %-10s %-4d %-5d | %.4f  %.4f  %.4f  %.4f  %.4f | %+.4f %2d/%-2d %.4f  %.4f  [%+.4f,%+.4f]\n', ...
        cells.Cell(j), cells.Mode(j), cells.DiskFallback(j), cells.Imclose(j), ...
        cells.MeanIou(j), cells.MedianIou(j), cells.MeanF1(j), ...
        cells.MeanPrecision(j), cells.MeanRecall(j), cells.DeltaIou(j), ...
        cells.Better(j), cells.Worse(j), cells.SignP(j), cells.WilcoxonP(j), ...
        cells.CiLo(j), cells.CiHi(j));
end

fprintf(fid, '\nThe Wilcoxon column is the NORMAL APPROXIMATION with tie correction.\n');
fprintf(fid, 'This MATLAB has no Statistics Toolbox, so signrank is unavailable.\n');
fprintf(fid, 'The sign test and the bootstrap CI are exact.\n');

fprintf(fid, '\nPer fold, mean IoU\n------------------\n');
fprintf(fid, '  fold   n');
for j = 1:height(cells), fprintf(fid, '   cell%d', j); end
fprintf(fid, '\n');
for f = unique(fold)'
    sel = fold == f;
    fprintf(fid, '  %4d %3d', f, nnz(sel));
    for j = 1:height(cells), fprintf(fid, '  %.4f', mean(iou(sel, j))); end
    fprintf(fid, '\n');
end

if numel(unique(dset)) > 1
    fprintf(fid, '\nPer dataset, mean IoU (a breakout, not a protocol)\n');
    fprintf(fid, '--------------------------------------------------\n');
    for d = unique(dset)'
        sel = dset == d;
        fprintf(fid, '  %-8s n=%-3d', d, nnz(sel));
        for j = 1:height(cells), fprintf(fid, '  %.4f', mean(iou(sel, j))); end
        fprintf(fid, '\n');
    end
end

fprintf(fid, '\nBiggest per-image movers, cell 5 (reduced) vs cell 1\n');
fprintf(fid, '---------------------------------------------------\n');
d = iou(:, 5) - iou(:, 1);
[~, ord] = sort(d);
show = [ord(1:min(5, numel(ord)))', flip(ord(max(1, end - 4):end))'];
for i = unique(show, 'stable')
    fprintf(fid, '  %-10s  %.4f -> %.4f  (%+.4f)\n', ...
        images(i), iou(i, 1), iou(i, 5), d(i));
end

fprintf(fid, '\nMean F1 per cell: ');
fprintf(fid, '%.4f ', mean(f1, 1));
fprintf(fid, '\n');
end


% =========================================================================
% PAIRED STATISTICS
% =========================================================================
% No Statistics and Machine Learning Toolbox on this machine, so signrank,
% bootci, binocdf, prctile and tiedrank are all unavailable. These are the
% three tests REDUCE.md asks for, written out.

function [p, nPos, nNeg] = signTest(d)
%SIGNTEST Exact two-sided sign test on the non-zero differences.
tol  = 1e-12;
nPos = nnz(d >  tol);
nNeg = nnz(d < -tol);
n    = nPos + nNeg;
if n == 0
    p = 1;
    return;
end
k = min(nPos, nNeg);
% Exact binomial tail at 0.5, doubled. nchoosek is base MATLAB.
p = 0;
for i = 0:k
    p = p + nchoosek(n, i);
end
p = min(1, 2 * p / 2^n);
end


function ci = bootstrapCI(d, B, alpha)
%BOOTSTRAPCI Percentile bootstrap CI for the mean paired difference.
if nargin < 2, B = 10000; end
if nargin < 3, alpha = 0.05; end
n = numel(d);
if n == 0 || all(abs(d) < 1e-12)
    ci = [0 0];
    return;
end
rng(0);   % deterministic: the same table twice must give the same interval
idx  = randi(n, n, B);
means = mean(d(idx), 1);
ci = [pct(means, alpha / 2), pct(means, 1 - alpha / 2)];
end


function q = pct(x, p)
%PCT Percentile by linear interpolation. prctile is Statistics Toolbox.
x = sort(x(:));
n = numel(x);
if n == 1
    q = x;
    return;
end
h = (n - 1) * p + 1;
lo = floor(h); hi = ceil(h);
q = x(lo) + (h - lo) * (x(hi) - x(lo));
end


function p = wilcoxonP(d)
%WILCOXONP Wilcoxon signed-rank, NORMAL APPROXIMATION with tie correction.
%  Not the exact test - signrank is Statistics Toolbox. At n = 39-68 the
%  approximation is close, but it is an approximation and is labelled as one
%  everywhere it is reported.
d = d(abs(d) > 1e-12);
n = numel(d);
if n < 2
    p = 1;
    return;
end

r = avgRank(abs(d));
W = sum(r(d > 0));

mu    = n * (n + 1) / 4;
[~, ~, g] = unique(abs(d));
tieCorr = sum(arrayfun(@(k) sum(g == k)^3 - sum(g == k), unique(g))) / 48;
sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24 - tieCorr);
if sigma == 0
    p = 1;
    return;
end

z = (abs(W - mu) - 0.5) / sigma;   % continuity correction
p = erfc(z / sqrt(2));             % two-sided normal tail, no toolbox needed
p = min(1, max(0, p));
end


function r = avgRank(x)
%AVGRANK Ranks with ties averaged. tiedrank is Statistics Toolbox.
[s, ord] = sort(x(:));
r = zeros(size(s));
i = 1;
while i <= numel(s)
    j = i;
    while j < numel(s) && s(j + 1) == s(i)
        j = j + 1;
    end
    r(i:j) = (i + j) / 2;
    i = j + 1;
end
out = zeros(size(s));
out(ord) = r;
r = reshape(out, size(x));
end
