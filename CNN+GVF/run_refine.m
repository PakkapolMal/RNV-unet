function run_refine(step, outputDir)
%RUN_REFINE The staged inference-only refinement plan, ../docs/REFINE.md.
%
%  run_refine(0)                 % reads result/DRACTUH_unclustered_threshold/cv/probcache
%  run_refine(0, outputDir)      % an arm output folder somewhere else
%
%  One runner for every CPU step, instead of one script per step: they all load
%  the same 68 cached P, ground-truth masks and seeds, and differ only in what
%  they score. Writes <outputDir>/refine/step<N>/.
%
%  Steps
%    0  lock the reference: reproduce relax 0.20, the cross-fit and the level
%       set; freeze the size quintiles; per-seed trace; REF0 record
%    1  cap the claim distance, cfg.Post.claimMaxDist
%
%  Ground truth is read HERE, never by the production functions it calls
%  (cnngvf.maskFromProb and everything under it take P, seeds and cfg only).
%
%  See also CNNGVF.CROSSFIT, CNNGVF.REFINESCORE, CNNGVF.PAIREDSTATS, RUN_CONFIRM_RELAX.

cfg = cnngvf.arm(cnngvf.defaults(), "DRACTUH_unclustered");
if nargin < 2 || strlength(outputDir) == 0
    outputDir = fullfile(cfg.Paths.Main, 'result', 'DRACTUH_unclustered_threshold');
end
cfg.Paths.Output = char(outputDir);
cfg.EvalMode = "image";
cfg.Post.mode = "threshold";

refineDir = fullfile(cfg.Paths.Output, 'refine');
outDir = fullfile(refineDir, sprintf('step%d', step));
if ~exist(outDir, 'dir'), mkdir(outDir); end

D = loadDomain(cfg);
Q = quintiles(D, refineDir);

switch step
    case 0
        step0(cfg, D, Q, outDir);
    case 1
        step1(cfg, D, Q, outDir, refineDir);
    otherwise
        error('run_refine:badStep', 'Step %d is not implemented yet.', step);
end
end


% =========================================================================
% STEP 0
% =========================================================================

function step0(cfg, D, Q, outDir)
fid = fopen(fullfile(outDir, 'step0_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
t0 = tic;
say('Step 0 - lock the reference | %s | %d images | tau %.2f | %s\n', cfg.Arm, D.n, cfg.Post.tau, cfg.Paths.Output);
say('Localization CSV mtimes (cache metadata): %s\n\n', D.csvStamp);

% ---- the arm's configuration ---------------------------------------------
assert(cfg.Post.fallbackMode == "relax" && cfg.Post.relaxFloor == 0.20, ...
    'the arm does not set relax 0.20');
d = cnngvf.defaults();
assert(d.Post.fallbackMode == "disk", 'defaults.m must keep the disk');

% ---- A. REF0: relax 0.20 through the production path, with the trace -----
[REF0, trace] = scoreAll(cfg, D, true);
recorded = readtable(fullfile(cfg.Paths.Output, 'confirm', 'confirm_per_image.csv'), 'TextType', 'string');
[ok, loc] = ismember(D.images, recorded.Image);
assert(all(ok), 'confirm_per_image.csv does not cover every image');
dA = max(abs(REF0.Iou - recorded.Relax_020(loc)));
r = cnngvf.refineScore(REF0, Q);
say('A. relax 0.20: IoU %.4f  median %.4f  F1 %.4f  prec %.4f  rec %.4f | max |dIoU| vs confirm %.2g\n', ...
    r.MeanIou, r.MedianIou, r.MeanF1, r.MeanPrec, r.MeanRec, dA);
okA = dA < 1e-12 && isequal(round([r.MeanIou r.MedianIou r.MeanF1 r.MeanPrec r.MeanRec], 4), ...
    [0.5097 0.5299 0.6489 0.6311 0.7579]);

% ---- B. the cross-fit through cnngvf.crossFit ----------------------------
floors = [0.05 0.10 0.15 0.20 0.25 0.30]';
grid = table(floors, 'VariableNames', {'relaxFloor'});
fn = @(g) scoreAll(setField(cfg, 'relaxFloor', g.relaxFloor), D, false);
CF = cnngvf.crossFit(grid, fn, D.fold, D.dset, 0);
dB = max(abs(CF.perImage.Iou - recorded.CrossFitted(loc)));
say('B. cross-fitted floor (plain max): IoU %.4f | max |dIoU| vs confirm %.2g | chose %s\n', ...
    mean(CF.perImage.Iou), dB, mat2str(grid.relaxFloor(CF.chosen.GridRow)'));
okB = dB < 1e-12 && round(mean(CF.perImage.Iou), 4) == 0.5061;
% The same grid under REFINE.md's 0.001 tie-break, lowest floor first. For
% information only: the tie-break is a new rule, and 0.5061 was a plain maximum.
CT = cnngvf.crossFit(grid, fn, D.fold, D.dset, 0.001);
say('   with the 0.001 tie-break (lowest floor first): IoU %.4f | chose %s\n', ...
    mean(CT.perImage.Iou), mat2str(grid.relaxFloor(CT.chosen.GridRow)'));
for k = 1:height(CF.crossDevice)
    say('   cross-device: choose on %s -> %.2f, score %s %.4f\n', CF.crossDevice.Train(k), ...
        grid.relaxFloor(CF.crossDevice.GridRow(k)), CF.crossDevice.Test(k), CF.crossDevice.TestIou(k));
end

% ---- C. level set untouched ----------------------------------------------
say('C. level set (arm cfg, relax set; initPhi must still plant disks) ...\n');
ls = readtable(fullfile(cfg.Paths.Output, 'ablate', 'per_image.csv'), 'TextType', 'string');
[~, lloc] = ismember(D.images, ls.Image);
lsIou = zeros(D.n, 1);
for i = 1:D.n
    P = double(D.P{i});
    c = cfg; c.Post.mode = "levelset";
    f = cnngvf.buildForce(P, P, c);                 % probGrad ignores the image
    [u, v] = cnngvf.gvfField(f, c);
    BW = cnngvf.drlse(P, cnngvf.initPhi(P, D.seeds{i}, c), u, v, c);
    BW = imfill(imclose(BW, c.Post.MorphElement), 'holes');
    [~, mask] = cnngvf.seedMasks(BW, D.seeds{i}, c.Post.MinArea, c.Post.ClaimRule);
    m = rnv.metrics(mask, D.gt{i});
    lsIou(i) = m.iou;
    fprintf('  levelset [%2d/%2d] %-12s %.4f (recorded %.4f)\n', i, D.n, D.images(i), lsIou(i), ls.Iou_cell1(lloc(i)));
end
dC = max(abs(lsIou - ls.Iou_cell1(lloc)));
say('   level set IoU %.4f | max |dIoU| vs ablate cell 1 %.2g\n', mean(lsIou), dC);
okC = dC < 1e-12 && round(mean(lsIou), 4) == 0.4708;

say('\nVerify: A %s | B %s | C %s\n\n', pf(okA), pf(okB), pf(okC));

% ---- REF0 record ---------------------------------------------------------
say('REF0 (68 images, relax 0.20 fixed)\n');
say('  pooled IoU %.4f  F1 %.4f  prec %.4f  rec %.4f\n', r.PooledIou, r.PooledF1, r.PooledPrec, r.PooledRec);
say('  datasets: DRAC22 %.4f | TUH %.4f\n', r.Iou_DRAC22, r.Iou_TUH);
say('  folds:%s\n', sprintf(' %.4f', r{1, startsWith(r.Properties.VariableNames, 'Iou_fold')}));
say('  quintile  n   GtPx range           IoU    Pred/GT\n');
for k = 1:5
    s = Q.Q == k;
    say('  Q%d       %2d  %7d - %-7d    %.4f  %.3f\n', k, nnz(s), min(Q.GtPx(s)), max(Q.GtPx(s)), ...
        r.("Iou_Q" + k), r.("PredGt_Q" + k));
end
nf = ~trace.fired;
say('\nTrace (%d seeds)\n', height(trace));
say('  fired %d | relaxed %d | disk %d | claimed nothing %d\n', nnz(trace.fired), ...
    nnz(~isnan(trace.relaxThrUsed)), nnz(nf & isnan(trace.relaxThrUsed)), nnz(trace.claimedCompId == 0));
say('  claimDist > 0: %d | 0 < claimDist <= 20: %d | 20 < claimDist < Inf: %d on %d images\n', ...
    nnz(trace.claimDist > 0 & isfinite(trace.claimDist)), nnz(trace.claimDist > 0 & trace.claimDist <= 20), ...
    nnz(trace.claimDist > 20 & isfinite(trace.claimDist)), ...
    numel(unique(trace.image(trace.claimDist > 20 & isfinite(trace.claimDist)))));
say('  images with a fallback seed: %d\n', numel(unique(trace.image(nf))));
say('\n%.0f s\n', toc(t0));
fclose(fid);

writetable([table("REF0", 'VariableNames', {'Arm'}), r], fullfile(outDir, 'summary.csv'));
writetable(REF0, fullfile(outDir, 'per_image_REF0.csv'));
writetable(CF.perImage, fullfile(outDir, 'per_image_crossfit.csv'));
writetable(CF.surface, fullfile(outDir, 'surface_relaxFloor.csv'));
writetable(CF.chosen, fullfile(outDir, 'chosen_relaxFloor.csv'));
writetable(trace, fullfile(outDir, 'trace.csv'));
writetable(table(D.images, lsIou, 'VariableNames', {'Image', 'LevelsetIou'}), fullfile(outDir, 'per_image_levelset.csv'));
assert(okA && okB && okC, 'run_refine:verify', 'Step 0 verification failed - see step0_summary.txt');
end


% =========================================================================
% STEP 1
% =========================================================================

function step1(cfg, D, Q, outDir, refineDir)
fid = fopen(fullfile(outDir, 'step1_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
t0 = tic;
say('Step 1 - cap the claim distance | %s | %d images | %s\n\n', cfg.Arm, D.n, cfg.Paths.Output);

% CUR = REF0 until something is adopted. Flag off must reproduce step 0 exactly.
assert(isinf(cfg.Post.claimMaxDist), 'claimMaxDist default is not Inf');
[CUR, trace] = scoreAll(cfg, D, true);
ref0 = readtable(fullfile(refineDir, 'step0', 'per_image_REF0.csv'), 'TextType', 'string');
dRef = max(abs(CUR.Iou - ref0.Iou));
say('Flag off (claimMaxDist Inf) vs step-0 REF0: max |dIoU| %.2g\n', dRef);
assert(dRef < 1e-12 && isequal(CUR.Image, ref0.Image), 'run_refine:flagOff', ...
    'claimMaxDist = Inf does not reproduce REF0');

far = trace.claimDist > 20 & isfinite(trace.claimDist);
say('Seeds with 20 < claimDist < Inf: %d on %d images\n\n', nnz(far), numel(unique(trace.image(far))));
if ~any(far)
    say('SKIP: mechanism absent.\n');
    fclose(fid);
    return;
end

% Most conservative (feature off) first, for the tie-break.
grid = table([Inf 128 64 32 20 10 0]', 'VariableNames', {'claimMaxDist'});
fn = @(g) scoreAll(setField(cfg, 'claimMaxDist', g.claimMaxDist), D, false);
CF = cnngvf.crossFit(grid, fn, D.fold, D.dset);
pre = CF.rows{grid.claimMaxDist == 20};
XF = CF.perImage;

arms = {"CUR (Inf)", CUR; "pre-registered 20", pre; "cross-fitted", XF};
[summary, stats] = reportArms(say, arms, CUR, Q);
say('\nCross-fitted chosen per fold: %s\n', mat2str(grid.claimMaxDist(CF.chosen.GridRow)'));
cdev = crossDeviceDeltas(say, CF, grid, 'claimMaxDist', CUR, D);

say('\nSurface (full domain, NOT a result)\n');
for g = 1:height(CF.surface)
    say('  claimMaxDist %-4g  IoU %.4f  prec %.4f  rec %.4f  pooled IoU %.4f\n', CF.surface.claimMaxDist(g), ...
        CF.surface.MeanIou(g), CF.surface.MeanPrec(g), CF.surface.MeanRec(g), CF.surface.PooledIou(g));
end

G = gate(say, XF, CUR, cdev);
if G.decision == "ADOPT" && ~all(grid.claimMaxDist(CF.chosen.GridRow) == 20) ...
        && mean(pre.Iou) >= mean(XF.Iou) - 0.002
    say('Pre-registered 20 is within 0.002 of the cross-fitted IoU: adopt 20.\n');
end

% ---- diagnostic, GT: which of the disconnected seeds are stray ------------
diagDir = fullfile(refineDir, 'diag');
if ~exist(diagDir, 'dir'), mkdir(diagDir); end
gtDist = nan(height(trace), 1);
for i = 1:D.n
    s = trace.image == D.images(i);
    dist = bwdist(D.gt{i});
    [h, w] = size(dist);
    r = min(max(round(trace.cy(s)), 1), h);
    c = min(max(round(trace.cx(s)), 1), w);
    gtDist(s) = dist(sub2ind([h w], r, c));
end
tol = cfg.Patch.seedTolerance;
cut = trace(far, {'image', 'seedIdx', 'cx', 'cy', 'claimDist', 'claimedCompArea'});
cut.gtDist = gtDist(far);
cut.stray  = cut.gtDist > tol;
writetable(cut, fullfile(diagDir, 'step1_disconnected.csv'));
say('\nDiagnostic (GT, diag/ only): seeds each cap disconnects; stray = more than %d px from FilledGT\n', tol);
for cap = [128 64 32 20 10 0]
    s = trace.claimDist > cap & isfinite(trace.claimDist);
    say('  cap %-4g disconnects %d seeds: %d stray, %d true\n', cap, nnz(s), ...
        nnz(s & gtDist > tol), nnz(s & gtDist <= tol));
end
for k = 1:height(cut)
    say('    %-12s seed %2d  claimDist %6.1f  gtDist %6.1f  stray %d\n', cut.image(k), cut.seedIdx(k), ...
        cut.claimDist(k), cut.gtDist(k), cut.stray(k));
end

say('\n%.0f s\n', toc(t0));
fclose(fid);
writetable(summary, fullfile(outDir, 'summary.csv'));
writetable(stats, fullfile(outDir, 'stats.csv'));
writetable(CUR, fullfile(outDir, 'per_image_CUR.csv'));
writetable(pre, fullfile(outDir, 'per_image_pre20.csv'));
writetable(XF, fullfile(outDir, 'per_image_crossfit.csv'));
writetable(CF.surface, fullfile(outDir, 'surface_claimMaxDist.csv'));
writetable(CF.chosen, fullfile(outDir, 'chosen_claimMaxDist.csv'));
writetable(cdev, fullfile(outDir, 'crossdevice.csv'));
end


% =========================================================================
% REPORTING AND THE ADOPTION RULE
% =========================================================================

function [summary, stats] = reportArms(say, arms, C, Q)
% One line per arm: aggregates and paired stats vs the comparator C, then the
% dataset and size-quintile breakouts, then the last arm's movers.
summary = table(); stats = table();
say('  %-20s    IoU  median      F1    prec     rec   pIoU   pRec |    dIoU  +/-     sign p  Wilc p  95%% CI\n', 'arm');
for a = 1:size(arms, 1)
    T = arms{a, 2};
    assert(isequal(T.Image, C.Image), 'arm %s is not paired with the comparator', arms{a, 1});
    r = cnngvf.refineScore(T, Q);
    d = T.Iou - C.Iou;
    S = cnngvf.pairedStats(d, T.Image);
    say('  %-20s %.4f  %.4f  %.4f  %.4f  %.4f %.4f %.4f | %+.4f %2d/%-2d  %.4f  %.4f [%+.4f, %+.4f]\n', ...
        arms{a, 1}, r.MeanIou, r.MedianIou, r.MeanF1, r.MeanPrec, r.MeanRec, r.PooledIou, r.PooledRec, ...
        S.mean, S.up, S.down, S.signP, S.wilcoxonP, S.ci(1), S.ci(2));
    if abs(S.mean) > 0.05
        say('  !! %s moves mean IoU by %+.4f - more likely a bug than a finding. STOP.\n', arms{a, 1}, S.mean);
    end
    summary = [summary; [table(arms{a, 1}, 'VariableNames', {'Arm'}), r]]; %#ok<AGROW>
    stats = [stats; table(arms{a, 1}, S.mean, S.up, S.down, S.signP, S.wilcoxonP, S.ci(1), S.ci(2), ...
        min(d), 'VariableNames', {'Arm', 'dIoU', 'Better', 'Worse', 'SignP', 'WilcoxonP', ...
        'CiLo', 'CiHi', 'WorstImageDelta'})]; %#ok<AGROW>
end
say('\n  %-20s  DRAC22  TUH    | Q1..Q5 as IoU / Pred-GT\n', 'arm');
for a = 1:height(summary)
    r = summary(a, :);
    q = sprintf('%.4f/%.2f  ', [r{1, "Iou_Q" + (1:5)}; r{1, "PredGt_Q" + (1:5)}]);
    say('  %-20s  %.4f  %.4f | %s\n', r.Arm, r.Iou_DRAC22, r.Iou_TUH, q);
end
T = arms{end, 2};
S = cnngvf.pairedStats(T.Iou - C.Iou, T.Image);
say('\n  Movers, %s: better %s | worse %s\n', arms{end, 1}, moverText(S.gains), moverText(S.losses));
end


function cdev = crossDeviceDeltas(say, CF, grid, param, C, D)
cdev = CF.crossDevice;
cdev.Value = grid.(param)(cdev.GridRow);
cdev.dIoU = zeros(height(cdev), 1);
for k = 1:height(cdev)
    te = D.dset == cdev.Test(k);
    cdev.dIoU(k) = mean(CF.rows{cdev.GridRow(k)}.Iou(te) - C.Iou(te));
    say('Cross-device: choose on %s -> %s = %g, score %s: %.4f (%+.4f vs CUR)\n', cdev.Train(k), param, ...
        cdev.Value(k), cdev.Test(k), cdev.TestIou(k), cdev.dIoU(k));
end
end


function G = gate(say, T, C, cdev)
% ../docs/REFINE.md, "The adoption rule", on the cross-fitted arm vs CUR.
d = T.Iou - C.Iou;
S = cnngvf.pairedStats(d, T.Image);
pT = cnngvf.refineScore(T).PooledIou;
pC = cnngvf.refineScore(C).PooledIou;
G.dIoU = S.mean;
G.r1 = S.mean >= 0.003;
G.r2 = S.ci(1) > -0.005;
G.r3 = pT >= pC - 1e-12;
G.r4 = min(d) >= -0.10;
G.r5 = all(cdev.dIoU >= -1e-12);
say('\nGate: rule 1 %s (dIoU %+.4f >= +0.003) | rule 2 %s (CI lo %+.4f > -0.005) | rule 3 %s (pooled IoU %.4f vs %.4f)\n', ...
    mark(G.r1), S.mean, mark(G.r2), S.ci(1), mark(G.r3), pT, pC);
say('      rule 4 %s (worst image %+.4f >= -0.10) | rule 5 %s (cross-device dIoU %s)\n', ...
    mark(G.r4), min(d), mark(G.r5), mat2str(round(cdev.dIoU', 4)));
if G.r1 && G.r2 && G.r3 && G.r4 && G.r5
    G.decision = "ADOPT";
elseif (G.r1 && G.r2 && ~(G.r4 && G.r5)) || (S.mean >= 0.001 && S.mean < 0.003)
    G.decision = "AMBIGUOUS";
else
    G.decision = "REJECT";
end
say('Decision by the rule: %s\n', G.decision);
end


function s = moverText(M)
if height(M) == 0
    s = '(none)';
    return;
end
s = char(strjoin(compose("%s %+.3f", M.Image, M.Delta), ', '));
end


function s = mark(ok)
s = 'FAIL';
if ok, s = 'OK'; end
end


% =========================================================================
% SHARED
% =========================================================================

function D = loadDomain(cfg)
% Everything every step needs, read once. Each cached P passes probCache's
% leakage guard on read.
cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
M = cnngvf.manifest(cfg);
images = cnngvf.probCache("list", cacheDir);
images = images(ismember(images, M.Name));
[names, seedRows] = cnngvf.loadPoints(cfg);
n = numel(images);
assert(n == 68 && numel(unique(images)) == 68, 'run_refine:domain', ...
    'Expected 68 unique cached images, found %d.', n);
D.n = n; D.images = images;
D.P = cell(n, 1); D.gt = cell(n, 1); D.seeds = cell(n, 1);
D.fold = nan(n, 1); D.dset = strings(n, 1); D.gtPx = nan(n, 1);
stamps = strings(n, 1);
for i = 1:n
    k = find(M.Name == images(i), 1);
    got = cnngvf.probCache("read", cacheDir, images(i), cfg);   % leakage-guarded
    D.P{i} = got.P;
    D.fold(i) = got.meta.fold;
    D.dset(i) = M.Dataset(k);
    D.seeds{i} = seedRows(names == images(i), :);
    D.gt{i} = rnv.loadMask(M.GtPath(k));
    D.gtPx(i) = nnz(D.gt{i});
    stamps(i) = stampOf(got.meta);
end
u = unique(stamps);
assert(numel(u) == 1, 'run_refine:csvStamps', ...
    'Cached P were written against %d different localization CSV states.', numel(u));
D.csvStamp = u;
fprintf('Loaded %d images from %s\n', n, cacheDir);
end


function s = stampOf(meta)
s = "(no csv stamp in metadata)";
f = fieldnames(meta);
hit = f(contains(lower(f), 'csv'));
if ~isempty(hit)
    s = strjoin(string(evalc('disp(meta.(hit{1}))')), '');
    s = strtrim(regexprep(s, '\s+', ' '));
end
end


function Q = quintiles(D, refineDir)
% Frozen once from GT areas, reused by every later step.
path = fullfile(refineDir, 'quintiles.mat');
if isfile(path)
    s = load(path, 'Q');
    Q = s.Q;
    assert(isequal(sort(Q.Image), sort(D.images)), 'run_refine:quintiles', ...
        'Frozen quintiles cover a different image set.');
    return;
end
[~, ord] = sort(D.gtPx, 'ascend');
q = zeros(D.n, 1);
q(ord) = ceil((1:D.n)' * 5 / D.n);       % 13 / 14 / 13 / 14 / 14 on 68 images
Q = table(D.images, D.dset, D.gtPx, q, 'VariableNames', {'Image', 'Dataset', 'GtPx', 'Q'});
if ~exist(refineDir, 'dir'), mkdir(refineDir); end
save(path, 'Q');
writetable(Q, fullfile(refineDir, 'quintiles.csv'));
end


function [T, trace] = scoreAll(cfg, D, wantTrace)
% The production path on every image, one per-image row each.
n = D.n;
[tp, fp, fn, predPx, iou, f1, prec, rec] = deal(zeros(n, 1));
traces = cell(n, 1);
for i = 1:n
    P = double(D.P{i});
    if wantTrace
        [mask, ~, tr] = cnngvf.maskFromProb(P, D.seeds{i}, cfg);
        tr = [table(repmat(D.images(i), height(tr), 1), 'VariableNames', {'image'}), tr];
        traces{i} = tr;
    else
        mask = cnngvf.maskFromProb(P, D.seeds{i}, cfg);
    end
    g = D.gt{i};
    m = rnv.metrics(mask, g);
    tp(i) = nnz(mask & g); predPx(i) = nnz(mask);
    fp(i) = predPx(i) - tp(i); fn(i) = D.gtPx(i) - tp(i);
    iou(i) = m.iou; f1(i) = m.f1; prec(i) = m.precision; rec(i) = m.recall;
end
T = table(D.images, D.dset, D.fold, D.gtPx, predPx, tp, fp, fn, iou, f1, prec, rec, ...
    'VariableNames', {'Image', 'Dataset', 'Fold', 'GtPx', 'PredPx', 'TP', 'FP', 'FN', ...
    'Iou', 'F1', 'Precision', 'Recall'});
trace = vertcat(traces{:});
end


function cfg = setField(cfg, name, value)
cfg.Post.(name) = value;
end


function s = pf(ok)
s = 'PASS';
if ~ok, s = 'FAIL'; end
end


function sayBoth(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end
