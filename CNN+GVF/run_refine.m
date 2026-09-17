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
%    2  seed features and the separability check (diagnostic, GT in diag/ only)
%    4  relax guard, cfg.Post.relaxAlpha (step 3 was skipped on step 2's gate)
%    5  where the large-lesion pixels are lost (diagnostic, GT in diag/ only)
%    6  test-time augmentation: GPU, builds <outputDir>/probcache_tta/ (resumable),
%       then scores CUR on both caches with relaxFloor cross-fitted on each
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
    case 2
        step2(cfg, D, outDir, refineDir);
    case 4
        step4(cfg, D, Q, outDir, refineDir);
    case 5
        step5(cfg, D, Q, outDir, refineDir);
    case 6
        step6(cfg, D, Q, outDir, refineDir);
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
% STEP 2
% =========================================================================

function step2(cfg, D, outDir, refineDir)
% Diagnostic: ground truth labels the seeds, written to diag/ only. Builds no
% production behaviour; decides whether step 3 is worth building.
fid = fopen(fullfile(outDir, 'step2_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
t0 = tic;
tol = cfg.Patch.seedTolerance;
say('Step 2 - seed features and separability | %s | %d images | stray = more than %d px from FilledGT\n\n', ...
    cfg.Arm, D.n, tol);

% ---- features (production-safe) and labels (GT) --------------------------
T = seedTable(cfg, D);
diagDir = fullfile(refineDir, 'diag');
if ~exist(diagDir, 'dir'), mkdir(diagDir); end
writetable(T, fullfile(diagDir, 'step2_seed_features.csv'));

nStray = nnz(T.stray);
nImg = numel(unique(T.image(T.stray)));
say('Seeds %d | near %d | stray %d on %d images (REDUCE_RESULTS section 10: 874 / 842 / 32 / 23)\n', ...
    height(T), nnz(~T.stray), nStray, nImg);
assert(height(T) == 874 && nStray == 32 && nImg == 23, 'run_refine:strayCounts', ...
    'Seed counts do not match REDUCE_RESULTS section 10. STOP.');

% ---- AUC per feature ------------------------------------------------------
feats = ["fired", "peakP_r16", "peakP_r32", "meanP_r16", "support_R64", "support_R100", ...
    "support_R150", "nnDist", "regionArea", "regionMeanP", "distToFiredComp", "imageSeedCount"];
A = aucTable(say, T, feats);
writetable(A, fullfile(outDir, 'auc.csv'));

% ---- candidate rules ------------------------------------------------------
oracle = oracleGain(cfg, D);
rules = table();
for k = [0 1 2 3 5]
    for p0 = [0.05 0.10 0.20 0.30 0.50 0.70 0.90]
        rules = [rules; ruleRow("~fired & support_R100<=k & peakP_r32<p0", k, p0, NaN, ...
            ~T.fired & T.support_R100 <= k & T.peakP_r32 < p0, T, oracle)]; %#ok<AGROW>
        rules = [rules; ruleRow("support_R100<=k & peakP_r32<p0", k, p0, NaN, ...
            T.support_R100 <= k & T.peakP_r32 < p0, T, oracle)]; %#ok<AGROW>
    end
end
for d0 = [0 20 50 100 150 200 300 500]
    rules = [rules; ruleRow("~fired & distToFiredComp>d0", NaN, NaN, d0, ...
        ~T.fired & T.distToFiredComp > d0, T, oracle)]; %#ok<AGROW>
end
writetable(rules, fullfile(outDir, 'rules.csv'));

budget = floor(0.02 * nnz(~T.stray));
ok = rules.TrueDropped <= budget & rules.SingleSeedOrphans == 0;
say('\nRules within budget (true dropped <= %d = 2%% of %d, no single-seed image orphaned), best catch first\n', ...
    budget, nnz(~T.stray));
say('  %-42s  k    p0     d0   stray  true  orphanImg  oracle-weighted dIoU\n', 'rule');
R = sortrows(rules(ok, :), {'StrayCaught', 'TrueDropped', 'OracleWeighted'}, {'descend', 'ascend', 'descend'});
for j = 1:min(15, height(R))
    say('  %-42s  %-3g  %-5g  %-4g  %2d/%d  %4d  %9d  %+.4f\n', R.Rule(j), R.k(j), R.p0(j), R.d0(j), ...
        R.StrayCaught(j), nStray, R.TrueDropped(j), R.ImagesAllDropped(j), R.OracleWeighted(j));
end
say('  oracle ceiling (every stray seed dropped, cached P): %+.4f\n', sum(oracle.Gain) / D.n);

best = 0;
if height(R) > 0
    best = R.StrayCaught(1) / nStray;
end
if best >= 0.5
    verdict = "GO to step 3";
elseif best < 0.3
    verdict = "SKIP step 3, go to step 4";
else
    verdict = "AMBIGUOUS - stop and ask";
end
say('\nBest in-budget catch rate %.0f%% -> %s\n', 100 * best, verdict);

% ---- the stray seeds themselves ------------------------------------------
say('\nStray seeds (oracle relax gain per image in brackets)\n');
S = T(T.stray, :);
[~, loc] = ismember(S.image, oracle.Image);
S.oracleGain = oracle.Gain(loc);
S = sortrows(S, {'oracleGain', 'image'}, {'descend', 'ascend'});
say('  %-12s seed  gtDist fired peak16 peak32 mean16 sup64 sup100 sup150 nnDist regArea regMeanP dFired nSeeds\n', 'image');
for j = 1:height(S)
    say('  %-12s %3d %7.0f %5d %6.2f %6.2f %6.2f %5d %6d %6d %6.0f %7d %8.2f %6.0f %6d  [%+.3f]\n', ...
        S.image(j), S.seedIdx(j), S.gtDist(j), S.fired(j), S.peakP_r16(j), S.peakP_r32(j), S.meanP_r16(j), ...
        S.support_R64(j), S.support_R100(j), S.support_R150(j), S.nnDist(j), S.regionArea(j), ...
        S.regionMeanP(j), S.distToFiredComp(j), S.imageSeedCount(j), S.oracleGain(j));
end
writetable(S, fullfile(diagDir, 'step2_stray_seeds.csv'));
say('\n%.0f s\n', toc(t0));
fclose(fid);
end


function T = seedTable(cfg, D)
% Per-seed features (production-safe, cnngvf.seedFeatures) plus the GT label:
% stray = more than Patch.seedTolerance px from FilledGT. D.V, when present
% (a TTA cache), adds ttaVar_r16: mean across-transform variance in a 16 px disk.
tol = cfg.Patch.seedTolerance;
parts = cell(D.n, 1);
for i = 1:D.n
    seeds = D.seeds{i};
    F = cnngvf.seedFeatures(double(D.P{i}), seeds, cfg);
    dist = bwdist(D.gt{i});
    [h, w] = size(dist);
    r = min(max(round(seeds(:, 2)), 1), h);
    c = min(max(round(seeds(:, 1)), 1), w);
    gtDist = dist(sub2ind([h w], r, c));
    if isfield(D, 'V') && ~isempty(D.V{i})
        F.ttaVar_r16 = diskMeans(double(D.V{i}), r, c, 16);
    end
    parts{i} = [table(repmat(D.images(i), size(seeds, 1), 1), repmat(D.dset(i), size(seeds, 1), 1), ...
        (1:size(seeds, 1))', seeds(:, 1), seeds(:, 2), gtDist, gtDist > tol, ...
        'VariableNames', {'image', 'dataset', 'seedIdx', 'cx', 'cy', 'gtDist', 'stray'}), F];
end
T = vertcat(parts{:});
end


function v = diskMeans(M, r, c, rad)
[h, w] = size(M);
v = zeros(numel(r), 1);
for s = 1:numel(r)
    r0 = max(r(s) - rad, 1); r1 = min(r(s) + rad, h);
    c0 = max(c(s) - rad, 1); c1 = min(c(s) + rad, w);
    [cc, rr] = meshgrid(c0:c1, r0:r1);
    sub = M(r0:r1, c0:c1);
    v(s) = mean(sub((rr - r(s)).^2 + (cc - c(s)).^2 <= rad^2));
end
end


function A = aucTable(say, T, feats)
say('\nAUC, stray vs true (oriented >= 0.5; "low" = stray seeds have LOWER values)\n');
say('  %-16s  %-4s   all   DRAC22    TUH\n', 'feature', 'dir');
A = table();
for f = feats
    x = double(T.(f));
    a = aucOf(x, T.stray);
    dirn = "high";
    if a < 0.5, dirn = "low"; end
    orient = @(v) max(v, 1 - v);
    ad = aucOf(x(T.dataset == "DRAC22"), T.stray(T.dataset == "DRAC22"));
    at = aucOf(x(T.dataset == "TUH"), T.stray(T.dataset == "TUH"));
    if dirn == "low", ad = 1 - ad; at = 1 - at; end
    say('  %-16s  %-4s  %.3f   %.3f   %.3f\n', f, dirn, orient(a), ad, at);
    A = [A; table(f, dirn, orient(a), ad, at, 'VariableNames', {'Feature', 'Direction', 'AUC', 'AUC_DRAC22', 'AUC_TUH'})]; %#ok<AGROW>
end
end


function a = aucOf(x, pos)
% Rank-based (Mann-Whitney) AUC, ties averaged. No Statistics Toolbox.
n1 = nnz(pos); n0 = nnz(~pos);
if n1 == 0 || n0 == 0
    a = NaN;
    return;
end
[~, ~, g] = unique(x(:));
cnt = accumarray(g, 1);
upper = cumsum(cnt);
rankOf = upper - (cnt - 1) / 2;         % average rank of each distinct value
r = rankOf(g);
a = (sum(r(pos)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end


function O = oracleGain(cfg, D)
% REDUCE_RESULTS section 10, relax 0.20 on cached P: near-GT seeds only vs all.
% Cached, not fresh, because a post-processing gate cannot re-infer P.
X = readtable(fullfile(cfg.Paths.Output, 'oracle_seeds', 'oracle_per_image.csv'), 'TextType', 'string');
all_ = X(X.Cond == "all_cachedP" & X.Post == "relax", :);
near = X(X.Cond == "near_cachedP" & X.Post == "relax", :);
[ok, loc] = ismember(D.images, all_.Image);
[ok2, loc2] = ismember(D.images, near.Image);
assert(all(ok) && all(ok2), 'oracle_per_image.csv does not cover every image');
O = table(D.images, near.Iou(loc2) - all_.Iou(loc), 'VariableNames', {'Image', 'Gain'});
end


function row = ruleRow(name, k, p0, d0, drop, T, oracle)
% Stray caught, true dropped, orphans, and the oracle-weighted capture: each
% image's oracle gain times the fraction of its stray seeds this rule catches.
imgs = unique(T.image);
allDropped = 0; single = 0; weighted = 0;
for im = imgs'
    s = T.image == im;
    if all(drop(s)), allDropped = allDropped + 1; end
    if nnz(s) == 1 && drop(s), single = single + 1; end
    st = s & T.stray;
    if any(st)
        weighted = weighted + oracle.Gain(oracle.Image == im) * nnz(drop(st)) / nnz(st);
    end
end
row = table(name, k, p0, d0, nnz(drop & T.stray), nnz(drop & ~T.stray), allDropped, single, ...
    weighted / numel(imgs), 'VariableNames', {'Rule', 'k', 'p0', 'd0', 'StrayCaught', 'TrueDropped', ...
    'ImagesAllDropped', 'SingleSeedOrphans', 'OracleWeighted'});
end


% =========================================================================
% STEP 4
% =========================================================================

function step4(cfg, D, Q, outDir, refineDir)
fid = fopen(fullfile(outDir, 'step4_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
t0 = tic;
say('Step 4 - relax guard | %s | %d images | %s\n\n', cfg.Arm, D.n, cfg.Paths.Output);

% CUR is still REF0: step 1 rejected, step 3 skipped.
assert(cfg.Post.relaxAlpha == 0 && cfg.Post.relaxFloor == 0.20 && isinf(cfg.Post.claimMaxDist), ...
    'CUR is not REF0');
[CUR, trace0] = scoreAll(cfg, D, true);
ref0 = readtable(fullfile(refineDir, 'step0', 'per_image_REF0.csv'), 'TextType', 'string');
dRef = max(abs(CUR.Iou - ref0.Iou));
say('Flag off (relaxAlpha 0) vs step-0 REF0: max |dIoU| %.2g\n\n', dRef);
assert(dRef < 1e-12, 'run_refine:flagOff', 'relaxAlpha = 0 does not reproduce REF0');

% Most conservative first: alpha 0 (feature off), and within an alpha the CUR
% floor 0.20 before the lower ones.
[fl, al] = meshgrid([0.20 0.15 0.10], [0 0.3 0.5 0.7 0.9]);
al = al'; fl = fl';
grid = table(al(:), fl(:), 'VariableNames', {'relaxAlpha', 'relaxFloor'});
fn = @(g) scoreAll(setField(setField(cfg, 'relaxAlpha', g.relaxAlpha), 'relaxFloor', g.relaxFloor), D, false);
CF = cnngvf.crossFit(grid, fn, D.fold, D.dset);
XF = CF.perImage;

arms = {"CUR (alpha 0, 0.20)", CUR; "cross-fitted", XF};
[summary, stats] = reportArms(say, arms, CUR, Q);
say('\nCross-fitted chosen per fold (alpha / floor): %s\n', strjoin(compose("%g/%.2f", ...
    grid.relaxAlpha(CF.chosen.GridRow), grid.relaxFloor(CF.chosen.GridRow)), ', '));
cdev = crossDeviceDeltas(say, CF, grid, 'relaxAlpha', CUR, D);
for k = 1:height(cdev)
    say('   (floor %.2f)\n', grid.relaxFloor(cdev.GridRow(k)));
end

say('\nSurface (full domain, NOT a result)\n');
say('  alpha  floor    IoU   prec    rec  pooled IoU  Pred/GT Q1  Q2\n');
for g = 1:height(CF.surface)
    s = CF.surface(g, :);
    r = cnngvf.refineScore(CF.rows{g}, Q);
    say('  %-5g  %.2f  %.4f %.4f %.4f  %.4f      %.2f  %.2f\n', s.relaxAlpha, s.relaxFloor, s.MeanIou, ...
        s.MeanPrec, s.MeanRec, s.PooledIou, r.PredGt_Q1, r.PredGt_Q2);
end

G = gate(say, XF, CUR, cdev);

% ---- relaxed seeds only: what the guard does to their regions ------------
g = mode(CF.chosen.GridRow);
c = setField(setField(cfg, 'relaxAlpha', grid.relaxAlpha(g)), 'relaxFloor', grid.relaxFloor(g));
[~, traceG] = scoreAll(c, D, true);
relaxed = ~isnan(trace0.relaxThrUsed);
[mP0, mPG] = deal(nan(height(trace0), 1));
for i = 1:D.n
    s = find(trace0.image == D.images(i) & relaxed);
    if isempty(s), continue; end
    P = double(D.P{i});
    [~, ps0] = cnngvf.maskFromProb(P, D.seeds{i}, cfg);
    [~, psG] = cnngvf.maskFromProb(P, D.seeds{i}, c);
    for j = s'
        k = trace0.seedIdx(j);
        if nnz(ps0{k}), mP0(j) = mean(P(ps0{k})); end
        if nnz(psG{k}), mPG(j) = mean(P(psG{k})); end
    end
end
say(['\nRelaxed seeds at alpha 0 (%d), modal chosen setting alpha %g / floor %.2f:\n' ...
     '  median claimed area %d -> %d | mean region P %.3f -> %.3f | still relaxed %d, now disk %d\n'], ...
    nnz(relaxed), grid.relaxAlpha(g), grid.relaxFloor(g), round(median(trace0.claimedCompArea(relaxed))), ...
    round(median(traceG.claimedCompArea(relaxed))), mean(mP0(relaxed), 'omitnan'), mean(mPG(relaxed), 'omitnan'), ...
    nnz(relaxed & ~isnan(traceG.relaxThrUsed)), nnz(relaxed & isnan(traceG.relaxThrUsed)));

say('\n%.0f s\n', toc(t0));
fclose(fid);
writetable(summary, fullfile(outDir, 'summary.csv'));
writetable(stats, fullfile(outDir, 'stats.csv'));
writetable(CUR, fullfile(outDir, 'per_image_CUR.csv'));
writetable(XF, fullfile(outDir, 'per_image_crossfit.csv'));
writetable(CF.surface, fullfile(outDir, 'surface_relaxAlpha_floor.csv'));
writetable(CF.chosen, fullfile(outDir, 'chosen_relaxAlpha_floor.csv'));
writetable(cdev, fullfile(outDir, 'crossdevice.csv'));
writetable(struct2table(G), fullfile(outDir, 'gate.csv'));
end


% =========================================================================
% STEP 5
% =========================================================================

function step5(cfg, D, Q, outDir, refineDir)
% Diagnostic on CUR: every false-negative pixel goes into exactly one bucket.
%   outside        outside every patch window
%   lowP           inside a window, P < relaxFloor
%   midP           inside, relaxFloor <= P < tau
%   highP_dropped  inside, P >= tau, but not in the output (claiming, MinArea)
% The windows come from cnngvf.seedPatch, the geometry cnngvf.inferProbMap
% pastes with. Its raised-cosine weight is floored at 1e-3, so the effective
% window (weight > 0) is the whole 512 px patch.
fid = fopen(fullfile(outDir, 'step5_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
t0 = tic;
say('Step 5 - where the large-lesion pixels are lost | %s | %d images | CUR: relax %.2f, tau %.2f, patch %d\n\n', ...
    cfg.Arm, D.n, cfg.Post.relaxFloor, cfg.Post.tau, cfg.Patch.size);
covDir = fullfile(refineDir, 'diag', 'coverage');
if ~exist(covDir, 'dir'), mkdir(covDir); end

[~, qloc] = ismember(D.images, Q.Image);
q = Q.Q(qloc);
names = ["outside", "lowP", "midP", "highP_dropped"];
B = zeros(D.n, 4);
[gtPx, fnPx, sumPgt, gtInWin, sumPgtWin] = deal(zeros(D.n, 1));
for i = 1:D.n
    P = double(D.P{i});
    gt = D.gt{i};
    [h, w] = size(P);
    pred = cnngvf.maskFromProb(P, D.seeds{i}, cfg);

    win = false(h, w);
    for s = 1:size(D.seeds{i}, 1)
        [~, rows, cols] = cnngvf.seedPatch(zeros(h, w, 'single'), D.seeds{i}(s, :), D.seeds{i}(s, :), cfg);
        win(rows(rows >= 1 & rows <= h), cols(cols >= 1 & cols <= w)) = true;
    end

    fn = gt & ~pred;
    bucket = zeros(h, w, 'uint8');
    bucket(fn & ~win) = 1;
    bucket(fn & win & P < cfg.Post.relaxFloor) = 2;
    bucket(fn & win & P >= cfg.Post.relaxFloor & P < cfg.Post.tau) = 3;
    bucket(fn & win & P >= cfg.Post.tau) = 4;
    assert(nnz(bucket) == nnz(fn), 'a false-negative pixel fell in no bucket');
    B(i, :) = accumarray(double(bucket(fn)), 1, [4 1])';
    gtPx(i) = nnz(gt); fnPx(i) = nnz(fn);
    sumPgt(i) = sum(P(gt)); gtInWin(i) = nnz(gt & win); sumPgtWin(i) = sum(P(gt & win));

    if q(i) == 5
        writeOverlay(fullfile(covDir, char(erase(D.images(i), ".png") + "_coverage.png")), ...
            P, gt, pred, win, bucket);
    end
end

T = [table(D.images, D.dset, q, gtPx, fnPx, 'VariableNames', {'Image', 'Dataset', 'Q', 'GtPx', 'FnPx'}), ...
    array2table(B, 'VariableNames', cellstr(names)), ...
    table(sumPgt ./ max(gtPx, 1), gtInWin ./ max(gtPx, 1), sumPgtWin ./ max(gtInWin, 1), ...
    'VariableNames', {'MeanPInGt', 'GtInWindowFrac', 'MeanPInGtInWindow'})];
writetable(T, fullfile(refineDir, 'diag', 'step5_fn_buckets.csv'));

say('FN pixel buckets by size quintile (pooled pixels; fractions of that quintile''s FN)\n');
say('  Q   FN px       outside   lowP    midP   highP_dropped | GT in window | mean P in GT | in GT & window\n');
for k = 1:5
    s = q == k;
    f = sum(B(s, :), 1) / max(sum(fnPx(s)), 1);
    say('  Q%d  %9d   %.3f   %.3f   %.3f   %.3f         |   %.3f      |   %.3f      |   %.3f\n', k, sum(fnPx(s)), f, ...
        sum(gtInWin(s)) / sum(gtPx(s)), sum(sumPgt(s)) / sum(gtPx(s)), sum(sumPgtWin(s)) / max(sum(gtInWin(s)), 1));
end
s = true(D.n, 1);
f = sum(B, 1) / sum(fnPx);
say('  all %9d   %.3f   %.3f   %.3f   %.3f\n', sum(fnPx), f);

say('\nTen largest GT images\n');
say('  %-12s %-6s %8s %8s  outside  lowP   midP   highP | GT in win  mean P in GT\n', 'image', 'dset', 'GT px', 'FN px');
[~, ord] = sort(gtPx, 'descend');
for i = ord(1:10)'
    f = B(i, :) / max(fnPx(i), 1);
    say('  %-12s %-6s %8d %8d  %.3f   %.3f  %.3f  %.3f |  %.3f      %.3f\n', D.images(i), D.dset(i), gtPx(i), fnPx(i), f, ...
        T.GtInWindowFrac(i), T.MeanPInGt(i));
end
say('  the ten hold %.1f%% of all GT pixels and %.1f%% of all FN pixels\n', ...
    100 * sum(gtPx(ord(1:10))) / sum(gtPx), 100 * sum(fnPx(ord(1:10))) / sum(fnPx));

q5 = sum(B(q == 5, 1)) / sum(fnPx(q == 5));
if q5 >= 0.40
    verdict = "step 7 ENABLED";
elseif q5 < 0.20
    verdict = "step 7 SKIPPED - large-lesion loss is model-side (low P inside the window); the lever is training";
else
    verdict = "AMBIGUOUS - stop and ask";
end
say('\nGate: outside = %.1f%% of Q5 FN pixels -> %s\n', 100 * q5, verdict);
say('Overlays: %s (%d Q5 images; green TP, blue FP, FN red outside / orange lowP / yellow midP / magenta highP, white window edge)\n', ...
    covDir, nnz(q == 5));
say('\n%.0f s\n', toc(t0));
fclose(fid);
writetable(T, fullfile(outDir, 'per_image_buckets.csv'));
end


% =========================================================================
% STEP 6
% =========================================================================

function step6(cfg, D, Q, outDir, refineDir)
fid = fopen(fullfile(outDir, 'step6_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
t0 = tic;
say('Step 6 - test-time augmentation (d4) | %s | %d images | device %s\n\n', cfg.Arm, D.n, cfg.GpuName);
M = cnngvf.manifest(cfg);
ttaSub = 'probcache_tta';
ttaDir = fullfile(cfg.Paths.Output, ttaSub);

% ---- test 1, before any TTA inference: "none" reproduces the cached P ------
i = find(D.images == "458.png");
net = foldNet(cfg, D.fold(i));
[~, chans] = cnngvf.loadChannels(M.ImgPath(M.Name == D.images(i)), cfg);
c = cfg; c.Infer.tta = "none";
Pnone = single(cnngvf.inferProbMap(chans, D.seeds{i}, net, c));
dNone = max(abs(Pnone(:) - D.P{i}(:)));
say('Test 1: tta "none" re-inferred for 458.png (fold %d) vs cached P: max |dP| %.3g\n', D.fold(i), dNone);
assert(isequal(Pnone, D.P{i}), 'run_refine:ttaNone', ...
    'Re-inferring 458.png with tta "none" does not reproduce the cached P exactly (max |dP| %.3g). STOP and ask.', dNone);

% ---- build the TTA cache, fold by fold, resumable -------------------------
c.Infer.tta = "d4";
tInfer = tic; nNew = 0;
for f = 1:max(D.fold)
    todo = find(D.fold == f & ~isfile(fullfile(ttaDir, erase(D.images, ".png") + ".mat")));
    if isempty(todo), continue; end
    [net, netPath] = foldNet(cfg, f);
    for i = todo'
        tImg = tic;
        [~, chans] = cnngvf.loadChannels(M.ImgPath(M.Name == D.images(i)), cfg);
        [P, V] = cnngvf.inferProbMap(chans, D.seeds{i}, net, c);
        meta = D.meta{i};
        meta.tta = "d4";
        meta.netFile = string(netPath);
        meta.source = "run_refine step 6";
        cnngvf.probCache("write", ttaDir, D.images(i), P, meta, V);
        nNew = nNew + 1;
        fprintf('  tta fold %d %-12s %3d seeds  %5.1f s\n', f, D.images(i), size(D.seeds{i}, 1), toc(tImg));
    end
end
say('TTA cache: %s | %d images inferred now, %.0f s\n\n', ttaDir, nNew, toc(tInfer));

Dt = loadDomain(cfg, ttaSub);                           % leakage-guarded again
assert(isequal(Dt.images, D.images) && isequal(Dt.fold, D.fold), 'TTA cache is not paired with the original');
assert(all(cellfun(@(m) isfield(m, 'tta') && m.tta == "d4", Dt.meta)), 'a TTA cache file lacks meta.tta = "d4"');

% ---- score CUR on both caches; relaxFloor cross-fitted on each ------------
floors = [0.05 0.10 0.15 0.20 0.25 0.30]';
grid = table(floors, 'VariableNames', {'relaxFloor'});
CFo = cnngvf.crossFit(grid, @(g) scoreAll(setField(cfg, 'relaxFloor', g.relaxFloor), D, false), D.fold, D.dset);
CFt = cnngvf.crossFit(grid, @(g) scoreAll(setField(cfg, 'relaxFloor', g.relaxFloor), Dt, false), Dt.fold, Dt.dset);
fixO = CFo.rows{floors == 0.20};
fixT = CFt.rows{floors == 0.20};
XFo = CFo.perImage; XFt = CFt.perImage;
say('Original cross-fit reproduced: %.4f (step 0: 0.5061)\n', mean(XFo.Iou));
assert(abs(mean(XFo.Iou) - 0.5061) < 5e-5, 'original cross-fit no longer reproduces 0.5061');

say('\nvs CUR cross-fitted (headline comparator)\n');
[summary, stats] = reportArms(say, {"CUR, cross-fitted floor", XFo; "TTA, cross-fitted floor", XFt}, XFo, Q);
say('\nvs REF0 (relax 0.20 fixed)\n');
[summary2, stats2] = reportArms(say, {"REF0, 0.20 fixed", fixO; "TTA, 0.20 fixed", fixT; "TTA, cross-fitted floor", XFt}, fixO, Q);
say('\nChosen floor per fold: original %s | TTA %s\n', mat2str(floors(CFo.chosen.GridRow)'), mat2str(floors(CFt.chosen.GridRow)'));

% Cross-device: on each cache choose the floor on one device, score the other;
% the delta is TTA's cross-device arm minus the original's.
cdev = CFt.crossDevice;
cdev.FloorTta = floors(CFt.crossDevice.GridRow);
cdev.FloorOrig = floors(CFo.crossDevice.GridRow);
cdev.dIoU = zeros(height(cdev), 1);
for k = 1:height(cdev)
    te = D.dset == cdev.Test(k);
    cdev.dIoU(k) = mean(CFt.rows{CFt.crossDevice.GridRow(k)}.Iou(te) - CFo.rows{CFo.crossDevice.GridRow(k)}.Iou(te));
    say('Cross-device: choose on %s, score %s: TTA %.2f -> %.4f | original %.2f -> %.4f | %+.4f\n', cdev.Train(k), ...
        cdev.Test(k), cdev.FloorTta(k), CFt.crossDevice.TestIou(k), cdev.FloorOrig(k), CFo.crossDevice.TestIou(k), cdev.dIoU(k));
end

say('\nSurface, relaxFloor (full domain, NOT a result)\n  floor   original  TTA\n');
for g = 1:height(grid)
    say('  %.2f    %.4f    %.4f\n', floors(g), CFo.surface.MeanIou(g), CFt.surface.MeanIou(g));
end

say('\ntau sensitivity, relax 0.20 fixed (a curve, NOT a result)\n  tau    original  TTA\n');
taus = 0.30:0.05:0.65;
curve = table(taus', zeros(numel(taus), 1), zeros(numel(taus), 1), 'VariableNames', {'tau', 'Original', 'TTA'});
for j = 1:numel(taus)
    ct = cfg; ct.Post.tau = taus(j);
    curve.Original(j) = mean(scoreAll(ct, D, false).Iou);
    curve.TTA(j) = mean(scoreAll(ct, Dt, false).Iou);
    say('  %.2f   %.4f    %.4f\n', taus(j), curve.Original(j), curve.TTA(j));
end

G = gate(say, XFt, XFo, cdev);

% ---- P itself: how much TTA moved it ---------------------------------------
dP = cellfun(@(a, b) mean(abs(double(a(:)) - double(b(:)))), Dt.P, D.P);
ext = @(C) mean(cellfun(@(a) mean(a(:) > 0.9 | (a(:) < 0.1 & a(:) > 0)), C));
say('\nP: mean |P_tta - P| per image %.4f (max %.4f) | share of covered pixels at P > 0.9 or 0 < P < 0.1: %.3f -> %.3f\n', ...
    mean(dP), max(dP), ext(D.P), ext(Dt.P));

% ---- step-2 separability on TTA P, plus the TTA variance feature ----------
Tt = seedTable(cfg, Dt);
writetable(Tt, fullfile(refineDir, 'diag', 'step6_seed_features_tta.csv'));
feats = ["fired", "peakP_r16", "peakP_r32", "meanP_r16", "support_R64", "support_R100", ...
    "support_R150", "nnDist", "regionArea", "regionMeanP", "distToFiredComp", "imageSeedCount", "ttaVar_r16"];
say('\nOn TTA P');
A = aucTable(say, Tt, feats);
writetable(A, fullfile(outDir, 'auc_tta.csv'));
oracle = oracleGain(cfg, Dt);
budget = floor(0.02 * nnz(~Tt.stray));
vq = sort(Tt.ttaVar_r16);
rules = table();
for k = [0 1 2 3 5]
    for p0 = [0.05 0.10 0.20 0.30 0.50 0.70 0.90]
        rules = [rules; ruleRow("support_R100<=k & peakP_r32<p0", k, p0, NaN, ...
            Tt.support_R100 <= k & Tt.peakP_r32 < p0, Tt, oracle)]; %#ok<AGROW>
        rules = [rules; ruleRow("~fired & support_R100<=k & peakP_r32<p0", k, p0, NaN, ...
            ~Tt.fired & Tt.support_R100 <= k & Tt.peakP_r32 < p0, Tt, oracle)]; %#ok<AGROW>
    end
    for qv = [0.5 0.75 0.9 0.95]
        v0 = vq(ceil(qv * numel(vq)));
        rules = [rules; ruleRow("support_R100<=k & ttaVar_r16>v0 (v0 = var quantile p0)", k, qv, v0, ...
            Tt.support_R100 <= k & Tt.ttaVar_r16 > v0, Tt, oracle)]; %#ok<AGROW>
    end
end
writetable(rules, fullfile(outDir, 'rules_tta.csv'));
ok = rules.TrueDropped <= budget & rules.SingleSeedOrphans == 0;
catchPre = max([0; rules.StrayCaught(ok & ~contains(rules.Rule, "ttaVar"))]) / nnz(Tt.stray);
catchVar = max([0; rules.StrayCaught(ok)]) / nnz(Tt.stray);
say('\nBest in-budget stray catch on TTA P: pre-registered forms %.0f%% | with the variance feature %.0f%% (step 2: 25%%)\n', ...
    100 * catchPre, 100 * catchVar);
if catchVar - 0.25 >= 0.15
    say('  Variance raises the best catch rate by >= 15 points: record and ASK whether to re-run step 3.\n');
end

say('\n%.0f s total\n', toc(t0));
fclose(fid);
writetable([summary; summary2], fullfile(outDir, 'summary.csv'));
writetable([stats; stats2], fullfile(outDir, 'stats.csv'));
writetable(fixT, fullfile(outDir, 'per_image_tta_fixed020.csv'));
writetable(XFo, fullfile(outDir, 'per_image_orig_crossfit.csv'));
writetable(XFt, fullfile(outDir, 'per_image_tta_crossfit.csv'));
writetable(CFt.surface, fullfile(outDir, 'surface_tta_relaxFloor.csv'));
writetable(CFt.chosen, fullfile(outDir, 'chosen_tta_relaxFloor.csv'));
writetable(curve, fullfile(outDir, 'tau_curve.csv'));
writetable(cdev, fullfile(outDir, 'crossdevice.csv'));
writetable(struct2table(G), fullfile(outDir, 'gate.csv'));
end


function [net, path] = foldNet(cfg, f)
% The network that held fold f out - the only one allowed to infer its images.
path = fullfile(cfg.Paths.Main, 'models', 'cv', sprintf('%s_fold%d.mat', cfg.CV.ModelArm, f));
s = load(path, 'net');
net = s.net;
end


function writeOverlay(path, P, gt, pred, win, bucket)
base = repmat(0.35 * P, 1, 1, 3);
paint = @(img, m, rgb) img .* ~m + reshape(rgb, 1, 1, 3) .* m;
img = base;
img = paint(img, gt & pred, [0.1 0.8 0.1]);
img = paint(img, pred & ~gt, [0.2 0.4 1.0]);
img = paint(img, bucket == 1, [1.0 0.1 0.1]);
img = paint(img, bucket == 2, [1.0 0.55 0.0]);
img = paint(img, bucket == 3, [1.0 1.0 0.0]);
img = paint(img, bucket == 4, [1.0 0.0 1.0]);
img = paint(img, bwperim(win), [1 1 1]);
imwrite(img, path);
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

function D = loadDomain(cfg, cacheSub)
% Everything every step needs, read once. Each cached P passes probCache's
% leakage guard on read. cacheSub defaults to the original cv/probcache;
% step 6 passes probcache_tta, which also carries V.
if nargin < 2
    cacheSub = fullfile('cv', 'probcache');
end
cacheDir = fullfile(cfg.Paths.Output, cacheSub);
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
D.V = cell(n, 1); D.meta = cell(n, 1); D.cacheDir = cacheDir;
stamps = strings(n, 1);
for i = 1:n
    k = find(M.Name == images(i), 1);
    got = cnngvf.probCache("read", cacheDir, images(i), cfg);   % leakage-guarded
    D.P{i} = got.P;
    D.meta{i} = got.meta;
    if isfield(got, 'V'), D.V{i} = got.V; end
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
