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
