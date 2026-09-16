function run_confirm_relax(outputDir)
%RUN_CONFIRM_RELAX Confirmatory pass for fallbackMode "relax" at the pre-registered floor 0.20.
%
%  run_confirm_relax                 % reads result/DRACTUH_unclustered/cv/probcache
%  run_confirm_relax(outputDir)      % an arm output folder that has been moved
%
%  The floor was chosen by run_ablate(arm, "fallback") on the images it was scored
%  on, and the images cannot be replaced, so re-scoring 0.20 alone would only
%  reproduce the sweep. What this adds is the selection made honest:
%
%   A  the production maskFromProb path at 0.20, paired vs disk r=15 and level set
%   B  cross-fitted: choose the floor on 4 folds, score the 5th
%   C  cross-device: choose on one dataset, score the other
%   D  the same floor on "unclustered_boundary", whose networks never saw TUH
%
%  Off cached P, ~70 s, no GPU. Writes <outputDir>/confirm/.
%  See also RUN_ABLATE, CNNGVF.SEEDFALLBACK, ../docs/REDUCE_RESULTS.md section 9.
floors = [0.05 0.10 0.15 0.20 0.25 0.30];
PRE = 0.20;

cfg = cnngvf.arm(cnngvf.defaults(), "DRACTUH_unclustered");
if nargin >= 1, cfg.Paths.Output = char(outputDir); end
outDir = fullfile(cfg.Paths.Output, 'confirm');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'confirm_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});

t0 = tic;
S = scoreArm(cfg, floors);
say('Confirmatory pass, relax floor %.2f - %s, %d images, tau %.2f, %.0f s\n\n', ...
    PRE, cfg.Arm, numel(S.images), cfg.Post.tau, toc(t0));

% ---- reference: level-set cell 1 from the factorial, same cache ----------
A = readtable(fullfile(cfg.Paths.Output, 'ablate', 'per_image.csv'), 'TextType', 'string');
[ok, loc] = ismember(S.images, A.Image);
assert(all(ok), 'factorial per_image.csv does not cover every image');
lsIou = A.Iou_cell1(loc);

% ---- A. pre-registered floor, production path ----------------------------
j = find(abs(floors - PRE) < 1e-9);
say('A. Pre-registered floor %.2f (maskFromProb, fallbackMode "relax")\n', PRE);
say('  %-22s    IoU  median      F1    prec     rec\n', '');
say('  %-22s %.4f  %.4f  %.4f  %.4f  %.4f\n', 'levelset (cell 1)', mean(lsIou), median(lsIou), mean(A.F1_cell1(loc)), NaN, NaN);
say('  %-22s %.4f  %.4f  %.4f  %.4f  %.4f\n', 'reduced, disk r=15', mean(S.disk.iou), median(S.disk.iou), mean(S.disk.f1), mean(S.disk.prec), mean(S.disk.rec));
say('  %-22s %.4f  %.4f  %.4f  %.4f  %.4f\n', 'reduced, relax 0.20', mean(S.iou(:,j)), median(S.iou(:,j)), mean(S.f1(:,j)), mean(S.prec(:,j)), mean(S.rec(:,j)));
pairLine(say, 'relax 0.20 vs disk r=15', S.iou(:,j) - S.disk.iou);
pairLine(say, 'relax 0.20 vs levelset ', S.iou(:,j) - lsIou);
say('  fallback fired (image changed vs disk): %d of %d images\n', nnz(abs(S.iou(:,j) - S.disk.iou) > 1e-12), numel(S.images));
say('  per dataset:\n');
for d = unique(S.dset)'
    s = S.dset == d;
    say('    %-7s n=%-3d levelset %.4f | disk %.4f | relax %.4f  (%+.4f vs disk)\n', d, nnz(s), ...
        mean(lsIou(s)), mean(S.disk.iou(s)), mean(S.iou(s,j)), mean(S.iou(s,j) - S.disk.iou(s)));
end
say('  per fold (relax - disk):');
for f = 1:max(S.fold), say(' %+.4f', mean(S.iou(S.fold==f,j) - S.disk.iou(S.fold==f))); end
say('\n  sweep reproduced: %.4f (fallback.csv reported 0.5097)\n\n', mean(S.iou(:,j)));

% ---- B. cross-fitted selection -------------------------------------------
say('B. Cross-fitted floor selection over {%s}\n', num2str(floors));
cf = nan(size(S.fold));
for f = 1:max(S.fold)
    tr = S.fold ~= f;
    [~, b] = max(mean(S.iou(tr, :), 1));
    cf(~tr) = S.iou(~tr, b);
    say('  fold %d: chose %.2f (train mean %.4f) -> held-out %.4f vs disk %.4f\n', f, floors(b), ...
        mean(S.iou(tr, b)), mean(S.iou(~tr, b)), mean(S.disk.iou(~tr)));
end
say('  cross-fitted mean IoU %.4f\n', mean(cf));
pairLine(say, 'cross-fitted vs disk r=15', cf - S.disk.iou);
say('\n');

% ---- C. cross-device selection -------------------------------------------
say('C. Choose on one device, score the other\n');
ds = unique(S.dset)';
for d = ds
    tr = S.dset ~= d;
    [~, b] = max(mean(S.iou(tr, :), 1));
    say('  choose on %-7s -> %.2f | score %-7s: %.4f vs disk %.4f (%+.4f)\n', strjoin(ds(ds ~= d), '+'), ...
        floors(b), d, mean(S.iou(~tr, b)), mean(S.disk.iou(~tr)), mean(S.iou(~tr, b) - S.disk.iou(~tr)));
end
say('\n');

% ---- D. replication on the 41-image arm (different networks) -------------
c2 = cnngvf.arm(cnngvf.defaults(), "unclustered_boundary");
S2 = scoreArm(c2, PRE);
say('D. Replication, %s (%d DRAC22 images, DRAC22-only fold networks)\n', c2.Arm, numel(S2.images));
say('  disk r=15 %.4f | relax 0.20 %.4f\n', mean(S2.disk.iou), mean(S2.iou));
pairLine(say, 'relax 0.20 vs disk r=15', S2.iou - S2.disk.iou);

T = table(S.images, S.dset, S.fold, lsIou, S.disk.iou, cf, 'VariableNames', ...
    {'Image', 'Dataset', 'Fold', 'Levelset', 'DiskR15', 'CrossFitted'});
for k = 1:numel(floors)
    T.(sprintf('Relax_%03d', round(floors(k)*100))) = S.iou(:, k);
end
writetable(T, fullfile(outDir, 'confirm_per_image.csv'));
fclose(fid);
fprintf('Records: %s\n', outDir);
end


function S = scoreArm(cfg, floors)
cfg.Post.mode = "threshold";
cfg.EvalMode  = "image";
cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
M = cnngvf.manifest(cfg);
images = cnngvf.probCache("list", cacheDir);
images = images(ismember(images, M.Name));
[names, seedRows] = cnngvf.loadPoints(cfg);
n = numel(images); nf = numel(floors);
[S.iou, S.f1, S.prec, S.rec] = deal(nan(n, nf));
[S.disk.iou, S.disk.f1, S.disk.prec, S.disk.rec] = deal(nan(n, 1));
S.fold = nan(n, 1); S.dset = strings(n, 1); S.images = images;
for i = 1:n
    k = find(M.Name == images(i), 1);
    seeds = seedRows(names == images(i), :);
    got = cnngvf.probCache("read", cacheDir, images(i), cfg);   % leakage-guarded
    P = double(got.P);
    gt = rnv.loadMask(M.GtPath(k));
    S.fold(i) = got.meta.fold; S.dset(i) = M.Dataset(k);

    c = cfg; c.Post.diskFallback = true; c.Post.fallbackMode = "disk";
    m = rnv.metrics(cnngvf.maskFromProb(P, seeds, c), gt);
    S.disk.iou(i) = m.iou; S.disk.f1(i) = m.f1; S.disk.prec(i) = m.precision; S.disk.rec(i) = m.recall;

    c.Post.fallbackMode = "relax";
    for j = 1:nf
        c.Post.relaxFloor = floors(j);
        m = rnv.metrics(cnngvf.maskFromProb(P, seeds, c), gt);
        S.iou(i,j) = m.iou; S.f1(i,j) = m.f1; S.prec(i,j) = m.precision; S.rec(i,j) = m.recall;
    end
    fprintf('  [%2d/%2d] %s\n', i, n, images(i));
end
end


function pairLine(say, label, d)
tol = 1e-12; up = nnz(d > tol); dn = nnz(d < -tol); nn = up + dn;
p = 1;
if nn > 0, p = min(1, 2 * sum(arrayfun(@(i) nchoosek(nn, i), 0:min(up, dn))) / 2^nn); end
rng(0); idx = randi(numel(d), numel(d), 10000); mu = sort(mean(d(idx), 1));
say('  %-26s dIoU %+.4f  %d better / %d worse  sign p %.3f  95%% CI [%+.4f, %+.4f]\n', ...
    label, mean(d), up, dn, p, mu(251), mu(9750));
end


function sayBoth(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end
