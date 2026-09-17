function T = run_seedless_compare(seededOutput)
%RUN_SEEDLESS_COMPARE Seeded vs seedless P, paired, off both caches. No GPU.
%
%  run_seedless_compare                   % seeded cache at result/DRACTUH_unclustered
%  run_seedless_compare(seededOutput)     % ... or wherever that folder now lives
%  T = run_seedless_compare(...)
%
%  ../docs/SEEDLESS.md. Scores every image present in BOTH caches - 14 after the
%  fold-1 pilot, 68 after the full run - through the identical reduced pipeline
%  (threshold 0.5, relax fallback 0.20, imclose, imfill, seedMasks), so the
%  source of P is the only variable. Both caches are read through
%  cnngvf.probCache, which refuses in-sample P.
%
%  Per image:
%    headline   IoU / F1 / precision / recall, seeded vs seedless
%    coverage   GT px with P > tau, and GT components touched by P > tau
%    outside    share of P > tau px lying off GT (the precision risk, pre-claiming)
%    unclaimed  GT px seedless P covers that no seed claims (the "seeds decide" cost)
%    missed     seeds whose own pixel has P <= tau, i.e. where the fallback fires
%
%  Secondary, reported as curves only: tau 0.30-0.80 for both, and the relax
%  floor 0.05-0.30 on seedless P (cross-fitted when more than one fold is cached).
%
%  With one fold cached it also evaluates the pilot gate written in SEEDLESS.md.
%  Writes result/DRACTUH_seedless/compare/.
%
%  See also RUN_CV, RUN_CONFIRM_RELAX, CNNGVF.MASKFROMPROB.

cs = cnngvf.arm(cnngvf.defaults(), "DRACTUH_seedless");
cfgD = cnngvf.arm(cnngvf.defaults(), "DRACTUH_unclustered");
cfgD.Post = cs.Post;               % identical post-processing: the reduced pipeline + relax 0.20
if nargin >= 1, cfgD.Paths.Output = char(seededOutput); end
tau = cs.Post.tau;

cacheS = fullfile(cs.Paths.Output, 'cv', 'probcache');
cacheD = fullfile(cfgD.Paths.Output, 'cv', 'probcache');
outDir = fullfile(cs.Paths.Output, 'compare');
if ~exist(outDir, 'dir'), mkdir(outDir); end

M = cnngvf.manifest(cs);
[names, seedRows] = cnngvf.loadPoints(cs);
images = intersect(cnngvf.probCache("list", cacheS), cnngvf.probCache("list", cacheD), 'stable');
images = images(ismember(images, M.Name));
assert(~isempty(images), 'cnngvf:seedlessCompare:empty', 'No image is cached in both %s and %s.', cacheS, cacheD);

taus   = 0.30:0.05:0.80;
floors = 0.05:0.05:0.30;
n = numel(images);
cols = {'Iou', 'F1', 'Prec', 'Rec', 'CovPx', 'CovComp', 'Outside', 'Missed'};
R = struct();
for src = ["Seeded", "Seedless"]
    for c = cols, R.(src + c{1}) = nan(n, 1); end
    R.(src + "Tau") = nan(n, numel(taus));
end
unclaimed = nan(n, 1); floorIou = nan(n, numel(floors));
fold = nan(n, 1); dset = strings(n, 1); nSeeds = nan(n, 1);

for i = 1:n
    k  = find(M.Name == images(i), 1);
    s  = seedRows(names == images(i), :);
    gt = rnv.loadMask(M.GtPath(k));
    dset(i) = M.Dataset(k); nSeeds(i) = size(s, 1);
    gS = cnngvf.probCache("read", cacheS, images(i), cs);
    gD = cnngvf.probCache("read", cacheD, images(i), cfgD);
    assert(gS.meta.fold == gD.meta.fold, 'fold mismatch on %s', images(i));
    fold(i) = gS.meta.fold;

    CC  = bwconncomp(gt);
    idx = sub2ind(size(gt), min(max(round(s(:, 2)), 1), size(gt, 1)), min(max(round(s(:, 1)), 1), size(gt, 2)));
    for src = ["Seeded", "Seedless"]
        if src == "Seeded", P = double(gD.P); c = cfgD; else, P = double(gS.P); c = cs; end
        mask = cnngvf.maskFromProb(P, s, c);
        m = rnv.metrics(mask, gt);
        on = P > tau;
        R.(src + "Iou")(i) = m.iou;  R.(src + "F1")(i)  = m.f1;
        R.(src + "Prec")(i) = m.precision; R.(src + "Rec")(i) = m.recall;
        R.(src + "CovPx")(i)   = nnz(on & gt) / nnz(gt);
        R.(src + "CovComp")(i) = mean(cellfun(@(px) any(on(px)), CC.PixelIdxList));
        R.(src + "Outside")(i) = nnz(on & ~gt) / max(nnz(on), 1);
        R.(src + "Missed")(i)  = nnz(~on(idx));
        for t = 1:numel(taus)
            ct = c; ct.Post.tau = taus(t);
            R.(src + "Tau")(i, t) = rnv.metrics(cnngvf.maskFromProb(P, s, ct), gt).iou;
        end
        if src == "Seedless"
            unclaimed(i) = nnz(on & gt & ~mask) / nnz(gt);
            for f = 1:numel(floors)
                cf = c; cf.Post.relaxFloor = floors(f);
                floorIou(i, f) = rnv.metrics(cnngvf.maskFromProb(P, s, cf), gt).iou;
            end
        end
    end
    fprintf('  [%2d/%2d] %-12s fold %d | IoU %.4f -> %.4f | GT cover %.3f -> %.3f\n', i, n, ...
        images(i), fold(i), R.SeededIou(i), R.SeedlessIou(i), R.SeededCovPx(i), R.SeedlessCovPx(i));
end

T = table(images, dset, fold, nSeeds, 'VariableNames', {'Image', 'Dataset', 'Fold', 'Seeds'});
for src = ["Seeded", "Seedless"]
    for c = cols, T.(src + c{1}) = R.(src + c{1}); end
end
T.SeedlessUnclaimed = unclaimed;
writetable(T, fullfile(outDir, 'compare_per_image.csv'));

% ---- report ---------------------------------------------------------------
fid = fopen(fullfile(outDir, 'compare_summary.txt'), 'w');
say = @(varargin) sayBoth(fid, varargin{:});
folds = unique(fold)';
say('Seeded vs seedless P - %d images, fold(s) %s, tau %.2f, relax floor %.2f\n', ...
    n, mat2str(folds), tau, cs.Post.relaxFloor);
say('Both through maskFromProb (threshold + relax); only the source of P differs.\n\n');

say('  %-26s %9s %9s %9s\n', '', 'seeded', 'seedless', 'delta');
rows = {'mean IoU', 'Iou'; 'mean F1', 'F1'; 'precision', 'Prec'; 'recall', 'Rec'; ...
        'GT px covered by P>tau', 'CovPx'; 'GT components touched', 'CovComp'; ...
        'P>tau px off GT', 'Outside'};
for r = 1:size(rows, 1)
    a = mean(R.("Seeded" + rows{r, 2})); b = mean(R.("Seedless" + rows{r, 2}));
    say('  %-26s %9.4f %9.4f %+9.4f\n', rows{r, 1}, a, b, b - a);
end
say('  %-26s %9d %9d %+9d  (of %d)\n', 'seeds with P<=tau (fallback)', ...
    sum(R.SeededMissed), sum(R.SeedlessMissed), sum(R.SeedlessMissed) - sum(R.SeededMissed), sum(nSeeds));
say('  %-26s %9s %9.4f\n', 'GT px covered, unclaimed', '', mean(unclaimed));
say('  median IoU %.4f -> %.4f\n', median(R.SeededIou), median(R.SeedlessIou));
[p, up, dn] = signTest(R.SeedlessIou - R.SeededIou); ci = bootCI(R.SeedlessIou - R.SeededIou);
say('  paired dIoU %+.4f | %d better / %d worse | sign p %.3f | 95%% CI [%+.4f, %+.4f]\n\n', ...
    mean(R.SeedlessIou - R.SeededIou), up, dn, p, ci(1), ci(2));

say('Per dataset, mean IoU (seeded -> seedless)\n');
for d = unique(dset)'
    sel = dset == d;
    say('  %-7s n=%-3d %.4f -> %.4f   recall %.4f -> %.4f   precision %.4f -> %.4f\n', d, nnz(sel), ...
        mean(R.SeededIou(sel)), mean(R.SeedlessIou(sel)), mean(R.SeededRec(sel)), mean(R.SeedlessRec(sel)), ...
        mean(R.SeededPrec(sel)), mean(R.SeedlessPrec(sel)));
end
say('Per fold, mean IoU (seeded -> seedless)\n');
for f = folds
    sel = fold == f;
    say('  fold %d n=%-3d %.4f -> %.4f\n', f, nnz(sel), mean(R.SeededIou(sel)), mean(R.SeedlessIou(sel)));
end

d = R.SeedlessIou - R.SeededIou;
[~, ord] = sort(d);
say('\nBiggest movers (seedless - seeded)\n');
for j = unique([ord(1:min(5, n))', flip(ord(max(1, n - 4):n))'], 'stable')
    say('  %-12s %-6s seeds %3d  %.4f -> %.4f (%+.4f)  cover %.3f -> %.3f\n', images(j), dset(j), ...
        nSeeds(j), R.SeededIou(j), R.SeedlessIou(j), d(j), R.SeededCovPx(j), R.SeedlessCovPx(j));
end

say('\ntau curve, mean IoU (A CURVE - never quote its maximum)\n  tau    ');
say('  %.2f ', taus); say('\n  seeded  '); say(' %.4f', mean(R.SeededTau, 1)); say('\n  seedless'); say(' %.4f', mean(R.SeedlessTau, 1)); say('\n');
say('relax floor on seedless P (a curve)\n  floor   '); say('  %.2f  ', floors); say('\n  IoU     '); say(' %.4f ', mean(floorIou, 1)); say('\n');
if numel(folds) > 1
    cf = nan(n, 1);
    for f = folds
        tr = fold ~= f;
        [~, b] = max(mean(floorIou(tr, :), 1));
        cf(~tr) = floorIou(~tr, b);
    end
    say('  cross-fitted floor, seedless: %.4f\n', mean(cf));
end

if isequal(folds, 1)
    covUp = mean(R.SeedlessCovPx) > mean(R.SeededCovPx);
    say('\nPILOT GATE (SEEDLESS.md, written before running)\n');
    say('  1. seedless fold-1 IoU >= 0.39            : %.4f  -> %s\n', mean(R.SeedlessIou), passFail(mean(R.SeedlessIou) >= 0.39));
    say('  2. GT coverage above seeded P             : %.4f vs %.4f -> %s\n', mean(R.SeedlessCovPx), mean(R.SeededCovPx), passFail(covUp));
    say('  decision: %s\n', ternary(mean(R.SeedlessIou) >= 0.39 && covUp, ...
        'CONTINUE to folds 2-5', 'STOP - do not run the remaining folds'));
    say('  (14 images gate a decision; they are not a result.)\n');
end
fclose(fid);
fprintf('Records: %s\n', outDir);
end


function s = passFail(ok), s = ternary(ok, 'PASS', 'FAIL'); end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end

function [p, up, dn] = signTest(d)
up = nnz(d > 1e-12); dn = nnz(d < -1e-12); nn = up + dn; p = 1;
if nn > 0, p = min(1, 2 * sum(arrayfun(@(i) nchoosek(nn, i), 0:min(up, dn))) / 2^nn); end
end

function ci = bootCI(d)
rng(0); mu = sort(mean(d(randi(numel(d), numel(d), 10000)), 1));
ci = [mu(251), mu(9750)];
end

function sayBoth(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end
