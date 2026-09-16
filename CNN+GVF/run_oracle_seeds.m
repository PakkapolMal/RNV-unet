function run_oracle_seeds(outputDir)
%RUN_ORACLE_SEEDS What a perfect seed filter would be worth. READS GROUND TRUTH.
%
%  run_oracle_seeds                  % DRACTUH_unclustered, its CV fold networks
%  run_oracle_seeds(outputDir)       % an arm output folder that has been moved
%
%  NOT A QUOTABLE SCORE. Seeds are kept or dropped by looking at FilledGT, so every
%  number here is an upper bound on what filtering localization points could buy,
%  used to rank the pipelines under a cleaner seed set - never as a result.
%
%  Seed sets   all  : every point (cached P - the published numbers)
%              in   : seed pixel lies inside FilledGT
%              near : within Patch.seedTolerance px of FilledGT (the training-label rule)
%  P source    fresh inference with the kept seeds (seeds ARE the patch centres and
%              the heat-map, so dropping one changes P), plus near-with-cached-P to
%              split the gain into "better P" vs "fewer claims".
%  Post        levelset (normal), threshold+disk r15 (reduced), threshold+relax 0.20.
%
%  Every image is inferred by the fold network that held it out. GPU, ~19 min.
%  Writes <outputDir>/oracle_seeds/.
%  See also RUN_CONFIRM_RELAX, ../docs/REDUCE_RESULTS.md section 10.
main = fileparts(mfilename('fullpath'));
cfg = cnngvf.arm(cnngvf.defaults(), "DRACTUH_unclustered");
if nargin >= 1, cfg.Paths.Output = char(outputDir); end
cfg.EvalMode = "image";
tol = cfg.Patch.seedTolerance;
cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
outDir = fullfile(cfg.Paths.Output, 'oracle_seeds');
if ~exist(outDir, 'dir'), mkdir(outDir); end

M = cnngvf.manifest(cfg);
folds = cnngvf.cvFolds(cfg);
[names, seedRows] = cnngvf.loadPoints(cfg);

conds = ["all_cachedP", "in_fresh", "near_fresh", "near_cachedP"];
posts = ["levelset", "disk", "relax"];
R = struct('Image', strings(0,1), 'Dataset', strings(0,1), 'Fold', [], 'Cond', strings(0,1), ...
    'Post', strings(0,1), 'SeedsAll', [], 'SeedsKept', [], 'Iou', [], 'F1', [], 'Prec', [], 'Rec', []);
rows = cell(0, 11);
t0 = tic;
for f = 1:numel(folds)
    loaded = load(fullfile(main, 'models', 'cv', sprintf('DRACTUH_unclustered_fold%d.mat', f)), 'net');
    net = loaded.net;
    for name = folds{f}'
        k = find(M.Name == name, 1);
        seeds = seedRows(names == name, :);
        gt = rnv.loadMask(M.GtPath(k));
        [h, w] = size(gt);
        r = min(max(round(seeds(:, 2)), 1), h);
        c = min(max(round(seeds(:, 1)), 1), w);
        idx = sub2ind([h w], r, c);
        dist = bwdist(gt);
        keepIn   = gt(idx);
        keepNear = dist(idx) <= tol;

        got = cnngvf.probCache("read", cacheDir, name, cfg);    % leakage-guarded, fold f
        assert(got.meta.fold == f);
        Pc = double(got.P);
        [~, chans] = cnngvf.loadChannels(M.ImgPath(k), cfg);

        for cond = conds
            switch cond
                case "all_cachedP",  s = seeds;             P = Pc;
                case "in_fresh",     s = seeds(keepIn, :);   P = cnngvf.inferProbMap(chans, s, net, cfg);
                case "near_fresh",   s = seeds(keepNear, :); P = cnngvf.inferProbMap(chans, s, net, cfg);
                case "near_cachedP", s = seeds(keepNear, :); P = Pc;
            end
            for post = posts
                mask = segmentFromP(P, s, cfg, post);
                m = rnv.metrics(mask, gt);
                rows(end+1, :) = {name, M.Dataset(k), f, cond, post, size(seeds,1), size(s,1), ...
                    m.iou, m.f1, m.precision, m.recall}; %#ok<AGROW>
            end
        end
        fprintf('  fold %d %-12s seeds %3d | in %3d | near %3d | %.0f s\n', f, name, ...
            size(seeds,1), nnz(keepIn), nnz(keepNear), toc(t0));
    end
end

T = cell2table(rows, 'VariableNames', fieldnames(R)');
writetable(T, fullfile(outDir, 'oracle_per_image.csv'));

fid = fopen(fullfile(outDir, 'oracle_summary.txt'), 'w');
out = @(varargin) sayBoth(fid, varargin{:});
S = T(T.Cond == "all_cachedP" & T.Post == "disk", :);
out('Oracle seed filter - %s, %d images, tol %d px, %.0f s\n', cfg.Arm, height(S), tol, toc(t0));
out('Seeds kept: all %d | in GT %d | near GT %d\n', sum(S.SeedsAll), ...
    sum(T.SeedsKept(T.Cond == "in_fresh" & T.Post == "disk")), sum(T.SeedsKept(T.Cond == "near_fresh" & T.Post == "disk")));
out('Images left with no seed: in %d | near %d\n\n', ...
    nnz(T.SeedsKept(T.Cond == "in_fresh" & T.Post == "disk") == 0), nnz(T.SeedsKept(T.Cond == "near_fresh" & T.Post == "disk") == 0));
out('  %-13s %-9s    IoU  median      F1    prec     rec |   dIoU   dF1  +/-  signP   95%% CI dIoU\n', 'seeds', 'post');
for post = posts
    base = T(T.Cond == "all_cachedP" & T.Post == post, :);
    for cond = conds
        X = T(T.Cond == cond & T.Post == post, :);
        assert(isequal(X.Image, base.Image));
        d = X.Iou - base.Iou;
        [p, up, dn] = signTest(d); ci = bootCI(d);
        out('  %-13s %-9s %.4f  %.4f  %.4f  %.4f  %.4f | %+.4f %+.4f %2d/%-2d %.4f [%+.4f,%+.4f]\n', ...
            cond, post, mean(X.Iou), median(X.Iou), mean(X.F1), mean(X.Prec), mean(X.Rec), ...
            mean(d), mean(X.F1 - base.F1), up, dn, p, ci(1), ci(2));
    end
    out('\n');
end
out('Per dataset, mean IoU\n');
for ds = unique(T.Dataset)'
    for post = posts
        out('  %-7s %-9s', ds, post);
        for cond = conds
            X = T(T.Cond == cond & T.Post == post & T.Dataset == ds, :);
            out('  %s %.4f', cond, mean(X.Iou));
        end
        out('\n');
    end
end
fclose(fid);
end


function mask = segmentFromP(P, seeds, cfg, post)
if isempty(seeds)
    mask = false(size(P));
    return;
end
switch post
    case "levelset"
        f = cnngvf.buildForce(P, P, cfg);           % probGrad ignores the image
        [u, v] = cnngvf.gvfField(f, cfg);
        BW = cnngvf.drlse(P, cnngvf.initPhi(P, seeds, cfg), u, v, cfg);
        BW = imfill(imclose(BW, cfg.Post.MorphElement), 'holes');
        [~, mask] = cnngvf.seedMasks(BW, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);
    case "disk"
        c = cfg; c.Post.mode = "threshold"; c.Post.fallbackMode = "disk";
        mask = cnngvf.maskFromProb(P, seeds, c);
    case "relax"
        c = cfg; c.Post.mode = "threshold"; c.Post.fallbackMode = "relax"; c.Post.relaxFloor = 0.20;
        mask = cnngvf.maskFromProb(P, seeds, c);
end
end


function [p, up, dn] = signTest(d)
up = nnz(d > 1e-12); dn = nnz(d < -1e-12); n = up + dn; p = 1;
if n > 0, p = min(1, 2 * sum(arrayfun(@(i) nchoosek(n, i), 0:min(up, dn))) / 2^n); end
end


function ci = bootCI(d)
rng(0); mu = sort(mean(d(randi(numel(d), numel(d), 10000)), 1));
ci = [mu(251), mu(9750)];
end


function sayBoth(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end
