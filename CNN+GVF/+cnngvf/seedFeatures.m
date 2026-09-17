function F = seedFeatures(P, seeds, cfg)
%CNNGVF.SEEDFEATURES Per-seed evidence from P and the seed layout. No ground truth.
%
%  F = cnngvf.seedFeatures(P, seeds, cfg)
%
%  P     : probability map (cached or fresh)
%  seeds : S x 2 [cx cy]
%  cfg   : reads Post.tau and whatever cnngvf.maskFromProb reads
%
%  F : S x 1 table, one row per seed
%      fired            P(cy,cx) > Post.tau - the same trigger as the fallback
%      peakP_r16/_r32   max P in a disk of radius 16 / 32 px around the seed
%      meanP_r16        mean P in the radius-16 disk
%      support_R64/100/150  number of OTHER seeds within R px
%      nnDist           distance to the nearest other seed (Inf if alone)
%      regionArea       area of the component this seed claims in the current
%      regionMeanP      chain (cnngvf.maskFromProb with cfg), and its mean P
%      distToFiredComp  px from the seed to the nearest component claimed by a
%                       fired seed (0 on one, Inf if no seed fired)
%      imageSeedCount   seeds in this image
%
%  Production-safe by construction: P, seeds and cfg in, nothing else read. The
%  point is a seed gate (../docs/REFINE.md step 3) that can run at inference, so
%  every feature here has to be one inference can compute.
%
%  See also CNNGVF.MASKFROMPROB, RUN_REFINE.

S = size(seeds, 1);
[h, w] = size(P);
row = min(max(round(seeds(:, 2)), 1), h);
col = min(max(round(seeds(:, 1)), 1), w);

fired = P(sub2ind([h w], row, col)) > cfg.Post.tau;

[peak16, peak32, mean16] = deal(zeros(S, 1));
for s = 1:S
    [peak16(s), mean16(s)] = diskStats(P, row(s), col(s), 16);
    peak32(s) = diskStats(P, row(s), col(s), 32);
end

D = hypot(seeds(:, 1) - seeds(:, 1)', seeds(:, 2) - seeds(:, 2)');
D(1:S+1:end) = Inf;                 % "other" seeds only
nnDist = min(D, [], 2);
if S <= 1
    nnDist = inf(S, 1);
end

[~, perSeed] = cnngvf.maskFromProb(P, seeds, cfg);
area  = cellfun(@nnz, perSeed);
meanP = zeros(S, 1);
firedBW = false(h, w);
for s = 1:S
    if area(s) > 0
        meanP(s) = mean(P(perSeed{s}));
        if fired(s)
            firedBW = firedBW | perSeed{s};
        end
    end
end
distFired = inf(S, 1);
if any(firedBW(:))
    dt = bwdist(firedBW);
    distFired = dt(sub2ind([h w], row, col));
end

F = table(fired, peak16, peak32, mean16, sum(D <= 64, 2), sum(D <= 100, 2), sum(D <= 150, 2), ...
    nnDist, area, meanP, distFired, repmat(S, S, 1), 'VariableNames', {'fired', 'peakP_r16', ...
    'peakP_r32', 'meanP_r16', 'support_R64', 'support_R100', 'support_R150', 'nnDist', ...
    'regionArea', 'regionMeanP', 'distToFiredComp', 'imageSeedCount'});
end


function [pk, mu] = diskStats(P, r, c, rad)
[h, w] = size(P);
r0 = max(r - rad, 1); r1 = min(r + rad, h);
c0 = max(c - rad, 1); c1 = min(c + rad, w);
[cc, rr] = meshgrid(c0:c1, r0:r1);
in = (rr - r).^2 + (cc - c).^2 <= rad^2;
v = P(r0:r1, c0:c1);
v = v(in);
pk = max(v);
mu = mean(v);
end
