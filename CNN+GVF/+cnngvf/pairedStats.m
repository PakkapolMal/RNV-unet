function S = pairedStats(d, names)
%CNNGVF.PAIREDSTATS Paired tests on per-image deltas, no Statistics Toolbox.
%
%  S = cnngvf.pairedStats(d, names)
%
%  d     : n x 1 per-image delta (arm - comparator), usually IoU
%  names : n x 1 string, image names, for the movers
%
%  S.mean, S.up, S.down        mean delta; better / worse counts (|d| > 1e-12)
%  S.signP                     exact two-sided sign test, ties excluded
%  S.wilcoxonP                 signed-rank, NORMAL APPROXIMATION with tie correction
%  S.ci                        percentile bootstrap 95% CI on the mean, 10 000
%                              resamples, rng(20260917) - ../docs/REFINE.md
%  S.gains, S.losses           the five largest gains / regressions, table Image, Delta
%
%  The tests are lifted from run_ablate's local functions. run_ablate keeps its
%  own copies, seeded rng(0), so its recorded intervals do not move.
%
%  See also RUN_REFINE, RUN_ABLATE.

d = d(:);
names = string(names(:));
tol = 1e-12;

S.mean = mean(d);
S.up   = nnz(d >  tol);
S.down = nnz(d < -tol);
S.signP     = signTest(S.up, S.down);
S.wilcoxonP = wilcoxonP(d(abs(d) > tol));
S.ci        = bootstrapCI(d);

[~, o] = sort(d, 'descend');
g = o(1:min(5, numel(o)));
g = g(d(g) > tol);
S.gains = table(names(g), d(g), 'VariableNames', {'Image', 'Delta'});
[~, o] = sort(d, 'ascend');
l = o(1:min(5, numel(o)));
l = l(d(l) < -tol);
S.losses = table(names(l), d(l), 'VariableNames', {'Image', 'Delta'});
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function p = signTest(up, dn)
n = up + dn;
if n == 0
    p = 1;
    return;
end
p = min(1, 2 * sum(arrayfun(@(i) nchoosek(n, i), 0:min(up, dn))) / 2^n);
end


function ci = bootstrapCI(d)
n = numel(d);
if n == 0 || all(abs(d) < 1e-12)
    ci = [0 0];
    return;
end
old = rng(20260917);
restore = onCleanup(@() rng(old));
means = mean(d(randi(n, n, 10000)), 1);
ci = [pct(means, 0.025), pct(means, 0.975)];
end


function q = pct(x, p)
% Linear interpolation, as run_ablate's. prctile is Statistics Toolbox.
x = sort(x(:));
n = numel(x);
h = (n - 1) * p + 1;
lo = floor(h); hi = ceil(h);
q = x(lo) + (h - lo) * (x(hi) - x(lo));
end


function p = wilcoxonP(d)
n = numel(d);
if n < 2
    p = 1;
    return;
end
r = avgRank(abs(d));
W = sum(r(d > 0));
mu = n * (n + 1) / 4;
[~, ~, g] = unique(abs(d));
tieCorr = sum(arrayfun(@(k) sum(g == k)^3 - sum(g == k), unique(g))) / 48;
sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24 - tieCorr);
if sigma == 0
    p = 1;
    return;
end
z = (abs(W - mu) - 0.5) / sigma;   % continuity correction
p = min(1, max(0, erfc(z / sqrt(2))));
end


function r = avgRank(x)
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
