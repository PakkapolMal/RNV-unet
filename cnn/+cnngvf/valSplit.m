function val = valSplit(cfg, exclude)
%CNNGVF.VALSPLIT Derive the validation image list for a single-split run.
%
%  val = cnngvf.valSplit(cfg)
%  val = cnngvf.valSplit(cfg, exclude)   % omit these images first (run_cv)
%
%  Returns cfg.Split.NumVal image names, spread across the ground-truth area
%  range and across datasets.
%
%  Derived, never written down. The old default was four DRAC22 filenames from
%  the 17-image era; it then survived the dataset growing to 41 and a second
%  dataset arriving, still validating on the same four images and on none of
%  TUH. A literal list cannot notice that it has gone stale.
%
%  Interior evenly-spaced positions on the area-sorted list, not the head:
%  sorted descending, the head is always the largest lesions, which in a
%  one-epoch dry run made a fold keep its untrained network and predict 0 px on
%  every held-out image. The same trap applies to a single split, so the same
%  fix is used - this is cnngvf.cvFolds's inner-validation rule, shared rather
%  than duplicated.
%
%  Selection is per dataset, so a two-dataset domain never validates on only
%  one of them - which is what a pooled area sort would do whenever one
%  dataset's lesions are systematically larger.
%
%  See also CNNGVF.ARM, CNNGVF.CVFOLDS, CNNGVF.PREPAREDATA.

if nargin < 2
    exclude = strings(0, 1);
end

M = cnngvf.manifest(cfg);
if ~isempty(exclude)
    M = M(~ismember(M.Name, string(exclude)), :);
end

wanted   = cfg.Split.NumVal;
datasets = unique(M.Dataset, 'stable');
val      = strings(0, 1);

% Share the budget over datasets in proportion to their size, at least one each
% so no dataset is unrepresented.
share = zeros(numel(datasets), 1);
for d = 1:numel(datasets)
    share(d) = max(1, round(wanted * nnz(M.Dataset == datasets(d)) / height(M)));
end

for d = 1:numel(datasets)
    sub = sortrows(M(M.Dataset == datasets(d), :), 'GtPx', 'descend');
    n   = height(sub);
    nv  = min(share(d), n - 1);      % never take a whole dataset
    if nv < 1
        continue;
    end
    pos = round(linspace(1, n, nv + 2));
    val = [val; sub.Name(pos(2:end-1))];   %#ok<AGROW>
end

val = unique(val, 'stable');

if isempty(val) || numel(val) >= height(M)
    error('cnngvf:valSplit:degenerate', ...
        ['Cannot derive a validation split: %d image(s) in the domain, %d selected. ' ...
         'Lower cfg.Split.NumVal or widen the domain.'], height(M), numel(val));
end
end
