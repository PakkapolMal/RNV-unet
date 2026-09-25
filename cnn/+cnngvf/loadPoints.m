function [names, seeds] = loadPoints(cfg, which, forTraining)
%CNNGVF.LOADPOINTS Localization points across every active dataset.
%
%  [names, seeds] = cnngvf.loadPoints(cfg)            % cfg.Points
%  [names, seeds] = cnngvf.loadPoints(cfg, "clustered" | "unclustered")
%  [names, seeds] = cnngvf.loadPoints(cfg, "", true)   % the training point set
%
%  forTraining reads a dataset's UnclusteredTrain CSV where one is set - points
%  corrected against ground truth, which training may use and inference must
%  not. Everything else, including every score, reads the detector's output.
%
%  names : N x 1 string, image file name per point
%  seeds : N x 2 double, [cx cy] = (column, row)
%
%  Each dataset keeps its own CSV, so this reads one per active dataset and
%  concatenates. Columns are read positionally as (image_name, x, y), so a
%  header rename is harmless but a column reorder is not.
%
%  Points naming an image outside the manifest are dropped, with a count. That
%  is normal rather than exceptional: a localization CSV covers whatever the
%  detector was run on, which is not necessarily the set that has ground truth.
%
%  "clustered" points are DBSCAN representatives, one per lesion; "unclustered"
%  are the raw detections behind them. Only DRAC22 has per-seed ground truth
%  keyed to its clustered coordinates, which is why cnngvf.evaluate fences
%  EvalMode "segment" to that dataset.
%
%  See also CNNGVF.MANIFEST, CNNGVF.ARM.

if nargin < 2 || strlength(which) == 0
    which = cfg.Points;
end
if nargin < 3
    forTraining = false;
end
which = lower(string(which));

switch which
    case "clustered",   field = 'Clustered';
    case "unclustered", field = 'Unclustered';
    otherwise
        error('cnngvf:loadPoints:badSet', ...
            'Unknown point set "%s". Use "clustered" or "unclustered".', which);
end

names = strings(0, 1);
seeds = zeros(0, 2);

for d = 1:numel(cfg.Datasets)
    Ds = cfg.Datasets(d);
    if ~any(cfg.ActiveDatasets == Ds.Name)
        continue;
    end

    csv = Ds.(field);
    if forTraining && field == "Unclustered" && strlength(Ds.UnclusteredTrain) > 0
        csv = Ds.UnclusteredTrain;
    end
    if strlength(csv) == 0
        error('cnngvf:loadPoints:noCsvField', ...
            'Dataset "%s" defines no %s point set.', Ds.Name, which);
    end
    if ~isfile(csv)
        error('cnngvf:loadPoints:noCsv', ...
            'Dataset "%s": %s points not found at %s', Ds.Name, which, csv);
    end

    T = readtable(csv);
    if width(T) < 3
        error('cnngvf:loadPoints:badCsv', ...
            'Expected >= 3 columns (name, x, y) in %s', csv);
    end
    names = [names; strtrim(string(T{:, 1}))];                 %#ok<AGROW>
    seeds = [seeds; double(T{:, 2}), double(T{:, 3})];         %#ok<AGROW>
end

% Keep only points whose image is in the domain.
M    = cnngvf.manifest(cfg);
keep = ismember(names, M.Name);
if ~all(keep)
    dropped = unique(names(~keep));
    fprintf(['Dropped %d point(s) on %d image(s) outside the domain ' ...
             '(no ground truth, or not in the active datasets): %s\n'], ...
        nnz(~keep), numel(dropped), truncate(dropped));
    names = names(keep);
    seeds = seeds(keep, :);
end

if isempty(names)
    error('cnngvf:loadPoints:none', ...
        'No %s points fall on any image in the domain (%s).', ...
        which, strjoin(cfg.ActiveDatasets, ', '));
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function s = truncate(items)
if numel(items) > 6
    s = strjoin(items(1:6)', ', ') + sprintf(", ... (+%d more)", numel(items) - 6);
else
    s = strjoin(items', ', ');
end
end
