function M = manifest(cfg)
%CNNGVF.MANIFEST The image domain: one row per image, across every dataset.
%
%  M = cnngvf.manifest(cfg)
%
%  M : table with one row per ground-truth image in the active datasets
%        Name     "458.png" / "nv (1).png"  - the key everything else uses
%        Dataset  "DRAC22" / "TUH"
%        ImgPath  absolute path to the image
%        GtPath   absolute path to its whole-image ground-truth mask
%        GtPx     ground-truth foreground area, for area stratification
%
%  This exists because an image used to BE its filename: one string keyed the
%  image file, the mask, the CSV rows and every output name, and every stage
%  rebuilt those paths with its own fullfile call. That held only while image
%  and mask shared a stem in one directory each. TUH pairs nvImages/nv (N).png
%  with nvGtImages/gt (N).png - different stem, different directory - so the
%  mapping is now per-dataset data (Ds.GtFrom / Ds.GtTo) resolved here, once,
%  instead of seven times inline.
%
%  Only datasets named in cfg.ActiveDatasets are included, so an arm can run on
%  one dataset or on all of them without any other file knowing which.
%
%  Names must be unique across datasets - they are the join key for the
%  localization CSVs and for the output filenames - so a collision is an error
%  here rather than a mask silently scored against the wrong image.
%
%  GtPx is read once and cached with the row because cnngvf.cvFolds stratifies
%  on it and would otherwise re-read every mask.
%
%  See also CNNGVF.LOADPOINTS, CNNGVF.ARM, CNNGVF.CVFOLDS, CNNGVF.PREPAREDATA.

active = cfg.ActiveDatasets;
rows   = cell(0, 5);

for d = 1:numel(cfg.Datasets)
    Ds = cfg.Datasets(d);
    if ~any(active == Ds.Name)
        continue;
    end
    if ~isfolder(Ds.Img)
        error('cnngvf:manifest:noImageDir', ...
            'Dataset "%s": image directory not found: %s', Ds.Name, Ds.Img);
    end
    if ~isfolder(Ds.GT)
        error('cnngvf:manifest:noGtDir', ...
            'Dataset "%s": ground-truth directory not found: %s', Ds.Name, Ds.GT);
    end

    % The domain is defined by the IMAGES that have a mask, not by the mask
    % directory: RNV-DRAC22GT/FilledGT also holds a stray fill.py, and a
    % dataset may ship masks for images it does not ship.
    f = dir(fullfile(Ds.Img, '*.png'));
    for i = 1:numel(f)
        name   = string(f(i).name);
        gtPath = gtFor(Ds, name);
        if ~isfile(gtPath)
            continue;   % no ground truth: outside the domain
        end
        rows(end + 1, :) = {name, Ds.Name, ...
            string(fullfile(Ds.Img, char(name))), string(gtPath), ...
            nnz(cnngvf.loadMask(gtPath))}; %#ok<AGROW>
    end
end

if isempty(rows)
    error('cnngvf:manifest:empty', ...
        'No images with ground truth in the active datasets (%s).', ...
        strjoin(active, ', '));
end

M = cell2table(rows, 'VariableNames', ...
    {'Name', 'Dataset', 'ImgPath', 'GtPath', 'GtPx'});
M.Name     = string(M.Name);
M.Dataset  = string(M.Dataset);
M.ImgPath  = string(M.ImgPath);
M.GtPath   = string(M.GtPath);

[u, ~, ic] = unique(M.Name);
if numel(u) < height(M)
    dup = u(accumarray(ic, 1) > 1);
    error('cnngvf:manifest:duplicateName', ...
        ['Image name(s) appear in more than one dataset: %s.\n' ...
         'Names are the key for the localization CSVs and the output files, ' ...
         'so they must be unique across the whole domain.'], strjoin(dup', ', '));
end

M = sortrows(M, 'Name');
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function p = gtFor(Ds, imgName)
%GTFOR Ground-truth path for one image, via the dataset's stem rewrite.
%  DRAC22 is the identity ("458.png" -> "458.png"); TUH maps "nv (1)" -> "gt (1)".
[~, stem, ext] = fileparts(char(imgName));
gtStem = regexprep(stem, Ds.GtFrom, Ds.GtTo);
p = fullfile(Ds.GT, [gtStem ext]);
end
