function M = manifest(cfg)
%CNNGVF.MANIFEST The image domain: one row per image, across every dataset.
%
%  M = cnngvf.manifest(cfg)
%
%  M : table with one row per image in the active datasets
%        Name     "458.png" / "nv (1).png"  - the key everything else uses
%        Dataset  "DRAC22" / "TUH"
%        ImgPath  absolute path to the image
%        GtPath   absolute path to its whole-image ground-truth mask
%        GtPx     ground-truth foreground area, for area stratification
%
%  This exists because an image used to BE its filename: one string keyed the
%  image file, the mask, the CSV rows and every output name, and every stage
%  rebuilt those paths with its own fullfile call. That held only while image
%  and mask shared a name in one directory each. TUH pairs TUH_raw/nv (N).png
%  with TUH_filled_gt/gt (N).tif, so the mapping is per-dataset data
%  (Ds.GtFrom / Ds.GtTo) resolved here, once.
%
%  Every image is in the domain, and every image must have a mask: an image
%  without one is an error, never a silent skip (CLAUDE.md, "Datasets").
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

    % The domain is defined by the IMAGES, not by the mask directory:
    % DRAC22_filled_gt also holds a stray fill.py.
    f = dir(fullfile(Ds.Img, '*.png'));
    for i = 1:numel(f)
        name   = string(f(i).name);
        gtPath = fullfile(Ds.GT, regexprep(name, Ds.GtFrom, Ds.GtTo));
        if ~isfile(gtPath)
            error('cnngvf:manifest:noGt', ...
                'Dataset "%s": %s has no mask at %s', Ds.Name, name, gtPath);
        end
        rows(end + 1, :) = {name, Ds.Name, ...
            string(fullfile(Ds.Img, char(name))), string(gtPath), ...
            nnz(cnngvf.loadMask(gtPath))}; %#ok<AGROW>
    end
end

if isempty(rows)
    error('cnngvf:manifest:empty', ...
        'No images in the active datasets (%s).', ...
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

