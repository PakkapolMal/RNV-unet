function [names, seeds] = loadSeeds(cfg)
%RNV.LOADSEEDS Read the localization CSV into image names and seed points.
%
%  [names, seeds] = rnv.loadSeeds(cfg)
%
%  names : N x 1 string, image file name per seed
%  seeds : N x 2 double, [cx cy] in image coordinates (column, row)
%
%  Rows before cfg.StartRow are dropped. Columns are read positionally, so
%  the CSV header may be named anything as long as the order is
%  (image_name, x, y).
%
%  See also RNV.SEGMENTIMAGE.

T = readtable(cfg.Paths.Localization);

if width(T) < 3
    error('rnv:loadSeeds:badCsv', ...
        'Expected >= 3 columns (name, x, y) in %s', cfg.Paths.Localization);
end

first = max(1, cfg.StartRow);
if first > height(T)
    error('rnv:loadSeeds:startRow', ...
        'StartRow %d exceeds the %d rows in the CSV.', first, height(T));
end
T = T(first:end, :);

names = strtrim(string(T{:, 1}));
seeds = [double(T{:, 2}), double(T{:, 3})];
end
