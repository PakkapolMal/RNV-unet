function p = segGtFile(cfg, imgName, seed)
%CNNGVF.SEGGTFILE Path of the per-seed ground-truth segment for one seed.
%
%  p = cnngvf.segGtFile(cfg, imgName, seed)
%
%  imgName : "458.png"
%  seed    : [cx cy] straight from the localization CSV
%  p       : .../SegmentFilledGT/458-631-641.png
%
%  separate_segments.py names each segment by the seed that claimed it, so the
%  seed coordinates are the key - no ordering assumption, no segment index to
%  keep in step with the CSV. Coordinates are the integers in the CSV, so they
%  are rounded rather than formatted as floats.
%
%  The file is missing when the seed fell outside every ground-truth component
%  (separate_segments.py reports those as "skipped"). Callers use isfile.
%
%  See also CNNGVF.EVALUATE, CNNGVF.PREPAREDATA.

[~, stem] = fileparts(char(imgName));
p = fullfile(cfg.Paths.SegGT, ...
    sprintf('%s-%d-%d.png', stem, round(seed(1)), round(seed(2))));
end
