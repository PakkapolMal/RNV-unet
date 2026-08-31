function [perUnit, summary] = run_eval(resultDir, cfg)
%RUN_EVAL Score masks that run_batch already wrote, without re-segmenting.
%
%  run_eval                              % score cfg.Paths.Output, per segment
%  run_eval(resultDir)                   % score another run's directory
%  run_eval(resultDir, cfg)              % cfg.EvalMode picks the unit
%  [perUnit, summary] = run_eval(...)
%
%  run_batch welds segmentation to scoring, so changing how results are scored
%  used to mean a full GPU pass over every image - 205 s to reproduce
%  pixel-identical masks. This reads combined/ and segment/ off disk and scores
%  them in about a second.
%
%  Use it to re-score after changing cfg.EvalMode, after correcting ground
%  truth, or to score one run's masks under another run's protocol.
%
%  Seeds are recovered from the segment/ filenames (<stem>-<x>-<y>.png), not
%  from the localization CSV. The masks and the coordinates that scored them
%  then cannot disagree, however the CSV has changed since - which matters
%  here, because the localization files have been revised twice.
%
%  Writes per_<mode>.csv and run_summary.txt into resultDir, exactly as
%  run_batch does.
%
%  See also RUN_BATCH, CNNGVF.EVALUATE, CNNGVF.WRITESUMMARY.

if nargin < 2
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);   % run_eval(dir, "DRACTUH_unclustered")
end
if nargin < 1 || isempty(resultDir)
    resultDir = cfg.Paths.Output;
end
cfg.Paths.Output = char(resultDir);

combinedDir = fullfile(cfg.Paths.Output, 'combined');
segmentDir  = fullfile(cfg.Paths.Output, 'segment');

if ~isfolder(combinedDir)
    error('cnngvf:runEval:noMasks', ...
        'No combined/ masks in %s. Run run_batch there first.', cfg.Paths.Output);
end

d = dir(fullfile(combinedDir, '*.png'));
if isempty(d)
    error('cnngvf:runEval:noMasks', 'No masks found in %s', combinedDir);
end

fprintf('run_eval | %s | %d images | mode: %s\n', cfg.Paths.Output, numel(d), cfg.EvalMode);

results(numel(d)) = struct('image', "", 'seeds', [], 'perSeed', {{}}, 'mask', []);
for i = 1:numel(d)
    [~, stem] = fileparts(d(i).name);
    results(i).image = string(stem) + ".png";
    results(i).mask  = imread(fullfile(combinedDir, d(i).name)) > 0;

    [seeds, masks] = readSegments(segmentDir, stem);
    results(i).seeds   = seeds;
    results(i).perSeed = masks;
end

[perUnit, summary] = cnngvf.evaluate(results, cfg);

csvName = "per_" + cfg.EvalMode + ".csv";
writetable(perUnit, fullfile(cfg.Paths.Output, csvName));
cnngvf.writeSummary(fullfile(cfg.Paths.Output, 'run_summary.txt'), ...
    cfg, perUnit, summary, []);

if summary.nUnits > 0
    fprintf('Scored %d %ss: mean IoU %.4f (median %.4f), mean F1 %.4f (median %.4f)\n', ...
        summary.nUnits, cfg.EvalMode, summary.meanIou, summary.medianIou, ...
        summary.meanF1, summary.medianF1);
    fprintf('Precision %.4f, Recall %.4f\n', summary.meanPrecision, summary.meanRecall);
    if height(summary.perDataset) > 1
        for i = 1:height(summary.perDataset)
            fprintf('  %-8s n=%-3d IoU %.4f  F1 %.4f\n', ...
                summary.perDataset.Dataset(i), summary.perDataset.GroupCount(i), ...
                summary.perDataset.mean_Iou(i), summary.perDataset.mean_F1(i));
        end
    end
else
    fprintf('Nothing scored - no matching ground truth.\n');
end
fprintf('Records: %s\n', fullfile(cfg.Paths.Output, csvName));
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [seeds, masks] = readSegments(segmentDir, stem)
% Recover this image's per-seed masks and the coordinates they were written
% under. The trailing two integers are x and y; anything before them is the
% stem, so an image name containing a hyphen still parses.
seeds = zeros(0, 2);
masks = {};

if ~isfolder(segmentDir)
    return;
end

d = dir(fullfile(segmentDir, [stem '-*.png']));
for i = 1:numel(d)
    tok = regexp(d(i).name, '^(.*)-(\d+)-(\d+)\.png$', 'tokens', 'once');
    if isempty(tok) || ~strcmp(tok{1}, stem)
        continue;   % a different image whose stem happens to be a prefix
    end
    seeds(end + 1, :) = [str2double(tok{2}), str2double(tok{3})];  %#ok<AGROW>
    masks{end + 1, 1} = imread(fullfile(segmentDir, d(i).name)) > 0; %#ok<AGROW>
end
end
