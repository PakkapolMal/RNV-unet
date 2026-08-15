function writeSummary(txtPath, cfg, perUnit, summary, elapsed)
%CNNGVF.WRITESUMMARY Write the aggregate scores and the config behind them.
%
%  cnngvf.writeSummary(txtPath, cfg, perUnit, summary, elapsed)
%
%  elapsed : per-image segmentation times, or [] when scoring cached masks
%
%  Shared by run_batch and run_eval so a re-scored run is recorded in exactly
%  the same format as the run that produced the masks.
%
%  See also CNNGVF.EVALUATE, RUN_BATCH, RUN_EVAL.

fid = fopen(txtPath, 'w');
if fid < 0
    warning('cnngvf:writeSummary:noWrite', 'Could not write %s', txtPath);
    return;
end
cleanup = onCleanup(@() fclose(fid));

byImage = cfg.EvalMode == "image";

fprintf(fid, 'RNV CNN+GVF segmentation run\n');
fprintf(fid, '============================\n');
fprintf(fid, 'Timestamp : %s\n', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
fprintf(fid, 'Device    : %s\n', cfg.GpuName);
fprintf(fid, 'Model     : %s\n', cfg.Paths.Model);
fprintf(fid, 'Seeds     : %s\n', cfg.Paths.Localization);
if isempty(elapsed)
    fprintf(fid, 'Source    : cached masks (scored by run_eval, not re-segmented)\n');
    fprintf(fid, 'Images    : %d\n\n', height(perUnit));
else
    fprintf(fid, 'Images    : %d\n', numel(elapsed));
    fprintf(fid, 'Runtime   : %.1f s (%.1f s/image)\n\n', sum(elapsed), mean(elapsed));
end

if byImage
    fprintf(fid, 'Whole-image scores, union mask vs FilledGT (%d images)\n', summary.nUnits);
else
    fprintf(fid, 'Per-segment scores, seed mask vs SegmentFilledGT (%d segments)\n', summary.nUnits);
end
fprintf(fid, '  mean IoU        : %.6f\n', summary.meanIou);
fprintf(fid, '  median IoU      : %.6f\n', summary.medianIou);
fprintf(fid, '  mean F1         : %.6f\n', summary.meanF1);
fprintf(fid, '  median F1       : %.6f\n', summary.medianF1);
fprintf(fid, '  mean Precision  : %.6f\n', summary.meanPrecision);
fprintf(fid, '  mean Recall     : %.6f\n', summary.meanRecall);
if ~byImage
    fprintf(fid, '  unclaimed GT px : %d (lesions no localization seed points at)\n', ...
        summary.unclaimedGtPx);
end
fprintf(fid, '  missing GT files: %d\n\n', summary.missingGtFiles);

writeParams(fid, 'Force',  cfg.Force);
writeParams(fid, 'GVF',    cfg.Gvf);
writeParams(fid, 'DRLSE',  cfg.Drlse);
writeParams(fid, 'Patch',  cfg.Patch);

if byImage
    fprintf(fid, '\nPer-image\n---------\n');
    fprintf(fid, '  %-10s %5s %7s %7s %7s %7s %9s %9s\n', ...
        'Image', 'Seeds', 'IoU', 'F1', 'Prec', 'Rec', 'PredPx', 'GtPx');
    for i = 1:height(perUnit)
        fprintf(fid, '  %-10s %5d %7.4f %7.4f %7.4f %7.4f %9d %9d\n', ...
            perUnit.Image(i), perUnit.Seeds(i), perUnit.Iou(i), perUnit.F1(i), ...
            perUnit.Precision(i), perUnit.Recall(i), perUnit.PredPx(i), perUnit.GtPx(i));
    end
else
    fprintf(fid, '\nPer-segment\n-----------\n');
    fprintf(fid, '  %-10s %5s %5s %7s %7s %7s %7s %9s %9s\n', ...
        'Image', 'X', 'Y', 'IoU', 'F1', 'Prec', 'Rec', 'PredPx', 'GtPx');
    for i = 1:height(perUnit)
        fprintf(fid, '  %-10s %5d %5d %7.4f %7.4f %7.4f %7.4f %9d %9d\n', ...
            perUnit.Image(i), perUnit.X(i), perUnit.Y(i), ...
            perUnit.Iou(i), perUnit.F1(i), perUnit.Precision(i), ...
            perUnit.Recall(i), perUnit.PredPx(i), perUnit.GtPx(i));
    end
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function writeParams(fid, name, s)
fprintf(fid, '%s parameters\n', name);
f = fieldnames(s);
for k = 1:numel(f)
    v = s.(f{k});
    if isnumeric(v) || islogical(v)
        fprintf(fid, '  %-12s %g\n', f{k}, double(v));
    else
        fprintf(fid, '  %-12s %s\n', f{k}, string(v));
    end
end
fprintf(fid, '\n');
end
