function [bestParams, bestScore] = run_tuning(varargin)
%RUN_TUNING Coordinate-descent tuner for the Snake2D parameters.
%
%  run_tuning                       % tune the ray-casting pipeline
%  run_tuning("adaptive")           % tune the circle pipeline
%  run_tuning(cfg)                  % tune a custom config
%  run_tuning(..., 'Grids', g, 'MaxRounds', r, 'SweepOrder', {...})
%
%  Sweeps one parameter at a time over its grid, keeping the best value
%  found so far, and maximizes mean IoU over the GT set. Repeats until a
%  full round yields no improvement or MaxRounds is reached.
%
%  Replaces tune_snake_coorddescent.m and tune_adaptive_gridsearch.m, which
%  were the same tuner wrapped around a different pipeline.
%
%  Writes to <cfg.Paths.Main>/tuning_<init>/:
%    tuning_log.csv     one row per evaluation, with per-image IoU
%    sweep_<r>_<p>.png  IoU/F1 vs. parameter value for each sweep
%    convergence.png    best IoU per round
%    best_params.mat/.txt
%
%  See also RNV.EVALUATE, RUN_BATCH.

% --- Arguments -------------------------------------------------------------
% Leading argument may be a config struct or an init-method name; anything
% else is a name/value option.
cfg = [];
if ~isempty(varargin)
    first = varargin{1};
    if isstruct(first)
        cfg = first;
        varargin(1) = [];
    elseif (ischar(first) || isstring(first)) && ~isConfigOption(first)
        cfg = rnv.defaults(first);
        varargin(1) = [];
    end
end
if isempty(cfg)
    cfg = rnv.defaults();
end
rest = varargin;

p = inputParser;
p.addParameter('SweepOrder', {'Kappa', 'Delta', 'Alpha', 'Beta'});
p.addParameter('MaxRounds', 3);
p.addParameter('MinImprove', 1e-3);
p.addParameter('Grids', struct( ...
    'Kappa', [2, 4, 6, 8, 10], ...
    'Delta', [-0.05, -0.1, -0.2, -0.35, -0.55], ...
    'Alpha', [0.01, 0.03, 0.05, 0.07, 0.1], ...
    'Beta',  [0, 0.001, 0.005, 0.01, 0.02]));
p.parse(rest{:});
opt = p.Results;

outDir = fullfile(cfg.Paths.Main, char("tuning_" + cfg.Init));
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

gtFiles   = dir(fullfile(cfg.Paths.GT, '*.png'));
valImages = sort(string({gtFiles.name}))';
fprintf('run_tuning | init=%s | %d GT images | out=%s\n\n', ...
    cfg.Init, numel(valImages), outDir);

logPath = fullfile(outDir, 'tuning_log.csv');
writeHeader(logPath, valImages);

cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

% --- Baseline --------------------------------------------------------------
bestParams = cfg.Snake;
fprintf('=== Baseline ===\n');
[bestScore, bestPerImg] = scoreCached(bestParams, cfg, valImages, cache);
appendRow(logPath, 0, 'baseline', bestParams, bestScore, bestPerImg, valImages);
fprintf('mean IoU = %.4f, mean F1 = %.4f (%.1f s)\n', ...
    bestScore.meanIou, bestScore.meanF1, bestScore.runtimeSec);

convRounds = 0;
convIou    = bestScore.meanIou;

% --- Coordinate descent ----------------------------------------------------
for roundIdx = 1:opt.MaxRounds
    fprintf('\n=== Round %d / %d ===\n', roundIdx, opt.MaxRounds);
    improved = false;

    for s = 1:numel(opt.SweepOrder)
        pname = opt.SweepOrder{s};
        gridVals = opt.Grids.(pname);
        ious     = zeros(size(gridVals));
        f1s      = zeros(size(gridVals));

        fprintf('  Sweeping %s (current %g)\n', pname, bestParams.(pname));

        for g = 1:numel(gridVals)
            cand = bestParams;
            cand.(pname) = gridVals(g);

            [sc, perImg] = scoreCached(cand, cfg, valImages, cache);
            ious(g) = sc.meanIou;
            f1s(g)  = sc.meanF1;

            appendRow(logPath, roundIdx, pname, cand, sc, perImg, valImages);
            fprintf('    %s = %-8g IoU = %.4f  F1 = %.4f  (%.1f s)\n', ...
                pname, gridVals(g), sc.meanIou, sc.meanF1, sc.runtimeSec);

            if sc.meanIou > bestScore.meanIou + opt.MinImprove
                bestScore  = sc;
                bestParams = cand;
                bestPerImg = perImg;
                improved   = true;
                fprintf('      ** new best: %.4f **\n', bestScore.meanIou);
            end
        end

        saveSweepPlot(outDir, roundIdx, pname, gridVals, ious, f1s, bestParams);
    end

    convRounds(end+1) = roundIdx;         %#ok<AGROW>
    convIou(end+1)    = bestScore.meanIou; %#ok<AGROW>

    if ~improved
        fprintf('No improvement in round %d; stopping.\n', roundIdx);
        break;
    end
end

% --- Results ---------------------------------------------------------------
saveConvergencePlot(outDir, convRounds, convIou);
saveBest(outDir, cfg, bestParams, bestScore, bestPerImg);

fprintf('\n=== Done ===\nBest mean IoU = %.4f\n', bestScore.meanIou);
fprintf('Alpha=%.4f Beta=%.4f Delta=%.4f Kappa=%.4f\n', ...
    bestParams.Alpha, bestParams.Beta, bestParams.Delta, bestParams.Kappa);
fprintf('Saved to %s\n', outDir);
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function tf = isConfigOption(name)
tf = ismember(string(name), ["SweepOrder", "MaxRounds", "MinImprove", "Grids"]);
end

function [score, perImage] = scoreCached(snakeOpts, cfg, valImages, cache)
% Coordinate descent revisits the incumbent value in every sweep; caching on
% the four tuned parameters removes those redundant full evaluations.
key = sprintf('A%.6f_B%.6f_D%.6f_K%.6f', ...
    snakeOpts.Alpha, snakeOpts.Beta, snakeOpts.Delta, snakeOpts.Kappa);

if isKey(cache, key)
    hit      = cache(key);
    score    = hit.score;
    perImage = hit.perImage;
    return;
end

cfg.Snake = snakeOpts;
[score, perImage] = rnv.evaluate(cfg, valImages);

% containers.Map is a handle, so this mutation is visible to the caller.
% checkcode flags it as a possibly-unused assignment; it is not.
cache(key) = struct('score', score, 'perImage', perImage);
end

function writeHeader(logPath, valImages)
% Rewrite the header whenever the validation set changes, so old rows can
% never be appended under mismatched columns.
header = "timestamp,round,swept_param,Alpha,Beta,Delta,Kappa,mean_iou,mean_f1,runtime_sec";
for i = 1:numel(valImages)
    [~, stem] = fileparts(valImages(i));
    header = header + "," + stem + "_iou";
end

if isfile(logPath)
    existing = readlines(logPath);
    if ~isempty(existing) && strcmp(strtrim(existing(1)), header)
        return;   % same columns: keep appending to the existing log
    end
end
writelines(header, logPath);
end

function appendRow(logPath, roundIdx, sweptParam, params, score, perImage, valImages)
row = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')) + "," + ...
    roundIdx + "," + sweptParam + "," + ...
    params.Alpha + "," + params.Beta + "," + params.Delta + "," + params.Kappa + "," + ...
    score.meanIou + "," + score.meanF1 + "," + score.runtimeSec;

for i = 1:numel(valImages)
    row = row + "," + perImage.Iou(i);   % NaN prints as "NaN"
end

writelines(row, logPath, 'WriteMode', 'append');
end

function saveSweepPlot(outDir, roundIdx, pname, gridVals, ious, f1s, bestParams)
fig = figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
cleanup = onCleanup(@() close(fig));

plot(gridVals, ious, '-ok', 'LineWidth', 2, 'MarkerFaceColor', 'r');
hold on;
plot(gridVals, f1s, '--ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel(pname);
ylabel('Score');
title(sprintf('Round %d: %s sweep (best %s = %g)', ...
    roundIdx, pname, pname, bestParams.(pname)));
legend('IoU', 'F1', 'Location', 'best');
grid on;

if all(gridVals <= 0)
    set(gca, 'XDir', 'reverse');
end

exportgraphics(fig, fullfile(outDir, sprintf('sweep_%d_%s.png', roundIdx, pname)), ...
    'Resolution', 200);
end

function saveConvergencePlot(outDir, rounds, ious)
fig = figure('Visible', 'off', 'Position', [100, 100, 700, 450]);
cleanup = onCleanup(@() close(fig));

plot(rounds, ious, '-o', 'LineWidth', 2, 'MarkerFaceColor', [0.2 0.6 0.9]);
xlabel('Round');
ylabel('Best mean IoU');
title('Coordinate-descent convergence');
xticks(rounds);
grid on;

exportgraphics(fig, fullfile(outDir, 'convergence.png'), 'Resolution', 200);
end

function saveBest(outDir, cfg, bestParams, bestScore, bestPerImg)
save(fullfile(outDir, 'best_params.mat'), 'bestParams', 'bestScore', 'bestPerImg', 'cfg');
writetable(bestPerImg, fullfile(outDir, 'best_per_image.csv'));

txtPath = fullfile(outDir, 'best_params.txt');
fid = fopen(txtPath, 'w');
if fid < 0
    warning('rnv:runTuning:noWrite', 'Could not write %s', txtPath);
    return;
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Best Snake2D parameters (coordinate descent, init = %s)\n', cfg.Init);
fprintf(fid, '=======================================================\n');
fprintf(fid, 'Mean IoU: %.6f\n', bestScore.meanIou);
fprintf(fid, 'Mean F1:  %.6f\n', bestScore.meanF1);
fprintf(fid, 'Images scored: %d\n\n', bestScore.nScored);
fprintf(fid, 'Apply with:\n');
fprintf(fid, '  cfg = rnv.defaults("%s");\n', cfg.Init);
fprintf(fid, '  cfg.Snake.Alpha = %.4f;\n', bestParams.Alpha);
fprintf(fid, '  cfg.Snake.Beta  = %.4f;\n', bestParams.Beta);
fprintf(fid, '  cfg.Snake.Delta = %.4f;\n', bestParams.Delta);
fprintf(fid, '  cfg.Snake.Kappa = %.4f;\n', bestParams.Kappa);
fprintf(fid, '  run_batch(cfg);\n\n');
fprintf(fid, 'Per-image IoU:\n');
for i = 1:height(bestPerImg)
    fprintf(fid, '  %-12s %.4f\n', bestPerImg.Image(i), bestPerImg.Iou(i));
end
end
