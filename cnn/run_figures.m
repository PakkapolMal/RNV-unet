function outDir = run_figures(cfg, outDir)
%RUN_FIGURES Error map vs ground truth for every cached image, as scored.
%
%  run_figures                    % the current configuration
%  run_figures(cfg)               % custom config, or a frozen arm's name
%  run_figures(cfg, outDir)       % default <Output>/figures/<reference CSV stem>
%
%  One PNG per image, from the cached CV P through cnngvf.cachedMask: the same
%  mask run_cache scores, so each title's numbers are the scored ones.
%    left   error map on the raw OCTA: TP green, FP red, FN blue (the
%           manuscript's colours), localization points as pink dots
%    right  ground truth as a white layer at 0.7 opacity over the raw OCTA:
%           lesion bright, background dimmed to 0.3
%    title  image, IoU, Dice, precision, recall
%  Prints the largest IoU difference from cfg.CV.reference: ~1e-15, or the
%  figures are not of the scored masks. Minutes; no GPU.
%
%  See also RUN_CACHE, CNNGVF.CACHEDMASK.

if nargin < 1
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);
end
cacheDir = cfg.CV.probCache;
if strlength(cacheDir) == 0
    cacheDir = fullfile(cfg.Paths.Output, 'cv', 'probcache');
end
if nargin < 2
    [~, stem] = fileparts(cfg.CV.reference);
    if strlength(stem) == 0, stem = "cv"; end
    outDir = fullfile(cfg.Paths.Output, 'figures', stem);
end
outDir = char(outDir);
if ~exist(outDir, 'dir'), mkdir(outDir); end

M = cnngvf.manifest(cfg);
[names, seeds] = cnngvf.loadPoints(cfg);
images = cnngvf.probCache("list", cacheDir);
ref = table(strings(0, 1), zeros(0, 1), 'VariableNames', {'Image', 'Iou'});
if strlength(cfg.CV.reference) > 0 && isfile(cfg.CV.reference)
    ref = readtable(cfg.CV.reference, 'TextType', 'string');
end
fprintf('run_figures | arm %s | %d images -> %s\n', cfg.Arm, numel(images), outDir);

pink = [1 0.35 0.75];
worst = 0;
for i = 1:numel(images)
    img = images(i);
    k = find(M.Name == img, 1);
    s = seeds(names == img, :);
    mask = cnngvf.cachedMask(cacheDir, img, s, cfg);
    gt = cnngvf.loadMask(M.GtPath(k));
    m = cnngvf.metrics(mask, gt);
    r = ref.Iou(ref.Image == img);
    if ~isempty(r), worst = max(worst, abs(m.iou - r)); end

    I = cnngvf.loadChannels(M.ImgPath(k), cfg).I;
    L = zeros(size(gt), 'uint8');
    L(mask & gt) = 1;
    L(mask & ~gt) = 2;
    L(~mask & gt) = 3;
    errRgb = labeloverlay(I, categorical(L, 1:3), 'Colormap', [0 0.8 0; 1 0 0; 0.15 0.35 1], 'Transparency', 0.3);
    gtImg = 0.7 * double(gt) + 0.3 * I;   % the GT layer at 0.7 opacity over the raw image

    f = figure('Visible', 'off', 'Color', 'w', 'Position', [0 0 1400 760]);
    t = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax = nexttile(t);
    imshow(errRgb, 'Parent', ax);
    hold(ax, 'on');
    if ~isempty(s)
        plot(ax, s(:, 1), s(:, 2), 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', pink, 'MarkerEdgeColor', pink);
    end
    title(ax, sprintf('Inference error map (%d points)', size(s, 1)), 'FontSize', 11);
    xlabel(ax, 'green TP · red FP · blue FN · pink: localization points', 'FontSize', 10);
    ax = nexttile(t);
    imshow(gtImg, 'Parent', ax);
    title(ax, 'Ground truth, 0.7 opacity over raw OCTA', 'FontSize', 11);
    xlabel(ax, sprintf('%s · GT %d px · predicted %d px', M.Dataset(k), nnz(gt), nnz(mask)), 'FontSize', 10);
    title(t, sprintf('%s  |  IoU %.3f  ·  Dice %.3f  ·  Precision %.3f  ·  Recall %.3f', ...
        img, m.iou, m.f1, m.precision, m.recall), 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
    [~, stem] = fileparts(img);
    exportgraphics(f, fullfile(outDir, stem + ".png"), 'Resolution', 110);
    close(f);
end
fprintf('run_figures | %d figures | max |IoU - reference| %.3g\n', numel(images), worst);
end
