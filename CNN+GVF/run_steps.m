function out = run_steps(imageName, arm)
%RUN_STEPS Run the pipeline on ONE image, saving every intermediate stage.
%
%  run_steps("458.png")                       % the current arm
%  run_steps("nv (1).png")                    % a TUH image
%  run_steps 458                              % the .png is optional
%  run_steps("458.png", "unclustered")        % a specific arm
%  out = run_steps(...)                       % also return every intermediate
%
%  Configuration comes from cnngvf.arm, so the final mask is identical to that
%  image's mask in the arm's result directory. The difference is that every
%  intermediate is written out instead of discarded.
%
%  Outputs land in <Output>/steps/<stem>/, numbered in pipeline order:
%
%    01_raw.png            the image as read
%    02_denoised.png       after the median filter, if any (CNN channel 1)
%    03_vesselness.png     Frangi response, only when the arm enables it
%    04_seed_heatmap.png   Gaussian seed map    (the last CNN channel)
%    05_prob.png           P, the CNN probability map
%    06_force.png          f = |grad P|, the GVF energy field
%    07_gvf_magnitude.png  |(u,v)| of the diffused field
%    08_gvf_quiver.png     the field itself, over the image
%    09_edge_stopper.png   g, where the level set is slowed
%    10_phi0_init.png      the initialisation, P > tau
%    11_levelset_raw.png   phi < 0 straight out of DRLSE
%    12_after_morph.png    after imclose + imfill
%    13_final_mask.png     after seedMasks - the pipeline's answer
%    14_overlay.png        image + seeds + final contours
%    15_gt_overlay.png     ground truth, if present
%    16_error_map.png      green TP / red FP / blue FN, if ground truth present
%    patches/seed_NN.png   the 3-channel network input for each seed
%    steps.mat             every array above, for further analysis
%    steps.txt             per-stage pixel counts and, with GT, the metrics
%
%  Scalar fields are written normalised to their own range, so a dark
%  vesselness image means low absolute response, not an error.
%
%  See also RUN_BATCH_UNCLUSTERED, CNNGVF.SEGMENTIMAGE.

% --- configuration ---------------------------------------------------------
% Taken from cnngvf.arm, not rebuilt here. This file used to set the model and
% the CSV by hand and inherit everything else from defaults, so when the
% unclustered arm dropped the median filter and the vesselness channel this
% file kept building 3 channels for a 2-channel network and died in
% inferProbMap with a channel mismatch - while ../docs/UNCLUSTERED.md was still
% telling readers to run it. Asking for the arm by name makes that impossible.
if nargin < 2 || isempty(arm)
    arm = "DRACTUH_unclustered";
end
cfg = cnngvf.arm(cnngvf.defaults(), arm);

imageName = string(imageName);
if ~endsWith(imageName, ".png")
    imageName = imageName + ".png";
end
[~, stem] = fileparts(imageName);

M = cnngvf.manifest(cfg);
k = find(M.Name == imageName, 1);
if isempty(k)
    error('cnngvf:runSteps:noImage', ...
        '%s is not in the domain of arm "%s" (datasets: %s).', ...
        imageName, cfg.Arm, strjoin(cfg.ActiveDatasets, ', '));
end
imgPath = M.ImgPath(k);

if ~isfile(cfg.Paths.Model)
    error('cnngvf:runSteps:noModel', ...
        'No trained network at %s. Run run_train("%s") first.', cfg.Paths.Model, cfg.Arm);
end

[names, seeds] = cnngvf.loadPoints(cfg);
seeds = seeds(names == imageName, :);
if isempty(seeds)
    error('cnngvf:runSteps:noSeeds', '%s carries no %s points.', imageName, cfg.Points);
end

outDir = fullfile(cfg.Paths.Output, 'steps', char(stem));
patchDir = fullfile(outDir, 'patches');
for d = string({outDir, patchDir})
    if ~exist(d, 'dir'), mkdir(d); end
end

fprintf('run_steps | %s | %d seed(s) | device: %s\n', imageName, size(seeds, 1), cfg.GpuName);
fprintf('Output: %s\n\n', outDir);

loaded = load(cfg.Paths.Model, 'net');
net = loaded.net;

% =========================================================================
% the pipeline, one stage at a time
% =========================================================================
t = tic;

% 1-3. channels ------------------------------------------------------------
[img, chans] = cnngvf.loadChannels(imgPath, cfg);
save1(outDir, '01_raw.png',      img.raw);
save1(outDir, '02_denoised.png', img.I);

% Channel 2 exists only while the arm enables vesselness; the current arms do
% not. Derive it rather than indexing blindly.
hasVessel = size(chans, 3) >= 2;
if hasVessel
    save1(outDir, '03_vesselness.png', chans(:, :, 2));
end

% 4. seed heat-map over the whole image (the CNN sees it per patch) ---------
heat = zeros(img.h, img.w);
[xx, yy] = meshgrid(1:img.w, 1:img.h);
for s = 1:size(seeds, 1)
    heat = max(heat, exp(-((xx - seeds(s, 1)).^2 + (yy - seeds(s, 2)).^2) ...
        ./ (2 * cfg.Patch.sigmaSeed^2)));
end
save1(outDir, '04_seed_heatmap.png', heat);

% The exact tensor handed to the network, per seed: every channel side by side,
% however many the arm builds.
for s = 1:size(seeds, 1)
    X = cnngvf.seedPatch(chans, seeds(s, :), seeds(s, :), cfg);
    strip = [];
    for c = 1:size(X, 3)
        strip = cat(2, strip, norm01(X(:, :, c)));
    end
    imwrite(im2uint8(strip), fullfile(patchDir, sprintf('seed_%02d_%d-%d.png', s, ...
        round(seeds(s, 1)), round(seeds(s, 2)))));
end

% 5. probability map -------------------------------------------------------
P = cnngvf.inferProbMap(chans, seeds, net, cfg);
save1(outDir, '05_prob.png', P);

% 6. GVF energy field ------------------------------------------------------
f = cnngvf.buildForce(P, img.I, cfg);
save1(outDir, '06_force.png', f);

% 7-8. GVF field -----------------------------------------------------------
[u, v] = cnngvf.gvfField(f, cfg);
save1(outDir, '07_gvf_magnitude.png', hypot(u, v));
saveQuiver(fullfile(outDir, '08_gvf_quiver.png'), img.I, u, v);

% 9. edge stopper ----------------------------------------------------------
% Mirrors the g that cnngvf.drlse builds internally; drlse does not return it.
% Keep in step with that file if the formula there changes.
[gx, gy] = gradient(imgaussfilt(double(P), 1));
gm = sqrt(gx.^2 + gy.^2);
g  = 1 ./ (1 + (cfg.Drlse.edgeGain * gm ./ max(max(gm(:)), eps)).^2);
save1(outDir, '09_edge_stopper.png', g);

% 10. initialisation -------------------------------------------------------
[phi0, BW0] = cnngvf.initPhi(P, seeds, cfg);
save1(outDir, '10_phi0_init.png', BW0);

% 11. level set ------------------------------------------------------------
[BW, phi] = cnngvf.drlse(P, phi0, u, v, cfg);
save1(outDir, '11_levelset_raw.png', BW);

% 12. morphology -----------------------------------------------------------
BWm = imfill(imclose(BW, cfg.Post.MorphElement), 'holes');
save1(outDir, '12_after_morph.png', BWm);

% 13. seed-claimed components ---------------------------------------------
[perSeed, mask] = cnngvf.seedMasks(BWm, seeds, cfg.Post.MinArea, cfg.Post.ClaimRule);
save1(outDir, '13_final_mask.png', mask);

elapsed = toc(t);

% 14. overlay --------------------------------------------------------------
contours = bwboundaries(mask, 'noholes');
rnv.saveOverlay(fullfile(outDir, '14_overlay.png'), img, seeds, contours);

% 15-16. ground truth ------------------------------------------------------
gt = rnv.loadMask(M.GtPath(k));
m  = [];
if ~isempty(gt)
    save1(outDir, '15_gt_overlay.png', gt);
    err = cat(3, double(mask & ~gt), double(mask & gt), double(~mask & gt));
    imwrite(im2uint8(err), fullfile(outDir, '16_error_map.png'));
    m = rnv.metrics(mask, gt);
end

% =========================================================================
% records
% =========================================================================
vesselness = [];
if hasVessel
    vesselness = chans(:, :, 2);
end
out = struct('image', imageName, 'dataset', M.Dataset(k), 'seeds', seeds, ...
    'raw', img.raw, 'I', img.I, 'vesselness', vesselness, ...
    'heat', heat, 'P', P, 'f', f, 'u', u, 'v', v, ...
    'g', g, 'phi0', phi0, 'BW0', BW0, 'phi', phi, 'BW', BW, 'BWm', BWm, ...
    'perSeed', {perSeed}, 'mask', mask, 'gt', gt, 'metrics', m, 'cfg', cfg);
save(fullfile(outDir, 'steps.mat'), '-struct', 'out');

writeReport(fullfile(outDir, 'steps.txt'), imageName, cfg, seeds, ...
    P, BW0, BW, BWm, mask, gt, m, elapsed);

fprintf('\nstage             pixels\n');
fprintf('  P > %.2f        %8d\n', cfg.Drlse.tau, nnz(P > cfg.Drlse.tau));
fprintf('  phi0 init       %8d\n', nnz(BW0));
fprintf('  level set       %8d\n', nnz(BW));
fprintf('  after morph     %8d\n', nnz(BWm));
fprintf('  final mask      %8d\n', nnz(mask));
if ~isempty(gt)
    fprintf('  ground truth    %8d\n', nnz(gt));
    fprintf('\nIoU %.4f  F1 %.4f  precision %.4f  recall %.4f\n', ...
        m.iou, m.f1, m.precision, m.recall);
end
fprintf('\n%.1f s. Wrote %d files to %s\n', elapsed, ...
    numel(dir(fullfile(outDir, '*.png'))) + numel(dir(fullfile(patchDir, '*.png'))) + 2, outDir);
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function save1(dir_, name, A)
%SAVE1 Write a scalar field or mask as 8-bit, normalised to its own range.
if islogical(A)
    imwrite(A, fullfile(dir_, name));
else
    imwrite(im2uint8(norm01(A)), fullfile(dir_, name));
end
end


function A = norm01(A)
A = double(A);
lo = min(A(:)); hi = max(A(:));
if hi > lo
    A = (A - lo) ./ (hi - lo);
else
    A = zeros(size(A));
end
end


function saveQuiver(outFile, I, u, v)
%SAVEQUIVER The GVF field over the image, decimated so the arrows are legible.
fig = figure('Visible', 'off');
cleanup = onCleanup(@() close(fig));
imshow(I, []);
hold on;
step = 16;
[xx, yy] = meshgrid(1:step:size(I, 2), 1:step:size(I, 1));
quiver(xx, yy, u(1:step:end, 1:step:end), v(1:step:end, 1:step:end), 2, 'y');
axis ij off;
exportgraphics(gca, outFile, 'Resolution', 200);
end


function writeReport(txtPath, imageName, cfg, seeds, P, BW0, BW, BWm, mask, gt, m, elapsed)
fid = fopen(txtPath, 'w');
if fid < 0
    warning('cnngvf:runSteps:noWrite', 'Could not write %s', txtPath);
    return;
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, 'Per-stage trace: %s\n', imageName);
fprintf(fid, '================================\n');
fprintf(fid, 'Timestamp : %s\n', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
fprintf(fid, 'Model     : %s\n', cfg.Paths.Model);
fprintf(fid, 'Seeds     : %d from %s\n', size(seeds, 1), cfg.Paths.Localization);
fprintf(fid, 'Runtime   : %.1f s\n\n', elapsed);

fprintf(fid, 'Stage pixel counts\n------------------\n');
fprintf(fid, '  P > %-12.2f %8d\n', cfg.Drlse.tau, nnz(P > cfg.Drlse.tau));
fprintf(fid, '  phi0 init        %8d\n', nnz(BW0));
fprintf(fid, '  level set        %8d\n', nnz(BW));
fprintf(fid, '  after morph      %8d\n', nnz(BWm));
fprintf(fid, '  final mask       %8d\n', nnz(mask));
if ~isempty(gt)
    fprintf(fid, '  ground truth     %8d\n', nnz(gt));
    fprintf(fid, '  over-prediction  %8.2fx\n', nnz(mask) / max(nnz(gt), 1));
end

fprintf(fid, '\nProbability map\n---------------\n');
fprintf(fid, '  max              %.4f\n', max(P(:)));
fprintf(fid, '  mean             %.6f\n', mean(P(:)));
fprintf(fid, '  px above tau     %8d\n', nnz(P > cfg.Drlse.tau));

if ~isempty(m)
    fprintf(fid, '\nScores vs FilledGT\n------------------\n');
    fprintf(fid, '  IoU              %.6f\n', m.iou);
    fprintf(fid, '  F1               %.6f\n', m.f1);
    fprintf(fid, '  precision        %.6f\n', m.precision);
    fprintf(fid, '  recall           %.6f\n', m.recall);
end

fprintf(fid, '\nSeeds (x, y)\n------------\n');
for s = 1:size(seeds, 1)
    fprintf(fid, '  %3d  %5d %5d\n', s, round(seeds(s, 1)), round(seeds(s, 2)));
end
end
