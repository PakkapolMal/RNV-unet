function cfg = defaults(initMethod)
%RNV.DEFAULTS Default configuration for the RNV segmentation pipeline.
%
%  cfg = rnv.defaults()             % ray-casting pipeline (recommended)
%  cfg = rnv.defaults("adaptive")   % single global-radius circle pipeline
%
%  Returns a config struct consumed by every rnv.* function. Override any
%  field before use, e.g.:
%     cfg = rnv.defaults(); cfg.Snake.Kappa = 10;
%
%  Fields
%    Paths        Base/Img/Localization/GT/Output directories (absolute,
%                 resolved from this file's location - no hard-coded drives).
%    Init         "raycast" | "adaptive" - initial-contour strategy.
%    PostProc     "contour" | "component" - how the snake output becomes a
%                 mask. "contour" repairs and rasterises the contour O;
%                 "component" picks a connected component of Snake2D's J.
%    Denoise      Median-filter kernel applied on load, [] to disable.
%    GvfMargin    Dilation radius (px) of the ROI the GVF field is computed
%                 in; 0 = use the full image.
%    StartRow     First localization CSV row to process (rows 1..39 are not
%                 part of the evaluated split).
%    Radius       Initialization parameters (see rnv.initContour).
%    Snake        Snake2D options (see Snake2D help and ../docs/CLAUDE.md).
%    Post         MinArea / MorphElement for mask cleanup.
%
%  See also RNV.SEGMENTIMAGE, RNV.INITCONTOUR, SNAKE2D.

if nargin < 1 || strlength(initMethod) == 0
    initMethod = "raycast";
end
initMethod = lower(string(initMethod));

% --- Paths -----------------------------------------------------------------
% This file lives at <base>/RNV-Claude/main/+rnv/defaults.m
pkgDir  = fileparts(mfilename('fullpath'));            % .../main/+rnv
mainDir = fileparts(pkgDir);                           % .../main
baseDir = fileparts(fileparts(mainDir));               % .../Fast-Gradient-Vector-Flow-main

% Snake2D and its helpers live in baseDir; the rnv package needs mainDir.
addpath(baseDir);
addpath(mainDir);

cfg.Paths.Base         = baseDir;
cfg.Paths.Main         = mainDir;
cfg.Paths.Img          = fullfile(baseDir, 'RNV-DRAC22NV');
cfg.Paths.Localization = fullfile(baseDir, 'RNV-DRAC22NV', 'DRAC22_RNV_Localization_Points_Sorted.csv');
cfg.Paths.GT           = fullfile(baseDir, 'RNV-DRAC22GT', 'FilledGT');
cfg.Paths.Output       = fullfile(baseDir, 'result', char("RNV-Claude-" + initMethod));

cfg.Init     = initMethod;
cfg.StartRow = 40;

% --- Per-method settings ---------------------------------------------------
switch initMethod
    case "raycast"
        cfg.PostProc  = "contour";
        cfg.Denoise   = [3 3];   % speckle removal before the gradient
        cfg.GvfMargin = 30;      % tight ROI keeps far-field edges out of the GVF

        cfg.Radius = struct( ...
            'r_min',            30, ...
            'r_max',            400, ...
            'n_points',         60, ...
            'smoothing_window', 5, ...
            'peak_drop_tol',    0.15, ...   % 15% drop = we passed the edge
            'min_peak_radius',  60);        % ignore noise close to the seed

        cfg.Snake = struct( ...
            'Verbose',     false, ...
            'Wterm',       0, ...
            'Delta',      -0.1, ...   % gentle inward pressure vs. outward drift
            'Alpha',       0.05, ...  % elasticity
            'Beta',        0.01, ...  % near-zero rigidity -> deep concavities
            'Kappa',       4, ...     % GVF edge-snapping weight
            'Mu',          0.3, ...   % keep < 0.5, see docs/CLAUDE.md
            'Sigma1',      3, ...
            'Sigma2',      3, ...
            'GIterations', 800, ...
            'Iterations',  600, ...
            'Wedge',       4, ...
            'Wline',       0.4);

    case "adaptive"
        cfg.PostProc  = "component";
        cfg.Denoise   = [];      % original adaptive pipeline used no median filter
        cfg.GvfMargin = 0;       % GVF over the full image

        cfg.Radius = struct( ...
            'r_min',            30, ...
            'r_max',            400, ...
            'r_step',           5, ...
            'n_points',         60, ...
            'smoothing_window', 5, ...
            'peak_drop_tol',    0.15, ...
            'min_peak_radius',  60);

        cfg.Snake = struct( ...
            'Verbose',     false, ...
            'Wterm',       0, ...
            'Delta',      -0.55, ...
            'Alpha',       0.07, ...
            'Beta',        0.01, ...
            'Kappa',       8, ...
            'Mu',          0.15, ...
            'Sigma1',      3, ...
            'Sigma2',      3, ...
            'GIterations', 1000, ...
            'Iterations',  1200, ...
            'Wedge',       4, ...
            'Wline',       0.4);

    otherwise
        error('rnv:defaults:badInit', ...
            'Unknown Init "%s". Use "raycast" or "adaptive".', initMethod);
end

% --- Shared post-processing -----------------------------------------------
cfg.Post.MinArea      = 500;
cfg.Post.MorphElement = strel('disk', 2);

% --- Edge map --------------------------------------------------------------
cfg.EdgeSigma = 2;   % Gaussian blur before the gradient magnitude
end
