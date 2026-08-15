function cfg = defaults()
%CNNGVF.DEFAULTS Default configuration for the CNN + GVF pipeline.
%
%  cfg = cnngvf.defaults()
%
%  Implements Approach 1 of ../docs/spec.md: a seed-conditioned U-Net predicts
%  a probability map P, P drives a GVF field, and a DRLSE level set evolves in
%  that field to the final mask.
%
%  Override any field before use, e.g.
%     cfg = cnngvf.defaults(); cfg.Drlse.iter = 600;
%
%  Fields
%    Paths    Base/Img/Localization/GT/Model/Output (absolute, resolved from
%             this file - no hard-coded drive letters).
%    Split    Validation image list; every other GT image trains.
%    Patch    CNN patch geometry and seed heat-map sigma.
%    Net      U-Net architecture and training hyper-parameters.
%    Force    How P becomes the GVF energy field f.
%    Gvf      GVF diffusion parameters (Xu & Prince).
%    Drlse    Level-set parameters (Li et al. 2010).
%    Post     Mask cleanup.
%    UseGpu   Run GVF/DRLSE on the GPU. Defaults to canUseGPU().
%
%  Defaults come from the parameter table in ../docs/spec.md.
%
%  See also CNNGVF.SEGMENTIMAGE, RUN_TRAIN, RUN_BATCH.

requireToolboxes();

% --- Paths -----------------------------------------------------------------
% This file lives at <base>/RNV-Claude/CNN+GVF/+cnngvf/defaults.m
pkgDir  = fileparts(mfilename('fullpath'));            % .../CNN+GVF/+cnngvf
mainDir = fileparts(pkgDir);                           % .../CNN+GVF
baseDir = fileparts(fileparts(mainDir));               % .../Fast-Gradient-Vector-Flow-main

addpath(mainDir);
addpath(fullfile(baseDir, 'lib', 'Fast-Gradient-Vector-Flow'));  % GVF mex
addpath(fullfile(baseDir, 'RNV-Claude', 'main'));                % rnv.* helpers

cfg.Paths.Base         = baseDir;
cfg.Paths.Main         = mainDir;
cfg.Paths.Img          = fullfile(baseDir, 'RNV-DRAC22NV');
cfg.Paths.Localization = fullfile(baseDir, 'RNV-DRAC22NV', 'DRAC22_RNV_Localization_Points_Sorted.csv');
cfg.Paths.GT           = fullfile(baseDir, 'RNV-DRAC22GT', 'FilledGT');
% One mask per localization seed, named <stem>-<x>-<y>.png. Written by
% separate_segments.py, which keeps the ground-truth component each seed falls
% in and drops the components no seed claims. This is the evaluation unit.
cfg.Paths.SegGT        = fullfile(baseDir, 'RNV-DRAC22GT', 'SegmentFilledGT');
% Where TRAINING seeds come from. Evaluation always uses cfg.Paths.Localization
% (the clustered centres the ground-truth segments are keyed to), so pointing
% this at a denser point set changes only what the network sees, never what it
% is scored against. run_train_unclustered.m sets it to the unclustered points.
cfg.Paths.LocalizationTrain = cfg.Paths.Localization;

cfg.Paths.Model        = fullfile(mainDir, 'models', 'rnvNet.mat');
cfg.Paths.Output       = fullfile(mainDir, 'result');

% rnv.loadSeeds / rnv.loadImage read these two.
% 1, not the 40 that RNV-Claude/main uses. That convention dated from a CSV
% whose first 39 rows were images outside the evaluated split; the sorted CSV
% now holds exactly the seeds of the ground-truth images. Nothing depends on
% this staying right: cnngvf.prepareData filters to images that actually have
% a ground-truth mask, so extra rows are dropped rather than mistrained on.
cfg.StartRow = 1;
cfg.Denoise  = [3 3];   % median kernel, speckle removal on load
cfg.EdgeSigma = 2;      % blur before the raw-image gradient (only used by "blend")

% --- Train / validation split ---------------------------------------------
% Patient-level: whole images, never patches, so no lesion leaks across the
% split. 4 of the 17 ground-truth images validate; the other 13 train.
cfg.Split.Val = ["461.png", "535.png", "576.png", "664.png"];

% Images removed from training AND validation entirely. run_cv sets this to a
% fold's held-out test images, so they are never seen by training or by
% best-checkpoint selection. Empty by default, so the existing runners are
% unaffected.
cfg.Split.Exclude = strings(0, 1);

% --- Vesselness channel ----------------------------------------------------
% Second CNN input channel. Small thicknesses: RNV vessels are thin, and
% ../docs/CLAUDE.md notes large scales blur the structure away entirely.
cfg.Vessel.scales = [2 4 6 8];

% --- CNN patches -----------------------------------------------------------
% 512, not the spec's default 256, and it must stay divisible by
% 2^EncoderDepth. The CNN cannot predict what the patch does not contain, and
% measured on this dataset 13 of the 35 ground-truth segments need a
% seed-centred window wider than 256 px - every one of them scored badly
% (1115: needs 391x1501, IoU 0.019; 795: 501x1815, IoU 0.074) while every
% segment that fitted scored 0.50-0.88. At 512, 31 of 35 fit. The remaining
% four (461, 535, 795, 1115) need 525-1815 px and stay clipped; only
% full-image inference would cover them.
cfg.Patch.size      = 512;
cfg.Patch.sigmaSeed = 32;    % Gaussian seed heat-map sigma (spec table)
cfg.Patch.jitter    = 32;    % +/- px random crop offset during training
cfg.Patch.perSeed   = 16;    % random crops per seed per epoch

% Spend the epoch equally across IMAGES rather than equally across seeds. Only
% matters when seed counts are uneven, i.e. for the unclustered detections; it
% is a no-op for the clustered centres, where every image has 1-7. See
% cnngvf.prepareData for the measurement that motivated it.
cfg.Patch.balanceByImage = true;

% --- U-Net -----------------------------------------------------------------
cfg.Net.encoderDepth = 4;
% 30, not 60. At patch 512 / miniBatch 2 an epoch is 240 iterations and takes
% ~5 min, so 60 epochs would be a 5 h run. 30 epochs x 480 patches sees 14400
% samples over 30 training seeds, and OutputNetwork="best-validation" keeps the
% best checkpoint rather than the last, so overshooting costs accuracy, not
% correctness.
cfg.Net.epochs       = 30;
% 512x512 patches are 4x the activations of 256, and the first attempt at 512
% with miniBatch 4 died out of memory at iteration 0 - the initial validation
% pass - on this 8 GB laptop GPU. Validation follows MiniBatchSize, so halving
% it covers both. (trainingOptions has no ValidationMiniBatchSize for adam.)
cfg.Net.miniBatch    = 2;
% Patches per predict() call at inference. Bounded for the same reason: an
% image can carry 62 unclustered seeds, and they cannot all go through the
% network at once.
cfg.Net.inferBatch   = 4;
cfg.Net.learnRate    = 1e-3;
% The progress plot is worth having interactively and is dead weight under
% matlab -batch, where nobody can see it.
if batchStartupOptionUsed
    cfg.Net.plots = "none";
else
    cfg.Net.plots = "training-progress";
end

% --- GVF energy field ------------------------------------------------------
%  "probGrad" f = |grad P|                 - the CNN boundary as an edge map
%  "blend"    w*|grad P| + (1-w)*|grad I|  - falls back towards the raw image
%                                            where the CNN is uncertain
cfg.Force.source = "probGrad";
cfg.Force.sigma  = 2;
cfg.Force.w      = 0.7;

% --- GVF diffusion (Xu & Prince) ------------------------------------------
cfg.Gvf.alpha = 0.2;    % explicit-Euler step size, 0 < alpha < 1
cfg.Gvf.mu    = 0.2;    % smoothing / noise trade-off
cfg.Gvf.iter  = 80;     % diffusion iterations (spec table)

% --- DRLSE level set (Li, Xu, Gui & Fox 2010) -----------------------------
cfg.Drlse.tau    = 0.5;   % P threshold that seeds phi0
cfg.Drlse.c0     = 2;     % |phi0| magnitude
cfg.Drlse.dt     = 2;     % time step (distance regularization permits it large)
cfg.Drlse.muDR   = 0.1;   % distance-regularization weight; must be < 0.25/dt
cfg.Drlse.lambda = 5;     % edge / length weight
cfg.Drlse.edgeGain = 8;   % how hard the edge stopper g bites; see cnngvf.drlse

% Balloon term. phi < 0 inside, so a NEGATIVE alfa expands. ../docs/spec.md
% suggests +/-1.5, but that is for a contour initialized far from the object.
% Here phi0 already comes from the probability map and sits on the boundary,
% and phi0 is binary rather than a distance function, so the Dirac has no
% narrow band and a balloon pushes on the whole image at once - it floods the
% background instead of filling concavities. Measured on the synthetic scene
% in test_cnngvf: alfa 0 gives IoU 0.965 stable to 600 iterations, alfa -0.2
% gives 0.735 and still shrinking, alfa -0.5 gives 0.130.
% Raise it only if you also initialize far from the lesion.
cfg.Drlse.alfa   = 0;
cfg.Drlse.eps    = 1.5;   % Dirac width
cfg.Drlse.iter   = 300;
cfg.Drlse.seedRadius = 15;  % fallback init disk where P never crosses tau

% --- Evaluation ------------------------------------------------------------
%  "segment" one row per seed: the seed's mask vs its own
%            SegmentFilledGT/<stem>-<x>-<y>.png. Keyed by seed coordinates, so
%            it only works for the clustered centres those files are named for.
%  "image"   one row per image: the union of all seeds' masks vs the whole
%            FilledGT/<stem>.png. Required for the unclustered points, which
%            have no per-point ground truth and several points per lesion -
%            scoring them individually would count one lesion many times.
cfg.EvalMode = "segment";

% Restrict scoring to these images ("" = all of them, the default). run_cv sets
% it to a fold's held-out images so a model is scored only on data it never
% saw; nothing else reads it, so existing runners behave exactly as before.
cfg.Images = strings(0, 1);

% --- Cross-validation ------------------------------------------------------
% Scoring the same 17 images that 13 of them trained on makes the headline
% number mostly in-sample: balanced sampling moved the reported mean +0.012
% while moving the out-of-sample mean +0.068. k folds give one out-of-sample
% score per image instead of 4 noisy ones.
cfg.CV.k        = 5;   % folds of 4/4/3/3/3 over the 17 GT images
cfg.CV.innerVal = 2;   % training images held out for best-checkpoint selection

% --- Post-processing -------------------------------------------------------
cfg.Post.MinArea      = 500;
cfg.Post.MorphElement = strel('disk', 2);

% --- Hardware --------------------------------------------------------------
% gpuDevice enumerates CUDA devices only, so an onboard Intel GPU never
% appears here and can never be picked up for compute - MATLAB's device index
% is unrelated to the GPU numbering Windows Task Manager shows. If Task
% Manager shows MATLAB busy on the integrated GPU, that is desktop/figure
% rendering (OpenGL), which is a Windows graphics preference, not this
% pipeline. cfg.GpuName is printed by run_train and run_batch so the device
% actually doing the work is on the record for every run.
cfg.UseGpu  = canUseGPU();
cfg.GpuName = "cpu";
if cfg.UseGpu
    cfg.GpuName = string(gpuDevice().Name);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function requireToolboxes()
% Fail once, clearly, instead of with a missing-function error deep inside
% training. R2025b on this machine has Image Processing only and cannot run
% this pipeline; R2026a has all four.
needed = { ...
    'image_toolbox',            'Image Processing Toolbox'; ...
    'neural_network_toolbox',   'Deep Learning Toolbox'; ...
    'video_and_image_blockset', 'Computer Vision Toolbox'};

missing = needed(~cellfun(@(f) license('test', f), needed(:, 1)), 2);
if ~isempty(missing)
    error('cnngvf:defaults:missingToolbox', ...
        ['The CNN+GVF pipeline needs: %s.\n' ...
         'Missing in this MATLAB (%s): %s.'], ...
        strjoin(needed(:, 2)', ', '), version('-release'), strjoin(missing', ', '));
end
end
