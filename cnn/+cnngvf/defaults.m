function cfg = defaults()
%CNNGVF.DEFAULTS The current configuration: DRAC22 + TUH, unclustered seeds.
%
%  cfg = cnngvf.defaults()
%
%  A seed-conditioned U-Net predicts a probability map P; the mask is P > tau,
%  a relaxed threshold around seeds P left empty, then morphology and seed
%  claiming. See docs/SPEC.md for the measurement behind each value.
%
%  This IS the "DRACTUH_unclustered" arm. cnngvf.arm overrides it for the three
%  frozen arms. Override any field before use, e.g.
%     cfg = cnngvf.defaults(); cfg.Post.tau = 0.55;
%
%  Fields
%    Datasets   Registry of the available datasets: where their images, masks
%               and localization CSVs live, and how an image name maps to its
%               mask name. See cnngvf.manifest.
%    ActiveDatasets  Which of them are in the domain.
%    Points     "clustered" | "unclustered" - which point set seeds the run.
%    Paths      Base/Main/Model/Output, plus SegGT for per-seed scoring.
%    Split      Derived validation split; every other domain image trains.
%    Vessel     Optional Frangi input channel (off).
%    Patch      CNN patch geometry, seed heat-map sigma, sampling.
%    Net        U-Net architecture and training hyper-parameters.
%    CV         Cross-validation.
%    Post       How P becomes a mask.
%    UseGpu     Run inference on the GPU. Defaults to canUseGPU().
%
%  See also CNNGVF.ARM, CNNGVF.MANIFEST, CNNGVF.SEGMENTIMAGE, RUN_TRAIN, RUN_BATCH.

requireToolboxes();

% --- Paths -----------------------------------------------------------------
% This file lives at <base>/RNV-Claude/<pkg folder>/+cnngvf/defaults.m
pkgDir  = fileparts(mfilename('fullpath'));            % .../+cnngvf
mainDir = fileparts(pkgDir);                           % the runners' folder
baseDir = fileparts(fileparts(mainDir));               % .../Fast-Gradient-Vector-Flow-main

addpath(mainDir);

cfg.Arm            = "DRACTUH_unclustered";
cfg.Paths.Base     = baseDir;
cfg.Paths.Main     = mainDir;
cfg.Paths.Model    = fullfile(mainDir, 'models', 'rnvNet_DRACTUH_unclustered.mat');
cfg.Paths.Output   = fullfile(mainDir, 'result', 'DRACTUH_unclustered');

% --- Dataset registry ------------------------------------------------------
% GtFrom/GtTo are a regexprep on the image STEM that gives the mask stem:
%   DRAC22  "^(.*)$" -> "$1"       identity
%   TUH     "^nv (.*)$" -> "gt $1"
% Resolved once by cnngvf.manifest.
drac = fullfile(baseDir, 'RNV-DRAC22NV');
tuh  = fullfile(baseDir, 'TUH');

cfg.Datasets(1).Name        = "DRAC22";
cfg.Datasets(1).Img         = drac;
cfg.Datasets(1).GT          = fullfile(baseDir, 'RNV-DRAC22GT', 'FilledGT');
% One mask per CLUSTERED seed, <stem>-<x>-<y>.png, from separate_segments.py.
% Only the "clustered" arm's EvalMode "segment" reads it.
cfg.Datasets(1).SegGT       = fullfile(baseDir, 'RNV-DRAC22GT', 'SegmentFilledGT');
cfg.Datasets(1).GtFrom      = "^(.*)$";
cfg.Datasets(1).GtTo        = "$1";
cfg.Datasets(1).Clustered   = fullfile(drac, 'DRAC22_RNV_Localization_Points_Sorted.csv');
cfg.Datasets(1).Unclustered = fullfile(drac, 'DRAC_RNV_Points_Unclustered.csv');
% Training reads a copy with the broken detections of 299.png and 445.png
% replaced by points placed from ground truth. Never at inference: that would
% be using the answer to choose the question. docs/SPEC.md, "Corrected
% training points".
cfg.Datasets(1).UnclusteredTrain = fullfile(drac, 'DRAC_RNV_Points_Unclustered_train.csv');

% ZEISS PLEX Elite. The burned-in acquisition overlay is deliberately left in
% (docs/SPEC.md, "The TUH acquisition overlay").
cfg.Datasets(2).Name        = "TUH";
cfg.Datasets(2).Img         = fullfile(tuh, 'nvImages');
cfg.Datasets(2).GT          = fullfile(tuh, 'nvGtImages');
cfg.Datasets(2).SegGT       = "";     % no per-seed ground truth exists
cfg.Datasets(2).GtFrom      = "^nv (.*)$";
cfg.Datasets(2).GtTo        = "gt $1";
cfg.Datasets(2).Clustered   = fullfile(tuh, 'TUH_RNV_Localization_Points.csv');
cfg.Datasets(2).Unclustered = fullfile(tuh, 'TU_RNV_Points_Unclustered.csv');
cfg.Datasets(2).UnclusteredTrain = "";   % "" = train on the inference points

cfg.ActiveDatasets = [cfg.Datasets.Name];
cfg.Points         = "unclustered";
cfg.Paths.SegGT    = cfg.Datasets(1).SegGT;

% Median filter kernel on load, [] for none. Off: it measured ~0.
cfg.Denoise = [];

% --- Train / validation split ---------------------------------------------
% Patient-level: whole images, never patches. Derived by cnngvf.valSplit, not
% written down - a literal list once went stale as the domain grew.
cfg.Split.NumVal = 6;
cfg.Split.Val    = strings(0, 1);   % filled in below, once Datasets exist

% Images removed from training AND validation. run_cv sets it to a fold's
% held-out images.
cfg.Split.Exclude = strings(0, 1);

% --- Vesselness channel ----------------------------------------------------
% Off: it measured ~0. A network is tied to the channel count it trained
% with, and cnngvf.inferProbMap errors on a mismatch.
cfg.Vessel.enabled = false;
cfg.Vessel.scales  = [2 4 6 8];

% --- CNN patches -----------------------------------------------------------
% 512 because 13 of 35 DRAC22 segments need a window wider than 256 px. Must
% stay divisible by 2^encoderDepth.
cfg.Patch.size      = 512;
cfg.Patch.sigmaSeed = 32;    % Gaussian seed heat-map sigma
cfg.Patch.jitter    = 32;    % +/- px random crop offset during training
cfg.Patch.perSeed   = 4;     % 1 underfit by 0.141 IoU; docs/results/

% How far outside a lesion a training seed may fall and still be labelled with
% it (true boundary distance, px). Off-lesion distances are bimodal with
% nothing in 50-100 px, so anything in 20-100 separates near-misses from
% detector errors identically.
cfg.Patch.seedTolerance = 20;

% Spend the epoch equally across IMAGES rather than seeds, so detection-rich
% images (and TUH, 70% of seeds on 40% of images) do not dominate.
cfg.Patch.balanceByImage = true;

% --- U-Net -----------------------------------------------------------------
cfg.Net.encoderDepth = 4;
cfg.Net.epochs       = 30;   % run_cv uses 20
% 4 ran out of memory at iteration 0 on an 8 GB GPU. Validation follows
% MiniBatchSize; trainingOptions has no ValidationMiniBatchSize for adam.
cfg.Net.miniBatch    = 2;
cfg.Net.inferBatch   = 4;    % patches per predict() call at inference
cfg.Net.learnRate    = 1e-3;
if batchStartupOptionUsed
    cfg.Net.plots = "none";
else
    cfg.Net.plots = "training-progress";
end

% --- Evaluation ------------------------------------------------------------
%  "image"   one row per image: union mask vs the whole FilledGT. The only
%            mode valid across both datasets and for unclustered points.
%  "segment" one row per seed vs SegmentFilledGT/<stem>-<x>-<y>.png. DRAC22 +
%            clustered points only; the "clustered" arm uses it.
cfg.EvalMode = "image";

% Restrict scoring to these images (empty = all). run_cv sets it per fold.
cfg.Images = strings(0, 1);

% --- Cross-validation ------------------------------------------------------
% Folds are pooled across datasets and stratified by lesion area; see
% cnngvf.cvFolds.
cfg.CV.k        = 5;   % 68 domain images -> folds of 14/14/14/13/13
cfg.CV.innerVal = 2;   % training images held out for best-checkpoint selection

% Whose fold networks to score with. This arm's own, except for an arm that
% differs from another only after P (cnngvf.arm sets it).
cfg.CV.ModelArm = cfg.Arm;

% Reuse existing fold networks instead of training: inference only, minutes
% instead of ~18 h, and how an interrupted run resumes. cnngvf.probCache
% re-checks the fold of every cached P before anything scores it.
cfg.CV.skipTrain = false;

% Where run_cv caches P per fold, and run_cache reads it. "" = <Output>/cv/probcache.
cfg.CV.probCache = "";

% The per-image record run_cache compares against: the adopted configuration's
% cross-validated scores. A new adoption adds a new file here; old ones stay.
cfg.CV.reference = fullfile(fileparts(mainDir), 'docs', 'results', 'reference', ...
    '2026-09-17-relax-0.20.csv');

% --- Post-processing -------------------------------------------------------
% P > tau -> seed fallback -> imclose -> imfill -> seedMasks. See
% cnngvf.maskFromProb, and docs/results/ for why each step is there.
%
% tau is PRE-REGISTERED at 0.5. It sets the output directly, so sweeping it on
% the scored images and quoting the best is leakage.
cfg.Post.tau          = 0.5;
cfg.Post.MinArea      = 500;
cfg.Post.MorphElement = strel('disk', 2);

% What a seed the threshold left empty gets - see cnngvf.seedFallback.
%  "relax"  threshold P once at relaxFloor within relaxWindow px of the seed,
%           and take the component under it. Falls back to the disk when that
%           finds nothing claimable. +0.037 IoU over the disk, cross-fitted.
%  "disk"   a disk of seedRadius. The frozen arms use it.
cfg.Post.fallbackMode = "relax";
cfg.Post.relaxWindow  = 128;
cfg.Post.relaxFloor   = 0.20;   % pre-registered; 0.5061 cross-fitted, 0.5097 fixed
cfg.Post.seedRadius   = 15;

% How a seed on background picks a component: "boundary" (nearest pixel of a
% component, the same rule training labels use) or "centroid" (frozen arms).
cfg.Post.ClaimRule = "boundary";

% --- Hardware --------------------------------------------------------------
% gpuDevice sees CUDA devices only; an integrated GPU busy in Task Manager is
% desktop rendering, not this pipeline. GpuName goes into every run record.
cfg.UseGpu  = canUseGPU();
cfg.GpuName = "cpu";
if cfg.UseGpu
    cfg.GpuName = string(gpuDevice().Name);
end

% --- Derived ---------------------------------------------------------------
% Last, because it reads the manifest. Tolerant of missing data so defaults
% stays callable for inspecting parameters.
try
    cfg.Split.Val = cnngvf.valSplit(cfg);
catch err
    warning('cnngvf:defaults:noSplit', ...
        ['Could not derive Split.Val (%s).\n' ...
         'Parameters are still usable; anything that trains or scores will ' ...
         'fail until the datasets are reachable.'], err.message);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function requireToolboxes()
% Fail once, clearly, instead of with a missing-function error deep inside
% training. R2025b on this machine has Image Processing only.
needed = { ...
    'image_toolbox',            'Image Processing Toolbox'; ...
    'neural_network_toolbox',   'Deep Learning Toolbox'; ...
    'video_and_image_blockset', 'Computer Vision Toolbox'};

missing = needed(~cellfun(@(f) license('test', f), needed(:, 1)), 2);
if ~isempty(missing)
    error('cnngvf:defaults:missingToolbox', ...
        ['This pipeline needs: %s.\n' ...
         'Missing in this MATLAB (%s): %s.'], ...
        strjoin(needed(:, 2)', ', '), version('-release'), strjoin(missing', ', '));
end
end
