function cfg = defaults()
%CNNGVF.DEFAULTS The current configuration: DRAC22 + TUH, unclustered seeds.
%
%  cfg = cnngvf.defaults()
%
%  A seed-conditioned U-Net predicts a probability map P; the mask is P > tau,
%  a relaxed threshold around seeds P left empty, then morphology and seed
%  claiming. See docs/SPEC.md for the measurement behind each value.
%
%  This IS the "DRACTUH_unclustered" arm. cnngvf.arm overrides it for the two
%  frozen arms. Override any field before use, e.g.
%     cfg = cnngvf.defaults(); cfg.Post.tau = 0.55;
%
%  Fields
%    Datasets   Registry of the available datasets: where their images, masks
%               and localization CSVs live, and how an image name maps to its
%               mask name. See cnngvf.manifest.
%    ActiveDatasets  Which of them are in the domain.
%    Paths      Base/Main/Model/Output.
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
% This file lives at RNV-Claude/<pkg folder>/+cnngvf/defaults.m
pkgDir  = fileparts(mfilename('fullpath'));            % .../+cnngvf
mainDir = fileparts(pkgDir);                           % the runners' folder
dataDir = fullfile(fileparts(mainDir), 'datasets');    % RNV-Claude/datasets, read-only

addpath(mainDir);

cfg.Arm            = "DRACTUH_unclustered";
cfg.Paths.Main     = mainDir;
cfg.Paths.Model    = fullfile(mainDir, 'models', 'rnvNet_DRACTUH_unclustered.mat');
cfg.Paths.Output   = fullfile(mainDir, 'result', 'DRACTUH_unclustered');

% --- Dataset registry ------------------------------------------------------
% The standardized datasets shared with the peer comparisons. Never modified:
% CLAUDE.md, "Datasets". GtFrom/GtTo are a regexprep on the image FILE NAME
% that gives the mask file name:
%   DRAC22  "^(.*)$" -> "$1"                    identity
%   TUH     "^nv (.*)\.png$" -> "gt $1.tif"
% Resolved once by cnngvf.manifest.
cfg.Datasets(1).Name        = "DRAC22";
cfg.Datasets(1).Img         = fullfile(dataDir, 'DRAC22_raw');
cfg.Datasets(1).GT          = fullfile(dataDir, 'DRAC22_filled_gt');
cfg.Datasets(1).GtFrom      = "^(.*)$";
cfg.Datasets(1).GtTo        = "$1";
cfg.Datasets(1).Unclustered = fullfile(dataDir, 'DRAC_RNV_Points_Unclustered.csv');

% ZEISS PLEX Elite. The burned-in acquisition overlay is deliberately left in
% (docs/SPEC.md, "The TUH acquisition overlay").
cfg.Datasets(2).Name        = "TUH";
cfg.Datasets(2).Img         = fullfile(dataDir, 'TUH_raw');
cfg.Datasets(2).GT          = fullfile(dataDir, 'TUH_filled_gt');
cfg.Datasets(2).GtFrom      = "^nv (.*)\.png$";
cfg.Datasets(2).GtTo        = "gt $1.tif";
cfg.Datasets(2).Unclustered = fullfile(dataDir, 'TU_RNV_Points_Unclustered.csv');

cfg.ActiveDatasets = [cfg.Datasets.Name];

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
% Which trainer produces this configuration's networks. "pytorch": colab/
% rnvtorch.py trains it, through run_export's bundle, and MATLAB refuses to
% (cnngvf.trainNet, run_cv). Frozen arms set "matlab" (cnngvf.arm).
cfg.Net.trainer      = "pytorch";
% The adopted recipe, read by colab/rnvtorch.py: an ImageNet-pretrained U-Net
% with this encoder, CE + Dice, affine + CLAHE augmentation (+0.087 IoU,
% docs/results/2026-11-11-paper-recipe.md); epochsCv epochs, learning rate
% cosine-decayed from learnRate to lrMin, last epoch kept, no inner validation
% (+0.025 IoU, docs/results/2026-11-07-train-budget.md). The frozen arms keep
% the MATLAB recipe (cnngvf.arm).
cfg.Net.encoder      = "efficientnet-b3";
cfg.Net.epochsCv     = 40;
cfg.Net.lrMin        = 1e-6;
cfg.Net.epochs       = 30;   % MATLAB trainer only (frozen arms); run_cv uses 20
% 8: the encoder's BatchNorm needs more than 2 per batch.
cfg.Net.miniBatch    = 8;
cfg.Net.inferBatch   = 4;    % patches per predict() call at inference
cfg.Net.learnRate    = 1e-4;
% The seedless network (rnvtorch.py train-seedless, cnngvf.noPointMask): whole
% frames, intensity only, each training image seedlessCopies times per epoch.
cfg.Net.seedlessEpochs    = 60;
cfg.Net.seedlessMiniBatch = 4;
cfg.Net.seedlessCopies    = 10;
if batchStartupOptionUsed
    cfg.Net.plots = "none";
else
    cfg.Net.plots = "training-progress";
end

% --- Evaluation ------------------------------------------------------------
% Restrict scoring to these images (empty = all). run_cv sets it per fold.
cfg.Images = strings(0, 1);

% --- Cross-validation ------------------------------------------------------
% Folds are pooled across datasets and stratified by lesion area; see
% cnngvf.cvFolds.
cfg.CV.k        = 5;   % 69 domain images -> folds of 14/14/14/14/13
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
    '2026-11-12-seedless-fallback-69.csv');

% --- Post-processing -------------------------------------------------------
% P > tau -> seed fallback -> imclose -> imfill -> seedMasks. See
% cnngvf.maskFromProb, and docs/results/ for why each step is there.
%
% tau is PRE-REGISTERED at 0.5. It sets the output directly, so sweeping it on
% the scored images and quoting the best is leakage. Cross-fitting confirms 0.5;
% docs/results/2026-11-05-tau-crossfit.md.
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

% Seed hull, see cnngvf.seedHull: after claiming, fill Delaunay triangles whose
% seeds are pairwise within hullLink px; 0 = off. 128 was cross-fitted over
% 64-256 and adopted over its precision guard; docs/results/2026-09-30-seed-hull-d.md.
cfg.Post.hullLink = 128;

% An image with zero localization points takes cnngvf.noPointMask of the
% seedless network's P, read from noPointCache; "" = off, and such an image
% predicts nothing (the frozen arms). noPointTau was cross-fitted on the images
% with 1-3 points; 0.3-0.9 all score within 0.011 on the targets.
% docs/results/2026-11-12-seedless-fallback.md.
cfg.Post.noPointCache = fullfile(cfg.Paths.Output, 'cv', 'probcache_seedless');
cfg.Post.noPointTau   = 0.40;

% EXPERIMENT (docs/briefs/fusion-uncertainty.md): an image with points whose
% seeded P is unsure (cnngvf.seededUncertainty >= fuseCut) takes the pixel max
% of its seeded P and the seedless P above fuseTauS (cnngvf.fuseUncertain).
% Inf = off; -Inf fuses every image with points.
cfg.Post.fuseCut  = Inf;
cfg.Post.fuseTauS = 0.9;

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
