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
%    Datasets   Registry of the available datasets: where their images, masks
%               and localization CSVs live, and how an image name maps to its
%               mask name. See cnngvf.manifest.
%    ActiveDatasets  Which of them are in the domain right now. Set by
%               cnngvf.arm; defaults to all.
%    Points     "clustered" | "unclustered" - which point set seeds the run.
%    Paths      Base/Model/Output, plus single-CSV fields kept in step by
%               cnngvf.arm for the legacy consumers (absolute, resolved from
%               this file - no hard-coded drive letters).
%    Split      Derived validation split; every other domain image trains.
%    Patch      CNN patch geometry and seed heat-map sigma.
%    Net        U-Net architecture and training hyper-parameters.
%    Force      How P becomes the GVF energy field f.
%    Gvf        GVF diffusion parameters (Xu & Prince).
%    Drlse      Level-set parameters (Li et al. 2010).
%    Post       Mask cleanup.
%    UseGpu     Run GVF/DRLSE on the GPU. Defaults to canUseGPU().
%
%  Defaults come from the parameter table in ../docs/spec.md; the measured
%  justification for each is in ../docs/UNCLUSTERED.md.
%
%  See also CNNGVF.ARM, CNNGVF.MANIFEST, CNNGVF.SEGMENTIMAGE, RUN_TRAIN, RUN_BATCH.

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
cfg.Paths.Model        = fullfile(mainDir, 'models', 'rnvNet.mat');
cfg.Paths.Output       = fullfile(mainDir, 'result');

% --- Dataset registry ------------------------------------------------------
% An image used to BE its filename: one string keyed the image, the mask, the
% CSV rows and the output names, because image and mask happened to share a
% stem in one directory each. TUH pairs nvImages/nv (N).png with
% nvGtImages/gt (N).png, so the image -> mask mapping is per-dataset data.
%
% GtFrom/GtTo are a regexprep on the STEM (extension is carried over):
%   DRAC22  "^(.*)$" -> "$1"       identity
%   TUH     "^nv (.*)$" -> "gt $1"
%
% Resolved once by cnngvf.manifest, which every stage reads instead of building
% paths with its own fullfile call.
drac = fullfile(baseDir, 'RNV-DRAC22NV');
tuh  = fullfile(baseDir, 'TUH');

cfg.Datasets(1).Name        = "DRAC22";
cfg.Datasets(1).Img         = drac;
cfg.Datasets(1).GT          = fullfile(baseDir, 'RNV-DRAC22GT', 'FilledGT');
% One mask per localization seed, named <stem>-<x>-<y>.png. Written by
% separate_segments.py, which keeps the ground-truth component each seed falls
% in and drops the components no seed claims. DRAC22 only, and keyed to the
% CLUSTERED coordinates - which is why EvalMode "segment" is fenced to this
% dataset and that point set.
cfg.Datasets(1).SegGT       = fullfile(baseDir, 'RNV-DRAC22GT', 'SegmentFilledGT');
cfg.Datasets(1).GtFrom      = "^(.*)$";
cfg.Datasets(1).GtTo        = "$1";
cfg.Datasets(1).Clustered   = fullfile(drac, 'DRAC22_RNV_Localization_Points_Sorted.csv');
cfg.Datasets(1).Unclustered = fullfile(drac, 'DRAC_RNV_Points_Unclustered.csv');

% OCTA en-face like DRAC22, but a different device (ZEISS PLEX Elite) with a
% different FOV crop. Every TUH image carries a burned-in acquisition overlay -
% a magenta vertical and cyan horizontal crosshair spanning the frame, plus a
% "PLEX Elite" watermark bottom-right. These are deliberately NOT removed; see
% ../docs/UNCLUSTERED.md, "The TUH acquisition overlay".
cfg.Datasets(2).Name        = "TUH";
cfg.Datasets(2).Img         = fullfile(tuh, 'nvImages');
cfg.Datasets(2).GT          = fullfile(tuh, 'nvGtImages');
cfg.Datasets(2).SegGT       = "";     % no per-seed ground truth exists
cfg.Datasets(2).GtFrom      = "^nv (.*)$";
cfg.Datasets(2).GtTo        = "gt $1";
cfg.Datasets(2).Clustered   = fullfile(tuh, 'TUH_RNV_Localization_Points.csv');
cfg.Datasets(2).Unclustered = fullfile(tuh, 'TU_RNV_Points_Unclustered.csv');

% Everything by default; cnngvf.arm narrows it.
cfg.ActiveDatasets = [cfg.Datasets.Name];
cfg.Points         = "unclustered";
cfg.Arm            = "";

% Kept in step with the arm by cnngvf.arm, for the consumers that still take a
% single CSV (rnv.loadSeeds called directly, cnngvf.segGtFile). Multi-dataset
% paths go through cnngvf.loadPoints and cnngvf.manifest.
cfg.Paths.Localization      = cfg.Datasets(1).Unclustered;
cfg.Paths.LocalizationTrain = cfg.Paths.Localization;
cfg.Paths.GT                = cfg.Datasets(1).GT;
cfg.Paths.Img               = cfg.Datasets(1).Img;
cfg.Paths.SegGT             = cfg.Datasets(1).SegGT;

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
% split.
%
% Derived by cnngvf.valSplit, not written down. The old default listed four
% DRAC22 filenames from the 17-image era and then survived the dataset growing
% to 41 and TUH arriving - still validating on those same four images, and on
% none of TUH. A literal list cannot notice it has gone stale.
cfg.Split.NumVal = 6;
cfg.Split.Val    = strings(0, 1);   % filled in below, once Datasets exist

% Images removed from training AND validation entirely. run_cv sets this to a
% fold's held-out test images, so they are never seen by training or by
% best-checkpoint selection. Empty by default, so the existing runners are
% unaffected.
cfg.Split.Exclude = strings(0, 1);

% --- Vesselness channel ----------------------------------------------------
% Second CNN input channel. Small thicknesses: RNV vessels are thin, and
% ../docs/CLAUDE.md notes large scales blur the structure away entirely.
%
% Vessel.enabled = false drops the channel, leaving the network 2 inputs
% (intensity + seed heat-map) instead of 3. The channel count is derived
% everywhere from cnngvf.loadChannels, so this is the only place to change it -
% but a network trained with one setting CANNOT be used with the other, and
% cnngvf.inferProbMap will error on the size mismatch rather than misbehave.
cfg.Vessel.enabled = true;
cfg.Vessel.scales  = [2 4 6 8];

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

% How far outside a lesion a training seed may fall and still be labelled with
% it, in pixels of true boundary distance.
%
% 0 reproduces the original rule: a seed is labelled only by the component it
% is strictly inside, and anything else is dropped. That threw away 81 of 874
% points - but they are not one population. Measured, their distance to the
% nearest lesion is sharply bimodal, with NOTHING at all between 50 and 100 px:
%
%      <=5 px   36        100-200 px    7
%     5-10 px    8           >200 px   21   (max 843)
%    10-20 px    5
%    20-50 px    4        50-100 px     0   <- clean separator
%
% The near group are boundary near-misses, not localization errors, and
% dropping them discarded ~6% of the usable training set on a dataset that has
% little to spare. The far group are genuine detector false positives -
% confirmed by eye on 299.png (4 detections 307-726 px away while the real
% lesion went undetected) and 419.png (one at 602 px on plain background).
%
% 20 sits inside the +/-32 px crop jitter the network already sees, and the
% empty 50-100 px band means the exact value does not matter: anything in
% 20-100 recovers the same near group and rejects the same far group.
cfg.Patch.seedTolerance = 20;

% Spend the epoch equally across IMAGES rather than equally across seeds. Only
% matters when seed counts are uneven, i.e. for the unclustered detections; it
% is a no-op for the clustered centres, where every image has 1-7. See
% cnngvf.prepareData for the measurement that motivated it.
%
% It matters more on the combined domain, and in a second way. TUH holds 558 of
% the 792 usable detections (70%) across 27 of the 68 images (40%), so
% per-image balancing gives TUH 40% of the epoch rather than 70%. That is the
% intended behaviour - an image is one observation regardless of how many times
% the detector fired on it - but it is a choice, and the opposite choice
% (weight by detection count) is one flag away.
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
%            DRAC22 + clustered points ONLY - TUH ships no per-seed masks, and
%            cnngvf.evaluate errors rather than scoring against nothing. Kept
%            for reproducing the original baseline; not used by any current arm.
%  "image"   one row per image: the union of all seeds' masks vs the whole
%            FilledGT/<stem>.png. Required for the unclustered points, which
%            have no per-point ground truth and several points per lesion -
%            scoring them individually would count one lesion many times. The
%            only mode valid across both datasets.
cfg.EvalMode = "image";

% Restrict scoring to these images ("" = all of them, the default). run_cv sets
% it to a fold's held-out images so a model is scored only on data it never
% saw; nothing else reads it, so existing runners behave exactly as before.
cfg.Images = strings(0, 1);

% --- Cross-validation ------------------------------------------------------
% Scoring images that trained the model makes the headline mostly in-sample:
% balanced sampling once moved the reported mean +0.012 while moving the
% out-of-sample mean +0.068 - the number was hiding the result. k folds give
% one out-of-sample score per image.
%
% Folds are pooled across datasets and stratified by lesion area (see
% cnngvf.cvFolds). Pooling is deliberate: the research question is RNV
% segmentation, not per-device performance, so a fold is not restricted to one
% device. per_image.csv still carries a Dataset column, so a pooled run can be
% broken out afterwards without re-running it.
cfg.CV.k        = 5;   % 68 domain images -> folds of 14/14/14/13/13
cfg.CV.innerVal = 2;   % training images held out for best-checkpoint selection

% --- Post-processing -------------------------------------------------------
cfg.Post.MinArea      = 500;
cfg.Post.MorphElement = strel('disk', 2);

% How a seed that lands on background picks a component: "boundary" (distance
% to the nearest pixel of a component) or "centroid" (distance to its centre of
% mass).
%
% "boundary" is the fixed rule and the default. It matches how
% cnngvf.prepareData labels TRAINING seeds, so the pipeline no longer answers
% "which lesion does this seed belong to" two different ways at the two ends.
% Centroid distance misassigns seeds around large concave lesions - a seed just
% outside one can be nearer another blob's centroid than its own - which is the
% bug RNV-Claude/main's selectSeedComponent was fixed for.
%
% cnngvf.arm pins the frozen arms back to "centroid": they are the record
% behind published numbers, and silently rescoring them would make those
% numbers unreproducible from this code.
cfg.Post.ClaimRule    = "boundary";

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

% --- Derived ---------------------------------------------------------------
% Last, because it reads the manifest and so needs every field above. Tolerant
% of a missing dataset: cnngvf.defaults must stay callable for test_cnngvf and
% for inspecting parameters on a machine that has no data mounted.
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
