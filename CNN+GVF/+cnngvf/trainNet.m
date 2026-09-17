function net = trainNet(cfg)
%CNNGVF.TRAINNET Train the seed-conditioned U-Net that produces P.
%
%  net = cnngvf.trainNet(cfg)
%
%  Trains on the datastores from cnngvf.prepareData and saves the result to
%  cfg.Paths.Model. Returns the trained dlnetwork.
%
%  Loss is 1 - generalizedDice. RNV occupies a tiny fraction of every patch;
%  generalized Dice weights each class by the inverse square of its expected
%  volume, which is what stops the network collapsing to all-background
%  (../docs/spec.md, "Extreme class imbalance in CNN training").
%
%  ExecutionEnvironment="auto" picks up the GPU when there is one.
%
%  See also CNNGVF.PREPAREDATA, CNNGVF.INFERPROBMAP, RUN_TRAIN.

[dsTrain, dsVal, info] = cnngvf.prepareData(cfg);

fprintf('Training on %d images / %d seeds (%d patches per epoch)\n', ...
    numel(info.trainImages), info.trainSeeds, info.patchesPerEpoch);
fprintf('Validating on %d images / %d seeds: %s\n', ...
    numel(info.valImages), info.valSeeds, strjoin(info.valImages, ', '));
fprintf('Compute device: %s\n', cfg.GpuName);

% Clear anything a previous run left on the card. Training at patch 512 has
% little headroom on an 8 GB laptop GPU, and a stale allocation is the
% difference between finishing and an out-of-memory at iteration 0.
if cfg.UseGpu
    gpuDevice(1);
end

chanDesc = "intensity";
if cfg.Vessel.enabled
    chanDesc = chanDesc + " + vesselness";
end
if cfg.Patch.seeded
    chanDesc = chanDesc + " + seed heat-map";
else
    chanDesc = chanDesc + ", seedless tiles";
end
fprintf('Input channels : %d (%s)\n', info.numChannels, chanDesc);
if isempty(cfg.Denoise)
    fprintf('Denoise        : none\n');
else
    fprintf('Denoise        : median %s\n', mat2str(cfg.Denoise));
end

net = unet([cfg.Patch.size cfg.Patch.size info.numChannels], 2, ...
    EncoderDepth = cfg.Net.encoderDepth);

opts = trainingOptions("adam", ...
    InitialLearnRate    = cfg.Net.learnRate, ...
    MaxEpochs           = cfg.Net.epochs, ...
    MiniBatchSize       = cfg.Net.miniBatch, ...
    Shuffle             = "every-epoch", ...
    ValidationData      = dsVal, ...
    ValidationFrequency = max(1, floor(info.patchesPerEpoch / cfg.Net.miniBatch)), ...
    OutputNetwork       = "best-validation", ...
    ExecutionEnvironment = "auto", ...
    Plots               = cfg.Net.plots, ...
    Verbose             = true);

net = trainnet(dsTrain, net, @(Y, T) 1 - mean(generalizedDice(Y, T), "all"), opts);

modelDir = fileparts(cfg.Paths.Model);
if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end

% Keep the previous network. A retrain is hours, and a configuration change
% that turns out to be a regression should not also destroy the model it was
% measured against - which is exactly what happened to the patch-256 network.
if isfile(cfg.Paths.Model)
    [~, stem, ext] = fileparts(cfg.Paths.Model);
    prev = dir(cfg.Paths.Model);
    archived = fullfile(modelDir, sprintf('%s_%s%s', stem, ...
        char(datetime(prev.datenum, 'ConvertFrom', 'datenum', 'Format', 'yyyyMMdd_HHmmss')), ext));
    movefile(cfg.Paths.Model, archived);
    fprintf('Previous network archived to %s\n', archived);
end

save(cfg.Paths.Model, 'net', 'cfg', 'info');
fprintf('Saved network to %s\n', cfg.Paths.Model);
end
