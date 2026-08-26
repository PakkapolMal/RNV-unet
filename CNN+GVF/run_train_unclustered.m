function net = run_train_unclustered(cfg)
%RUN_TRAIN_UNCLUSTERED Train the CNN on the unclustered localization points.
%
%  run_train_unclustered            % defaults
%  run_train_unclustered(cfg)       % custom config
%  net = run_train_unclustered(...) % also return the network
%
%  The seeds in DRAC22_RNV_Localization_Points_Sorted.csv are cluster centres:
%  DBSCAN representatives of the raw detections. DRAC_RNV_Points_Unclustered.csv
%  holds the detections themselves - 184 on the 17 ground-truth images against
%  35 centres, so 5.3x the training seeds from the same images and the same
%  masks.
%
%  Two reasons this is worth a separate run:
%
%  1. More seeds is more training data on a dataset that has very little. The
%     network sees the same lesions from many offsets, which is augmentation
%     grounded in real detector output rather than synthetic jitter.
%  2. It attacks the measured bottleneck. The largest lesions dominate the
%     error (1115: recall 0.044 over 239k px; 795: 0.061 over 100k px) and no
%     single 512 px patch can cover them. The unclustered set puts 62 points
%     across 1115 and 24 across 458, so those lesions are finally seen whole -
%     as a union of overlapping patches - during training.
%
%  Evaluation is deliberately NOT changed: run_batch still segments from the
%  clustered centres and scores against SegmentFilledGT, which is keyed to
%  them. Only the training seeds differ, so the result is comparable with
%  run_train's.
%
%  Writes to models/rnvNet_unclustered.mat, leaving models/rnvNet.mat alone.
%  Point run_batch at it with:
%     cfg = cnngvf.defaults();
%     cfg.Paths.Model = fullfile(fileparts(which('run_batch')), ...
%                                'models', 'rnvNet_unclustered.mat');
%     run_batch(cfg);
%
%  See also RUN_TRAIN, CNNGVF.PREPAREDATA, RUN_BATCH.

if nargin < 1
    cfg = cnngvf.defaults();
end

cfg.Paths.LocalizationTrain = fullfile(cfg.Paths.Img, 'DRAC_RNV_Points_Unclustered.csv');
cfg.Paths.Model = fullfile(fileparts(cfg.Paths.Model), 'rnvNet_unclustered.mat');

% No median filter, no vesselness channel: the network sees raw intensity plus
% the seed heat-map, 2 inputs instead of 3. run_batch_unclustered sets the same
% two flags - they must match or inference errors on the channel count.
cfg.Denoise        = [];
cfg.Vessel.enabled = false;

% 137 usable training seeds instead of 30, so keep the epoch the same size and
% let the extra seeds buy variety rather than a 4.6x longer run: 137 x 4 = 548
% patches per epoch against run_train's 30 x 16 = 480. Total samples over 30
% epochs stay comparable (16.4k vs 14.4k) and so does the ~2.5 h runtime.
% Without this an epoch is 1096 iterations and the run takes ~11 h.
cfg.Patch.perSeed = 4;

if ~isfile(cfg.Paths.LocalizationTrain)
    error('cnngvf:runTrainUnclustered:noCsv', ...
        'Unclustered points not found: %s', cfg.Paths.LocalizationTrain);
end

fprintf('Training seeds : %s\n', cfg.Paths.LocalizationTrain);
fprintf('Scored against : %s (unchanged)\n', cfg.Paths.Localization);

t   = tic;
net = cnngvf.trainNet(cfg);
fprintf('Training finished in %.1f min.\n', toc(t) / 60);
end
