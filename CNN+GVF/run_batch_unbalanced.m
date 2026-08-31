function [perImage, summary] = run_batch_unbalanced(cfg)
%RUN_BATCH_UNBALANCED Score the archived pre-balancing unclustered network.
%
%  run_batch_unbalanced           % defaults
%  run_batch_unbalanced(cfg)      % custom config
%  [perImage, summary] = run_batch_unbalanced(...)
%
%  Scores models/rnvNet_unclustered_20260814_231145.mat - the network trained
%  from the unclustered detections BEFORE cfg.Patch.balanceByImage existed, so
%  with the epoch spent uniformly per seed rather than per image. Its stored
%  cfg has no balanceByImage field and info.patchesPerEpoch = 548 = 137 seeds
%  x 4, which is how you can tell it apart from the current network.
%
%  It is kept so the balancing change can be re-measured on demand:
%  cnngvf.trainNet archives the previous network instead of overwriting it,
%  and this is that archive.
%
%  Everything else matches run_batch_unclustered - same unclustered seeds, same
%  whole-image scoring - so the two are directly comparable, both single-split.
%  Results go to <Output>/unclustered_unbalanced/, leaving
%  <Output>/unclustered/ (the balanced run) untouched.
%
%  Note both are SINGLE-SPLIT numbers: 13 of the 17 images trained the model,
%  so neither is a valid estimate of performance and neither is comparable with
%  a cross-validated figure. See ../docs/UNCLUSTERED.md, "Reporting rules".
%
%  See also RUN_BATCH_UNCLUSTERED, RUN_BATCH, RUN_CV.

if nargin < 1
    cfg = cnngvf.defaults();
end

% DRAC22-only: the archived network predates TUH entirely, so scoring it over
% the combined domain would charge it for 27 images it was never shown.
cfg = cnngvf.arm(cfg, "unclustered");

archived = fullfile(fileparts(cfg.Paths.Model), 'rnvNet_unclustered_20260814_231145.mat');
if ~isfile(archived)
    error('cnngvf:runBatchUnbalanced:noModel', ...
        ['Archived pre-balancing network not found:\n  %s\n' ...
         'It is produced by trainNet archiving the previous network. If it is ' ...
         'gone, retrain with cfg.Patch.balanceByImage = false.'], archived);
end

cfg.Paths.Model  = archived;
cfg.Paths.Output = fullfile(fileparts(cfg.Paths.Output), 'unclustered_unbalanced');

fprintf('Scoring the PRE-BALANCING network (epoch spent per seed, not per image)\n');
fprintf('Model: %s\n\n', archived);

[perImage, summary] = run_batch(cfg);
end
