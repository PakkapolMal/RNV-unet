function [perImage, summary] = run_batch_unclustered(cfg)
%RUN_BATCH_UNCLUSTERED Segment and score the unclustered-points pipeline.
%
%  run_batch_unclustered              % defaults
%  run_batch_unclustered(cfg)         % custom config
%  [perImage, summary] = run_batch_unclustered(...)
%
%  The counterpart to run_train_unclustered. Three things differ from
%  run_batch:
%
%  1. Seeds come from DRAC_RNV_Points_Unclustered.csv, so inference covers the
%     lesions the clustered centres never point at.
%  2. The network is models/rnvNet_unclustered.mat.
%  3. Scoring is whole-image: the union of every seed's mask against
%     FilledGT/<stem>.png. The unclustered points have no per-point ground
%     truth, and several land on one lesion, so per-segment scoring would count
%     the same lesion once per detection and weight lesions by how many
%     detections they attracted. Whole-image scoring also charges the
%     prediction for lesions it missed entirely.
%
%  Because the unit of evaluation differs, these numbers are NOT comparable
%  with run_batch's per-segment scores. To compare the two networks fairly,
%  run both under the same EvalMode.
%
%  Results go to <Output>/unclustered/ so the clustered run is not overwritten.
%
%  See also RUN_TRAIN_UNCLUSTERED, RUN_BATCH, CNNGVF.EVALUATE.

if nargin < 1
    cfg = cnngvf.defaults();
end

cfg.Paths.Localization = fullfile(cfg.Paths.Img, 'DRAC_RNV_Points_Unclustered.csv');
cfg.Paths.Model        = fullfile(fileparts(cfg.Paths.Model), 'rnvNet_unclustered.mat');
cfg.Paths.Output       = fullfile(cfg.Paths.Output, 'unclustered');
cfg.EvalMode           = "image";

% Must match run_train_unclustered: no median filter, no vesselness channel,
% so 2 input channels rather than 3. cnngvf.inferProbMap errors on a mismatch
% instead of producing quiet nonsense.
cfg.Denoise        = [];
cfg.Vessel.enabled = false;

if ~isfile(cfg.Paths.Localization)
    error('cnngvf:runBatchUnclustered:noCsv', ...
        'Unclustered points not found: %s', cfg.Paths.Localization);
end

[perImage, summary] = run_batch(cfg);
end
