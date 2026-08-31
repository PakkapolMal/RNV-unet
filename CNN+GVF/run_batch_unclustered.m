function [perImage, summary] = run_batch_unclustered(cfg)
%RUN_BATCH_UNCLUSTERED Segment and score the DRAC22 unclustered arm. [LEGACY]
%
%  run_batch_unclustered            % equivalent to run_batch("unclustered")
%
%  Kept so existing call sites and notes keep working. DRAC22-only and frozen:
%  this is the configuration behind the measured 0.4603 cross-validated IoU,
%  and its results live in result/unclustered/.
%
%  For new work use run_batch("DRACTUH_unclustered"), which spans both datasets
%  and writes to result/DRACTUH_unclustered/.
%
%  See also RUN_BATCH, RUN_TRAIN, CNNGVF.ARM.

if nargin < 1
    cfg = cnngvf.defaults();
end
[perImage, summary] = run_batch(cnngvf.arm(cfg, "unclustered"));
end
