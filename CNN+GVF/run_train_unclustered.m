function net = run_train_unclustered(cfg)
%RUN_TRAIN_UNCLUSTERED Train the DRAC22-only unclustered arm. [LEGACY WRAPPER]
%
%  run_train_unclustered            % equivalent to run_train("unclustered")
%
%  Kept so existing call sites and notes keep working. The arm it selects is
%  DRAC22-only and frozen: it is the configuration behind the measured 0.4603
%  cross-validated IoU, preserved so that number stays reproducible.
%
%  For new work use run_train, whose default arm is DRACTUH_unclustered
%  (DRAC22 + TUH, 68 images).
%
%  See also RUN_TRAIN, CNNGVF.ARM.

if nargin < 1
    cfg = cnngvf.defaults();
end
net = run_train("unclustered", cfg);
end
