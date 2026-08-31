function cfg = arm(cfg, name)
%CNNGVF.ARM Apply a named experimental configuration to cfg.
%
%  cfg = cnngvf.arm(cfg, "DRACTUH_unclustered")
%  names = cnngvf.arm()                      % list the available arms
%
%  An "arm" is one comparable configuration: which datasets are in the domain,
%  which point set seeds it, how many input channels it builds, where its model
%  and results live. Everything that must agree between training and inference
%  is set here, in ONE place.
%
%  That is deliberate. run_train and run_train_unclustered were near-copies
%  that drifted apart on Denoise and Vessel.enabled - a mismatch the network
%  cannot survive, since it is tied to the channel count it trained with - and
%  run_steps still carries the pre-ablation 3-channel settings because the
%  commit that dropped the filters updated the other two runners and not it.
%  A single arm table makes that class of drift impossible: a runner asks for
%  an arm by name and cannot disagree with any other runner about what it means.
%
%  Arms
%    "clustered"            DRAC22, clustered centres, 3 channels. The original
%                           baseline. Per-segment scoring is possible here and
%                           only here, because SegmentFilledGT is keyed to
%                           these coordinates.
%    "unclustered"          DRAC22, raw detections, 2 channels. The DRAC22-only
%                           record; frozen so its measured numbers stay
%                           reproducible.
%    "DRACTUH_unclustered"  DRAC22 + TUH, raw detections, 2 channels. The
%                           current arm.
%
%  perSeed is per-arm because it sets the epoch size, and the epoch must be
%  comparable across arms rather than across seed counts. See the note on
%  "DRACTUH_unclustered" below.
%
%  See also CNNGVF.DEFAULTS, CNNGVF.MANIFEST, CNNGVF.LOADPOINTS, RUN_CV.

table = armTable();

if nargin == 0 || isempty(name)
    cfg = string(fieldnames(table))';
    return;
end

key = char(name);
if ~isfield(table, key)
    error('cnngvf:arm:unknown', 'Unknown arm "%s". Available: %s.', ...
        name, strjoin(string(fieldnames(table))', ', '));
end
a = table.(key);

cfg.Arm            = string(key);
cfg.ActiveDatasets = a.datasets;
cfg.Points         = a.points;
cfg.EvalMode       = a.evalMode;
cfg.Denoise        = a.denoise;
cfg.Vessel.enabled = a.vessel;
cfg.Patch.perSeed  = a.perSeed;

% Built from Paths.Main rather than appended to whatever is already there, so
% applying an arm twice is a no-op instead of nesting result/x/y directories.
cfg.Paths.Model  = fullfile(cfg.Paths.Main, 'models', a.model);
cfg.Paths.Output = fullfile(cfg.Paths.Main, 'result', a.output);

% Kept in step with the arm so the legacy single-CSV consumers (rnv.loadSeeds
% called directly, cnngvf.segGtFile) still resolve. Multi-dataset code paths go
% through cnngvf.loadPoints and cnngvf.manifest instead.
primary = cfg.Datasets(arrayfun(@(d) d.Name == a.datasets(1), cfg.Datasets));
if a.points == "clustered"
    cfg.Paths.Localization = primary.Clustered;
else
    cfg.Paths.Localization = primary.Unclustered;
end
cfg.Paths.LocalizationTrain = cfg.Paths.Localization;
cfg.Paths.SegGT             = primary.SegGT;

% Split.Val is derived, never hard-coded. The old default named four DRAC22
% filenames from the 17-image era and survived two dataset changes unnoticed,
% which is exactly the failure a literal list invites.
cfg.Split.Val = cnngvf.valSplit(cfg);
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function t = armTable()
t.clustered = struct( ...
    'datasets', "DRAC22", 'points', "clustered", 'evalMode', "segment", ...
    'denoise', [3 3], 'vessel', true, 'perSeed', 16, ...
    'model', 'rnvNet.mat', 'output', '');

t.unclustered = struct( ...
    'datasets', "DRAC22", 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 4, ...
    'model', 'rnvNet_unclustered.mat', 'output', 'unclustered');

% perSeed 1, not 4. With balanceByImage the epoch is roughly perSeed x usable
% seeds, and the combined domain has 792 usable seeds against DRAC22's 234 - so
% perSeed 4 would be ~3170 patches per epoch against ~940, tripling every run
% and pushing a 5-fold CV from ~4 h to ~13 h. Nothing measured says 4 beats 1;
% it was only ever chosen to hold the epoch size constant against the clustered
% arm, which is what 1 now does. Raise it once the domain is known to train.
t.DRACTUH_unclustered = struct( ...
    'datasets', ["DRAC22", "TUH"], 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 1, ...
    'model', 'rnvNet_DRACTUH_unclustered.mat', 'output', 'DRACTUH_unclustered');
end
