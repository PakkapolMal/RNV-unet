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
%    "unclustered_boundary" "unclustered" with the seed claiming rule fixed and
%                           nothing else changed. Borrows that arm's networks,
%                           so it pairs with the frozen record image by image.
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
cfg.Patch.seedTolerance = a.seedTolerance;
cfg.Post.ClaimRule = a.claimRule;

% Optional per-arm fields, so the arms that predate them need no edit. seeded is
% always written, so moving from a seedless arm back to a seeded one cannot
% leave the network input stale.
cfg.Patch.seeded = ~isfield(a, 'seeded') || a.seeded;
if isfield(a, 'post')
    for f = string(fieldnames(a.post))'
        cfg.Post.(f) = a.post.(f);
    end
end

% Which arm's cross-validation networks this arm scores with. Normally its own.
% "unclustered_boundary" borrows "unclustered"'s, because it differs from it
% only in post-processing - no trainable parameter is involved, so retraining
% would destroy the pairing and buy nothing.
cfg.CV.ModelArm = a.cvModelArm;
if strlength(cfg.CV.ModelArm) == 0
    cfg.CV.ModelArm = cfg.Arm;
end

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
% seedTolerance 0 on both frozen arms: they are the record behind published
% numbers, and labelling more seeds than they were measured with would make
% them silently unreproducible. Only the current arm gets the tolerance.
%
% claimRule "centroid" on both frozen arms for the same reason: cnngvf.seedMasks
% now defaults to the fixed "boundary" rule, and letting that reach an arm whose
% numbers are already published would rescore the record without saying so.
t.clustered = struct( ...
    'datasets', "DRAC22", 'points', "clustered", 'evalMode', "segment", ...
    'denoise', [3 3], 'vessel', true, 'perSeed', 16, 'seedTolerance', 0, ...
    'claimRule', "centroid", 'cvModelArm', "", ...
    'model', 'rnvNet.mat', 'output', '');

t.unclustered = struct( ...
    'datasets', "DRAC22", 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 4, 'seedTolerance', 0, ...
    'claimRule', "centroid", 'cvModelArm', "", ...
    'model', 'rnvNet_unclustered.mat', 'output', 'unclustered');

% "unclustered" with the claiming rule fixed, and NOTHING else changed. Same
% domain, same point set, same channels, same networks - models/cv/unclustered_
% fold<k>.mat, borrowed via cvModelArm rather than retrained - so it pairs with
% the frozen 0.4603 record image by image, and the delta between them measures
% the claiming rule alone.
%
% This is also the arm the reduced-pipeline ablation runs on: REDUCE.md requires
% the claiming rule to be fixed BEFORE the level set is measured, so that the
% two changes are not confounded.
t.unclustered_boundary = struct( ...
    'datasets', "DRAC22", 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 4, 'seedTolerance', 0, ...
    'claimRule', "boundary", 'cvModelArm', "unclustered", ...
    'model', 'rnvNet_unclustered.mat', 'output', 'unclustered_boundary');

% perSeed 4, raised from 1 after measuring what 1 cost.
%
% With balanceByImage the epoch is roughly perSeed x usable seeds, so on this
% domain's 842 usable seeds perSeed 1 gave ~780 patches per epoch against the
% DRAC22-only arm's ~940. It was chosen to hold runtime down, and nothing
% measured supported it. It underfit:
%
%   - validation loss flat across 30 epochs (0.421 -> 0.389, wobble +/-0.03)
%   - single-split IoU 0.3852 over 68 images, of which 62 trained the model -
%     and its DRAC22 subset scored 0.379 IN-sample against the DRAC22-only
%     arm's 0.4603 cross-validated. An in-sample score below another model's
%     out-of-sample score is a regression, not noise.
%   - lesions over 100k px: precision 0.800 but recall 0.247. nv (21).png has
%     113 seeds tiling a 432k px lesion and still recalled 12%, so the network
%     was under-responding rather than running out of patch coverage.
%
% 4 restores the per-seed exposure of the other arms (~3370 patches/epoch).
% Costs ~4x the runtime: ~5 h to train, and a 5-fold CV goes from ~4.5 h to
% ~18 h at 20 epochs. Record in result/diagnostic_perSeed1/.
t.DRACTUH_unclustered = struct( ...
    'datasets', ["DRAC22", "TUH"], 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 4, 'seedTolerance', 20, ...
    'claimRule', "boundary", 'cvModelArm', "", ...
    'model', 'rnvNet_DRACTUH_unclustered.mat', 'output', 'DRACTUH_unclustered');

% "DRACTUH_unclustered" with the seed taken OUT of the network (../docs/SEEDLESS.md).
% Same domain, points, channels, folds and training recipe; P comes from 512
% tiles over the whole frame instead of seed-centred patches, and seeds enter
% only at claiming. Post-processing is the reduced pipeline with the relax
% fallback at its confirmed floor, so it pairs against REDUCE_RESULTS.md section 9.
%
% perSeed and seedTolerance still matter: they size the epoch exactly as the
% seeded arm's would, so the two spend the same compute per fold.
t.DRACTUH_seedless = struct( ...
    'datasets', ["DRAC22", "TUH"], 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 4, 'seedTolerance', 20, ...
    'claimRule', "boundary", 'cvModelArm', "", ...
    'model', 'rnvNet_DRACTUH_seedless.mat', 'output', 'DRACTUH_seedless', ...
    'seeded', false, ...
    'post', struct('mode', "threshold", 'fallbackMode', "relax", 'relaxFloor', 0.20));
end
