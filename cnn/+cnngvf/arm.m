function cfg = arm(cfg, name)
%CNNGVF.ARM Apply one of the frozen experimental configurations to cfg.
%
%  cfg = cnngvf.arm(cnngvf.defaults(), "unclustered")
%  names = cnngvf.arm()                      % list the available arms
%
%  An "arm" is one comparable configuration: which datasets are in the domain,
%  which point set seeds it, how many input channels it builds, where its model
%  and results live. cnngvf.defaults() IS the current arm,
%  "DRACTUH_unclustered"; asking for it by name here returns cfg unchanged.
%
%  The others are frozen: they set everything that differs from the defaults,
%  so their networks keep loading. Their recorded numbers were measured with
%  the level set, which is gone - they now run the threshold pipeline with the
%  disk fallback, so they no longer reproduce those records. Tag
%  archive/pre-cleanup does.
%
%  Arms
%    "clustered"            DRAC22, clustered centres, 3 channels. The original
%                           baseline; per-segment scoring is valid only here.
%    "unclustered"          DRAC22, raw detections, 2 channels.
%    "unclustered_boundary" "unclustered" with the "boundary" claim rule.
%                           Borrows that arm's fold networks, so the two pair
%                           image by image.
%
%  Always apply to a fresh cnngvf.defaults(): an arm only sets what differs.
%
%  See also CNNGVF.DEFAULTS, CNNGVF.MANIFEST, CNNGVF.LOADPOINTS, RUN_CV.

table = armTable();

if nargin == 0 || isempty(name)
    cfg = ["DRACTUH_unclustered", string(fieldnames(table))'];
    return;
end

key = char(name);
if key == "DRACTUH_unclustered"
    return;
end
if ~isfield(table, key)
    error('cnngvf:arm:unknown', 'Unknown arm "%s". Available: %s.', ...
        name, strjoin(cnngvf.arm(), ', '));
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
cfg.Post.ClaimRule    = a.claimRule;
cfg.Post.fallbackMode = "disk";
cfg.Post.hullLink     = 0;       % the seed hull came after every frozen arm

cfg.CV.reference = "";   % frozen arms have no reference on master
cfg.CV.ModelArm = a.cvModelArm;
if strlength(cfg.CV.ModelArm) == 0
    cfg.CV.ModelArm = cfg.Arm;
end

cfg.Paths.Model  = fullfile(cfg.Paths.Main, 'models', a.model);
cfg.Paths.Output = fullfile(cfg.Paths.Main, 'result', a.output);

% Per-seed ground truth belongs to the arm's first dataset.
primary = cfg.Datasets(arrayfun(@(d) d.Name == a.datasets(1), cfg.Datasets));
cfg.Paths.SegGT = primary.SegGT;

% The split depends on the domain, so it must be re-derived here.
cfg.Split.Val = cnngvf.valSplit(cfg);
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function t = armTable()
% seedTolerance 0 and claimRule "centroid" are what these arms were trained
% and measured with.
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

t.unclustered_boundary = struct( ...
    'datasets', "DRAC22", 'points', "unclustered", 'evalMode', "image", ...
    'denoise', [], 'vessel', false, 'perSeed', 4, 'seedTolerance', 0, ...
    'claimRule', "boundary", 'cvModelArm', "unclustered", ...
    'model', 'rnvNet_unclustered.mat', 'output', 'unclustered_boundary');
end
