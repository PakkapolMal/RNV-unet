function out = probCache(action, cacheDir, imgName, P, meta, V)
%CNNGVF.PROBCACHE Persist the CNN probability map so variants cost seconds.
%
%  cnngvf.probCache("write", dir, imgName, P, meta)
%  cnngvf.probCache("write", dir, imgName, P, meta, V)   % V: TTA variance, stored single
%  [P, meta] = cnngvf.probCache("read", dir, imgName, cfg)
%  names     = cnngvf.probCache("list", dir)
%
%  Nothing used to persist P, so every downstream variant - a different
%  threshold, a different post-processing switch, the whole reduced pipeline -
%  cost a full cross-validation to evaluate. That is ~18 h at perSeed 4 to
%  measure a change with no trainable parameter in it. Cached, the same
%  comparison is seconds and needs no GPU, and every variant sees byte-identical
%  P, which makes the comparison perfectly paired: training variance is exactly
%  zero.
%
%  Stored as single, one .mat per image, ~4 MB each (68 images ~ 285 MB per CV
%  run). NOT uint8: 1/255 quantization is coarse against a fine tau sweep, and
%  tau now sets the output directly rather than merely seeding phi0.
%  run_batch's prob/<stem>.png dump stays as the human-viewable artefact and is
%  unrelated to this.
%
%  meta records fold, arm, the network file, a timestamp and the localization
%  CSV mtimes - the last because every score is tied to the CSVs in place when
%  it was measured, and those live outside version control and have been revised
%  repeatedly.
%
%  LEAKAGE GUARD. "read" takes cfg and asserts the cached P carries a fold index
%  whose held-out set contains that image. A P produced by a network that
%  trained on its own image is in-sample, and an in-sample number quoted as
%  cross-validated is the single easiest way to make this whole exercise
%  worthless. It fails loudly instead of scoring.
%
%  See also RUN_ABLATE, RUN_CV, CNNGVF.CVFOLDS, CNNGVF.INFERPROBMAP.

switch string(action)
    case "write"
        if nargin < 6, V = []; end
        out = writeOne(cacheDir, imgName, P, meta, V);

    case "read"
        % The 4th positional argument is cfg on a read - it is what the leakage
        % guard needs to recompute the folds.
        out = readOne(cacheDir, imgName, P);

    case "list"
        d = dir(fullfile(cacheDir, '*.mat'));
        out = string({d.name})';
        out = erase(out, ".mat") + ".png";

    otherwise
        error('cnngvf:probCache:badAction', ...
            'Unknown action "%s". Use "write", "read" or "list".', action);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function path = writeOne(cacheDir, imgName, P, meta, V)
if ~exist(cacheDir, 'dir')
    mkdir(cacheDir);
end

[~, stem] = fileparts(imgName);
path = fullfile(cacheDir, char(stem + ".mat"));

P = single(P);
meta.image = string(imgName);
meta.saved = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');

if isempty(V)
    save(path, 'P', 'meta', '-v7.3');
else
    V = single(V);
    save(path, 'P', 'meta', 'V', '-v7.3');
end
end


function s = readOne(cacheDir, imgName, cfg)
[~, stem] = fileparts(imgName);
path = fullfile(cacheDir, char(stem + ".mat"));

if ~isfile(path)
    error('cnngvf:probCache:missing', ...
        'No cached P for %s in %s. Run run_cv with cfg.Cache.Prob set.', ...
        imgName, cacheDir);
end

s = load(path);                     % P, meta, and V when a TTA cache wrote one
P    = s.P;
meta = s.meta;

assert(isa(P, 'single'), 'cnngvf:probCache:notSingle', ...
    'Cached P for %s is %s, not single. A quantized cache cannot support a tau sweep.', ...
    imgName, class(P));

% --- the leakage guard ---------------------------------------------------
% cnngvf.cvFolds is deterministic on the manifest, so the fold that owned this
% image can be recomputed rather than trusted.
assert(isfield(meta, 'fold') && isscalar(meta.fold) && isfinite(meta.fold), ...
    'cnngvf:probCache:noFold', ...
    ['Cached P for %s carries no fold index, so it cannot be shown to be ' ...
     'out-of-sample. It was probably written by a plain run_batch, whose ' ...
     'model trained on most of the domain. Refusing to score it.'], imgName);

folds = cnngvf.cvFolds(cfg);
assert(meta.fold >= 1 && meta.fold <= numel(folds), ...
    'cnngvf:probCache:badFold', ...
    'Cached P for %s claims fold %d of %d.', imgName, meta.fold, numel(folds));
assert(ismember(string(imgName), folds{meta.fold}), ...
    'cnngvf:probCache:leakage', ...
    ['LEAKAGE: cached P for %s is tagged fold %d, but that fold does not hold ' ...
     'out %s - so the network that produced it trained on this image. Either ' ...
     'the cache is stale or the folds moved. Refusing to score it.'], ...
    imgName, meta.fold, imgName);
end
