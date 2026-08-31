function net = run_train(arm, cfg)
%RUN_TRAIN Train the CNN half of the CNN+GVF pipeline.
%
%  run_train                              % the current arm, DRACTUH_unclustered
%  run_train("unclustered")               % DRAC22 only, the frozen record
%  run_train("clustered")                 % the original baseline
%  run_train(arm, cfg)                    % custom config
%  run_train(cfg)                         % fully custom, no arm applied
%  net = run_train(...)                   % also return the network
%
%  An arm fixes everything that must agree between training and inference:
%  datasets, point set, channel count, epoch budget, model path, output
%  directory. See cnngvf.arm - that is the file to read, and the only file to
%  edit when adding a configuration.
%
%  This used to be two near-identical files, run_train and
%  run_train_unclustered, which drifted apart on Denoise and Vessel.enabled -
%  a mismatch the network cannot survive, since it is tied to the channel count
%  it trained with.
%
%  See also CNNGVF.ARM, CNNGVF.TRAINNET, RUN_BATCH, RUN_CV.

if nargin < 1 || isempty(arm)
    arm = "DRACTUH_unclustered";
end
if nargin < 2
    cfg = cnngvf.defaults();
end
if isstruct(arm)            % run_train(cfg)
    cfg = arm;
else
    cfg = cnngvf.arm(cfg, arm);
end

fprintf('run_train | arm %s | datasets %s | points %s\n', ...
    cfg.Arm, strjoin(cfg.ActiveDatasets, ' + '), cfg.Points);
fprintf('Model out : %s\n', cfg.Paths.Model);

t   = tic;
net = cnngvf.trainNet(cfg);
fprintf('Training finished in %.1f min.\n', toc(t) / 60);
end
