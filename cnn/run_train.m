function net = run_train(cfg)
%RUN_TRAIN Train the seed-conditioned U-Net that produces P.
%
%  run_train                              % the current configuration
%  run_train("unclustered")               % a frozen arm (see cnngvf.arm)
%  run_train(cfg)                         % custom config
%  net = run_train(...)                   % also return the network
%
%  Writes cfg.Paths.Model. The configuration fixes everything that must agree
%  between training and inference - datasets, point set, channel count, model
%  path - so train and segment with the same one.
%
%  See also CNNGVF.ARM, CNNGVF.TRAINNET, RUN_BATCH, RUN_CV.

if nargin < 1
    cfg = cnngvf.defaults();
elseif isstring(cfg) || ischar(cfg)
    cfg = cnngvf.arm(cnngvf.defaults(), cfg);
end

fprintf('run_train | arm %s | datasets %s | points %s\n', ...
    cfg.Arm, strjoin(cfg.ActiveDatasets, ' + '), cfg.Points);
fprintf('Model out : %s\n', cfg.Paths.Model);

t   = tic;
net = cnngvf.trainNet(cfg);
fprintf('Training finished in %.1f min.\n', toc(t) / 60);
end
