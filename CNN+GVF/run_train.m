function net = run_train(cfg)
%RUN_TRAIN Train the CNN half of the CNN+GVF pipeline.
%
%  run_train                % defaults
%  run_train(cfg)           % custom config
%  net = run_train(...)     % also return the network
%
%  Trains the seed-conditioned U-Net on the 17 ground-truth images and writes
%  it to cfg.Paths.Model. run_batch then loads it from there.
%
%  See also CNNGVF.TRAINNET, CNNGVF.DEFAULTS, RUN_BATCH.

if nargin < 1
    cfg = cnngvf.defaults();
end

t   = tic;
net = cnngvf.trainNet(cfg);
fprintf('Training finished in %.1f min.\n', toc(t) / 60);
end
