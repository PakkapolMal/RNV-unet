function [u, v] = gvfField(f, cfg)
%CNNGVF.GVFFIELD Gradient Vector Flow field of an energy map (Xu & Prince).
%
%  [u, v] = cnngvf.gvfField(f, cfg)
%
%  f    : h x w energy field in [0,1] from cnngvf.buildForce
%  u, v : the diffused field, scaled so max(hypot(u,v)) == 1
%
%  Wraps the compiled GVF in lib/Fast-Gradient-Vector-Flow (Quan Wang's
%  C++/MEX implementation, already in this repo) - reusing it rather than
%  re-deriving the diffusion. Falls back to an explicit-Euler MATLAB loop if
%  the mex will not load, so the pipeline still runs on a machine where the
%  binary is incompatible.
%
%  The field is scaled globally, not per-vector: the relative weakness of the
%  field far from the lesion is meaningful to the level set, and per-vector
%  normalisation would erase it. The global scale makes cfg.Drlse.lambda
%  independent of how confident the CNN happened to be.
%
%  See also CNNGVF.BUILDFORCE, CNNGVF.DRLSE.

f = double(f);

try
    [u, v] = GVF(f, cfg.Gvf.alpha, cfg.Gvf.mu, cfg.Gvf.iter);
catch err
    % ponytail: the mex is the fast path; run `mex GVF.cpp` in
    % lib/Fast-Gradient-Vector-Flow to restore it if you land here.
    warning('cnngvf:gvfField:noMex', ...
        'GVF mex unavailable (%s); using the MATLAB fallback.', err.message);
    [u, v] = gvfMatlab(f, cfg.Gvf.alpha, cfg.Gvf.mu, cfg.Gvf.iter);
end

peak = max(hypot(u(:), v(:)));
if peak > 0
    u = u ./ peak;
    v = v ./ peak;
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function [u, v] = gvfMatlab(f, alpha, mu, iter)
% Xu & Prince's Euler-Lagrange iteration, vectorised:
%   u <- u + alpha*(mu*lap(u) - |grad f|^2 (u - fx))
[fx, fy] = gradient(f);
b = fx.^2 + fy.^2;
u = fx;
v = fy;
for k = 1:iter
    u = u + alpha * (mu * 4 * del2(u) - b .* (u - fx));
    v = v + alpha * (mu * 4 * del2(v) - b .* (v - fy));
end
end
