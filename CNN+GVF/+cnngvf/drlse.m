function [BW, phi] = drlse(P, phi0, u, v, cfg)
%CNNGVF.DRLSE Distance-regularized level set driven by a GVF field.
%
%  [BW, phi] = cnngvf.drlse(P, phi0, u, v, cfg)
%
%  P        : the CNN probability map, from which the edge stopper g is built
%  phi0     : initial level set from cnngvf.initPhi (negative inside)
%  u, v     : GVF field from cnngvf.gvfField
%  BW       : phi < 0, the segmentation
%
%  Li, Xu, Gui & Fox, IEEE TIP 19(12):3243-3254, 2010, with the GVF field
%  substituted for grad(g) in the edge term. Expanding the DRLSE edge term
%  gives div(g*grad(phi)/|grad(phi)|) = grad(g).grad(phi)/|grad(phi)| +
%  g*curvature; replacing grad(g) with (u,v) is what extends the capture range,
%  so the initial contour need not already sit on the boundary.
%
%  Note the sign: grad(g) points AWAY from a boundary (g is small there), while
%  the GVF field points TOWARDS it. The substituted term is therefore
%  -(u,v).grad(phi) - the standard advection form, in which a front carried by
%  a velocity field F evolves as phi_t = -F.grad(phi). With the opposite sign
%  the front runs away from every edge it should stop on.
%
%  g comes from P, not from the force field: the force field is already a
%  gradient magnitude, so an edge stopper built from it would be a second
%  derivative and would go small at the inflections of the boundary rather
%  than on the boundary.
%
%  The level set is the reason this pipeline uses no per-lesion bookkeeping:
%  one phi holds every lesion in the image and splits and merges natively,
%  which the ~12 disconnected frond components per ground-truth image need and
%  a parametric snake cannot do.
%
%  The distance regularization is what removes re-initialization and permits
%  cfg.Drlse.dt to be large; it is only stable while muDR*dt < 0.25.
%
%  Runs on the GPU when cfg.UseGpu - the loop is elementwise work on a
%  1024x1024 grid for a few hundred iterations, which is exactly what gpuArray
%  is for. The maths is identical either way.
%
%  See also CNNGVF.INITPHI, CNNGVF.GVFFIELD, CNNGVF.SEGMENTIMAGE.

if cfg.Drlse.muDR * cfg.Drlse.dt >= 0.25
    error('cnngvf:drlse:unstable', ...
        'muDR*dt = %.3f; the distance regularization needs muDR*dt < 0.25.', ...
        cfg.Drlse.muDR * cfg.Drlse.dt);
end

% Edge stopper: small where P has a strong boundary, so the front slows there.
%
% The classic form 1/(1+|grad(G*I)|^2) assumes a 0..255 image, where gradients
% are O(10) and g genuinely approaches 0. P is a probability in [0,1], so the
% same expression yields g ~ 0.999 everywhere - no stopper at all, and the
% balloon term inflates the front through every boundary. Normalising the
% gradient by its own peak and applying edgeGain restores the intended range
% and makes it independent of how sharp this particular CNN's output happened
% to be.
gm = sqrt(gradMagSq(imgaussfilt(double(P), 1)));
gm = gm ./ max(max(gm(:)), eps);
g  = 1 ./ (1 + (cfg.Drlse.edgeGain * gm).^2);

phi = double(phi0);
u   = double(u);
v   = double(v);

if cfg.UseGpu
    [phi, g, u, v] = deal(gpuArray(phi), gpuArray(g), gpuArray(u), gpuArray(v));
end

dt   = cfg.Drlse.dt;
eps_ = cfg.Drlse.eps;

for k = 1:cfg.Drlse.iter
    phi = neumannBC(phi);

    [phi_x, phi_y] = gradient(phi);
    s  = hypot(phi_x, phi_y);
    nx = phi_x ./ (s + 1e-10);
    ny = phi_y ./ (s + 1e-10);

    curv  = divergence2(nx, ny);
    dirac = (eps_ / pi) ./ (eps_^2 + phi.^2);

    phi = phi + dt * ( ...
          cfg.Drlse.muDR   * distReg(phi, phi_x, phi_y, s) ...
        + cfg.Drlse.lambda * dirac .* (g .* curv - (u .* phi_x + v .* phi_y)) ...
        + cfg.Drlse.alfa   * g .* dirac);
end

BW = gather(phi < 0);
if nargout > 1
    phi = gather(phi);
end
end


% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function r = distReg(phi, phi_x, phi_y, s)
% Double-well potential p2 of Li et al. Keeps |grad(phi)| near 1 close to the
% front and near 0 far from it, so phi never needs re-initializing.
a  = (s >= 0) & (s <= 1);
b  = s > 1;
ps = a .* sin(2 * pi * s) / (2 * pi) + b .* (s - 1);

% dp(s) = p'(s)/s, defined as 1 where the numerator and s both vanish.
dps = ((ps ~= 0) .* ps + (ps == 0)) ./ ((s ~= 0) .* s + (s == 0));

r = divergence2(dps .* phi_x - phi_x, dps .* phi_y - phi_y) + 4 * del2(phi);
end


function d = divergence2(nx, ny)
% div(n) = d(nx)/dx + d(ny)/dy. gradient's first output differentiates along
% columns (x), its second along rows (y).
[nxx, ~] = gradient(nx);
[~, nyy] = gradient(ny);
d = nxx + nyy;
end


function m = gradMagSq(A)
[Ax, Ay] = gradient(A);
m = Ax.^2 + Ay.^2;
end


function g = neumannBC(f)
% Zero-flux boundary: mirror the interior into the border ring.
[nrow, ncol] = size(f);
g = f;
g([1 nrow], [1 ncol]) = g([3 nrow-2], [3 ncol-2]);
g([1 nrow], 2:end-1)  = g([3 nrow-2], 2:end-1);
g(2:end-1, [1 ncol])  = g(2:end-1, [3 ncol-2]);
end
