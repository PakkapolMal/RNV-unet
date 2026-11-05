function [comp, L, recovered] = seedLabels(gt, seeds, tol)
%CNNGVF.SEEDLABELS The ground-truth component each training seed is labelled with.
%
%  [comp, L, recovered] = cnngvf.seedLabels(gt, seeds, tol)
%
%  gt    : h x w logical FilledGT mask
%  seeds : S x 2 [cx cy]
%  tol   : cfg.Patch.seedTolerance, px
%
%  comp      : S x 1, the label in L of the seed's lesion; 0 = dropped
%  L         : labelmatrix(bwconncomp(gt))
%  recovered : how many seeds outside every lesion were labelled within tol
%
%  A seed inside a lesion takes that lesion. A seed outside every lesion takes
%  the nearest one when within tol of its BOUNDARY (not its centroid: a seed
%  just outside a large concave lesion can be nearer another blob's centroid),
%  and is dropped otherwise, because training it against pure background would
%  teach the network to ignore its own heat-map. Uses ground truth, so training
%  only (docs/SPEC.md, "Seed tolerance").
%
%  Shared by cnngvf.prepareData and run_export so the PyTorch trainer is
%  labelled by the same rule.
%
%  See also CNNGVF.PREPAREDATA, RUN_EXPORT.

L = labelmatrix(bwconncomp(gt));
[h, w] = size(gt);
[dist, nearest] = bwdist(gt);

row = min(max(round(seeds(:, 2)), 1), h);
col = min(max(round(seeds(:, 1)), 1), w);
at  = sub2ind([h w], row, col);

comp  = double(L(at));
near  = comp == 0 & dist(at) <= tol;
comp(near) = double(L(nearest(at(near))));
recovered  = nnz(near);
end
