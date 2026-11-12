function Pf = fuseSeedless(Pk, Ps, cfg)
%CNNGVF.FUSESEEDLESS Bring the seedless P into the seeded P before the threshold.
%
%  Pf = cnngvf.fuseSeedless(Pk, Ps, cfg)
%
%  Pk : seeded P;  Ps : seedless P (same size)
%  Pf : max(Pk, Ps where Ps > Post.fuseTauS, 0 elsewhere), over the whole image
%
%  For every image with points; cnngvf.maskFromProb then runs unchanged on Pf,
%  so claiming still drops any seedless blob no seed reaches. It recovers lesion
%  extent the seeded patches stop short of: +0.0104 IoU on a second seedless
%  network, +0.0120 on the adopted one, cells fixed in advance;
%  docs/results/2026-11-12-fusion-confirm.md.
%
%  See also CNNGVF.MASKFROMPROB, CNNGVF.NOPOINTMASK, RUN_CACHE.

Pf = max(Pk, Ps .* (Ps > cfg.Post.fuseTauS));
end
