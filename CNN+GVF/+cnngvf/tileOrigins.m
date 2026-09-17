function o = tileOrigins(n, cfg)
%CNNGVF.TILEORIGINS First index of each seedless tile along one image axis.
%
%  o = cnngvf.tileOrigins(n, cfg)
%
%  Tiles of cfg.Patch.size every cfg.Patch.tileStride, with the last one pinned
%  to the far edge so the axis is always covered. 1024 px, 512 / 256 -> [1 257 513].
%  Shared by the seedless validation set (cnngvf.prepareData) and inference
%  (cnngvf.inferProbMap), so checkpoint selection sees the inference geometry.
%
%  See also CNNGVF.INFERPROBMAP, CNNGVF.PREPAREDATA.

p = cfg.Patch.size;
assert(n >= p, 'cnngvf:tileOrigins:small', 'Axis of %d px is shorter than the %d px tile.', n, p);
o = unique([1:cfg.Patch.tileStride:(n - p + 1), n - p + 1]);
end
