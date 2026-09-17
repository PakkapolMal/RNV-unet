function Y = d4(X, t, inverse)
%CNNGVF.D4 One element of the dihedral group D4 on the first two dimensions.
%
%  Y = cnngvf.d4(X, t)            % forward
%  X = cnngvf.d4(Y, t, true)      % inverse
%
%  X : p x p x ... (any trailing dims - channels, batch - move together)
%  t : 0..7. t = 0..3 rotate by 90*t degrees counter-clockwise; t = 4..7 do the
%      same rotation and then flip left-right. t = 0 is the identity.
%
%  Every channel is transformed together, so the seed heat-map stays on the
%  transformed seed - a transposed [cx cy] would otherwise be silent
%  (../docs/REFINE.md step 6).
%
%  See also CNNGVF.INFERPROBMAP.

if nargin < 3
    inverse = false;
end
k = mod(t, 4);
if ~inverse
    Y = rot90(X, k);
    if t >= 4
        Y = flip(Y, 2);
    end
else
    if t >= 4
        X = flip(X, 2);
    end
    Y = rot90(X, -k);
end
end
