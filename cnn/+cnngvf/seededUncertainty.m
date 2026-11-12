function U = seededUncertainty(Pk)
%CNNGVF.SEEDEDUNCERTAINTY How unsure the seeded P is about its outlines, without GT.
%
%  U = cnngvf.seededUncertainty(Pk)
%
%  U = (pixels with 0.2 < Pk < 0.8) / (pixels with Pk >= 0.5), both counted
%  where the seeded network gave an answer (Pk > 0.01). Low: a decisive map.
%  High: a wide band of unsure pixels around the lesions. 0 when Pk has no
%  pixel at 0.5 or above. Experiment: docs/briefs/fusion-uncertainty.md.

foot = Pk > 0.01;
U = nnz(foot & Pk > 0.2 & Pk < 0.8) / max(1, nnz(foot & Pk >= 0.5));
end
