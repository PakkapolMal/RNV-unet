function saveOverlay(outFile, img, seeds, contours)
%CNNGVF.SAVEOVERLAY Write a PNG of the image with seeds and final contours.
%
%  cnngvf.saveOverlay(outFile, img, seeds, contours)
%
%  img      : struct from cnngvf.loadChannels
%  seeds    : S x 2 [cx cy]
%  contours : cell of N x 2 [row col] contours; empty entries skipped
%
%  See also CNNGVF.SEGMENTIMAGE.

fig = figure('Visible', 'off');
cleanup = onCleanup(@() close(fig));

imshow(img.I, []);
hold on;

plot(seeds(:, 1), seeds(:, 2), 'o', 'MarkerSize', 5, ...
    'MarkerFaceColor', [1 0.4 0.7], 'MarkerEdgeColor', [1 0.4 0.7]);

for s = 1:numel(contours)
    O = contours{s};
    if isempty(O)
        continue;
    end
    plot([O(:, 2); O(1, 2)], [O(:, 1); O(1, 1)], 'r', 'LineWidth', 2);
end

exportgraphics(gca, outFile, 'Resolution', 300);
end
