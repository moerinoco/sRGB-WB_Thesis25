function visualizeHist3D(hist, titlePrefix)
% Display RGB-UV histogram as a heatmap
    if nargin < 2
        titlePrefix = '';
    end

    channelNames = {'Red channel', 'Green channel', 'Blue channel'};
    figure;
    for c = 1:3
        subplot(1,3,c);
        imagesc(hist(:,:,c));
        axis image;
        colormap hot;
        colorbar;
        title(sprintf('%s%s', titlePrefix, channelNames{c}));
    end
end
