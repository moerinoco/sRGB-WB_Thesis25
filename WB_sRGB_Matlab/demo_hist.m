%% demo_hist_subsampling.m – visualise full vs sub‑sampled RGB‑UV hist
clc; clear; close all;

addpath models classes
load(fullfile('models','WB_model.mat'));           % loads “model”
model.gamut_mapping = 1;                           % any value OK for hist

%% ------------------------------------------------------------------ %
%  1) Pick image
% ------------------------------------------------------------------- %
[fn,pn] = uigetfile({'*.jpg;*.png','Image files'},'Choose an image');
if isequal(fn,0), disp('No file chosen.'); return; end
I = im2double(imread(fullfile(pn,fn)));

%% ------------------------------------------------------------------ %
%  2) Cap at 450×450 (≈202 500 px)
% ------------------------------------------------------------------- %
maxArea = 450*450;
if size(I,1)*size(I,2) > maxArea
    sf = sqrt(maxArea/(size(I,1)*size(I,2)));
    I  = imresize(I,sf,'nearest');
end

%% ------------------------------------------------------------------ %
%  3) User‑chosen subsample fraction
% ------------------------------------------------------------------- %
subFrac = input('Subsample fraction (e.g. 0.05) : ');
assert(~isempty(subFrac) && subFrac>0 && subFrac<=1, ...
       'Value must be in (0,1].');

%% ------------------------------------------------------------------ %
%  4) Full‑pixel histogram  + timing
% ------------------------------------------------------------------- %
tFull = tic;
hist_full = model.RGB_UVhist(I);          % h×h×3 (h≈64)
time_full = toc(tFull);
fprintf('Full‑pixel histogram:  %.4f s\n', time_full);

%% ------------------------------------------------------------------ %
%  5) Sub‑sampled histogram  + timing
% ------------------------------------------------------------------- %
I_lin = reshape(I,[],3);
I_lin = I_lin(all(I_lin>0,2),:);          % drop zero pixels
keep   = rand(size(I_lin,1),1) < subFrac; % Bernoulli mask
I_sub  = reshape(I_lin(keep,:),[],3);

tSub = tic;
hist_sub = model.RGB_UVhist(I_sub);
time_sub = toc(tSub);
fprintf('Sub‑sampled histogram (%.0f %%):  %.4f s\n', ...
        subFrac*100, time_sub);

%% ------------------------------------------------------------------ %
%  6) Visualise
% ------------------------------------------------------------------- %
titles = {'R‑channel','G‑channel','B‑channel'};
figure('Name','RGB‑UV histograms','Color','w');
for c = 1:3
    subplot(2,3,c);
    imagesc(hist_full(:,:,c)); axis square off;
    title(sprintf('FULL  – %s\n%.3f s', titles{c}, time_full));

    subplot(2,3,c+3);
    imagesc(hist_sub(:,:,c)); axis square off;
    title(sprintf('SUB %.0f%% – %s\n%.3f s', ...
          subFrac*100, titles{c}, time_sub));
end
colormap turbo;
