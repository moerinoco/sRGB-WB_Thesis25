clc; clear; close all;

% I love matlab 2024b
addpath('models');   
addpath('classes');  
addpath('evaluation');

%% Load model
load(fullfile('models','WB_model.mat'));  
model.gamut_mapping = 1;

%% CHOOSE IMAGE
[fn,pn] = uigetfile({'*.jpg;*.png','Image Files'},'Select Input Image');
if isequal(fn,0), disp('No image selected, crying time.'); return; end
I_in = im2double(imread(fullfile(pn,fn)));

%% Run Original Algorithm
RUN_ORIG = true;

if RUN_ORIG
    t0 = tic;
    I_orig     = model.correctImage(I_in);
    time_orig  = toc(t0);
    fprintf('Original:  %.3f s\n', time_orig);
else
    I_orig = []; time_orig = NaN;
end

%% Run Modified Algorithm
USE_DYNAMIC_K = false;
USE_FALLBACK  = true;
USE_FASTHIST  = true;
SIGMA         = 0.25;

t1 = tic;
[I_mod,~,~,dynamic_k] = model.correctImage(I_in,[],SIGMA,...
                    USE_DYNAMIC_K,USE_FALLBACK,USE_FASTHIST);
time_mod = toc(t1);
fprintf('Modified:  %.3f s  (dynamic K = %d)\n', time_mod, dynamic_k);

%% Compute runtime % diff. 
if RUN_ORIG
    pct_diff_time = 100*(time_mod - time_orig)/time_orig;  % negative = faster
    fprintf('Time Δ%% (Mod vs Orig): %.1f %%\n', pct_diff_time);
else
    pct_diff_time = NaN;
end

%% Compute ΔE2000 between outputs
if RUN_ORIG
    dE_map = calc_deltaE2000(im2uint8(I_orig), im2uint8(I_mod), 0);
    mean_dE = mean(dE_map(:),'omitnan');
    fprintf('Average ΔE2000 (Orig vs Mod): %.3f\n', mean_dE);
end

%% Display outputs
figure('Name','Demo','NumberTitle','off','Color','w');
ncols = 2 + RUN_ORIG;
subplot(1,ncols,1), imshow(I_in), title('Input');

if RUN_ORIG
    subplot(1,ncols,2), imshow(I_orig), ...
        title(sprintf('Original Method\n(%.2f s)',time_orig));
    col = 3;
else
    col = 2;
end

subplot(1,ncols,col), imshow(I_mod), ...
    title(sprintf('Modified Method\n(%.2f s, %.1f%%)', ...
          time_mod, pct_diff_time));
drawnow;
