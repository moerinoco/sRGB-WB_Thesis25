clc
clear

addpath('evaluation')
addpath('models'); 
addpath('classes');

% Profiler to eval runtime

profile clear;
profile('-detail','builtin');
profile on;

%% Test Parameters
RUN_ORIGINAL = true;     % Also run the original algorithm 
DISP = false;            % Whether to display images (overridden if in 'LIST' mode)
PERF = true;             % Print performance for each image

TEST_MODE = 'LIST';      % Options: 'FULL', 'SKIP', 'LIST'
                         % 'FULL'  Process all 10,212 images in Cube+
                         % 'SKIP'  Process every SKIP_STEP images in Cube++
                         % 'LIST'  Process all photos specified in list

SKIP_STEP = 2;           % Use every n images in Cube++ 

USE_DYNAMIC_K = false;    % Dynamically choose K # of neighbors 

USE_FALLBACK = true;     % Use fallback routine to target failure cases
                         % -- Shades of Gray + slight desaturation

USE_FASTHIST = true;

% For 'LIST' mode
SINGLE_IMG_NAMES = { ...
    '1549_D.JPG', '13_F.JPG', '1554_S.JPG', '535_D.JPG', '1460_S.JPG', ...
    '20_S.JPG','685_S.JPG','1640_D.JPG','685_D.JPG','225_S.JPG', ...
    '1640_S.JPG','545_S.JPG','1055_S.JPG','195_S.JPG','535_S.JPG', ...
    '1010_S.JPG', '225_D.JPG', '1475_S.JPG', '1620_S.JPG', ...
    '1055_D.JPG', '980_S.JPG'};

%% Load Models 
load(fullfile('models','WB_model.mat'));
model.gamut_mapping = 1;

% Automatically display results in 'LIST' mode
if strcmp(TEST_MODE, 'LIST')
    DISP = 1;
end

%% Set up results logging table
results_table = table('Size', [0 13], ...
    'VariableTypes', {'string','double','double','double','double','double',...
                      'double','double','double','double','double','double', 'logical'}, ...
    'VariableNames', {'ImageName','DeltaE00_Orig','MSE_Orig','MAE_Orig','DeltaE76_Orig',...
                      'DeltaE00_Mod','MSE_Mod','MAE_Mod','DeltaE76_Mod','DynamicK',...
                      'Time_Orig','Time_Mod', 'FallbackUsed'});

%% Dataset Folders 
dataset_name   = 'Rendered_Cube+';
in_base        = fullfile('..', 'Cube_input_images');
gt_base        = fullfile('..', 'Cube_ground_truth_images');
metadata_base  = ''; 


%% Get all JPGs in folder
image_files = dir(fullfile(in_base, '*.JPG'));
filenames   = {image_files.name}; 
all_indices = 1:length(image_files);

%% Decide Which Images to Process
switch TEST_MODE
    case 'FULL'
        indices_to_test = all_indices;

    case 'SKIP'

        indices_to_test = 1:SKIP_STEP:length(image_files);

    case 'LIST'
        indices_to_test = [];
        for i = 1:numel(SINGLE_IMG_NAMES)
            thisName = SINGLE_IMG_NAMES{i};
            idxFound = find(strcmp(filenames, thisName), 1);
            if isempty(idxFound)
                error('Could not find "%s" in folder: %s', thisName, in_base);
            end
            indices_to_test(end+1) = idxFound; 
        end

    otherwise
        % DEADBEEF
        error('TEST_MODE must be one of: FULL, SKIP, LIST');
end

n_images = numel(indices_to_test);

%% Pre-allocate arrays for metrics
if RUN_ORIGINAL
    time_orig_all       = zeros(n_images,1);
    deltaE00_orig_all   = zeros(n_images,1);
    MSE_orig_all        = zeros(n_images,1);
    MAE_orig_all        = zeros(n_images,1);
    deltaE76_orig_all   = zeros(n_images,1);
end

time_mod_all      = zeros(n_images,1);
deltaE00_mod_all  = zeros(n_images,1);
MSE_mod_all       = zeros(n_images,1);
MAE_mod_all       = zeros(n_images,1);
deltaE76_mod_all  = zeros(n_images,1);
fallback_used_all   = false(n_images,1);

% Start the overall runtime timer
tic; 

%% -------------------------------------------------------------------------
% Process images
% --------------------------------------------------------------------------
for idx = 1 : n_images
    k = indices_to_test(idx);
    imgin = filenames{k};
    fprintf('\n\nProcessing IMG: %s (index %d of %d)\n', imgin, idx, n_images);
    
    %% Read input + ground truth file
    I_in = im2double(imread(fullfile(in_base, imgin)));
    metadata = get_metadata(imgin, dataset_name, metadata_base);  
    gt = imread(fullfile(gt_base, metadata.gt_filename));
    
    %% Run Original Algorithm
    if RUN_ORIGINAL
        t0_orig = tic;
        I_corr_original = model.correctImage(I_in);
        time_orig = toc(t0_orig);

        I_corr_original = im2uint8(I_corr_original); % Process 

        [deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig] = ...
        evaluate_cc(I_corr_original, gt, metadata.cc_mask_area, 4);

        if PERF
            fprintf('Original Algorithm:\n');
            fprintf('DeltaE2000=%.2f, MSE=%.2f, MAE=%.2f, DeltaE76=%.2f, Time=%.4fs\n',...
                deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig, time_orig);
        end
        
        deltaE00_orig_all(idx) = deltaE00_orig;
        MSE_orig_all(idx)      = MSE_orig;
        MAE_orig_all(idx)      = MAE_orig;
        deltaE76_orig_all(idx) = deltaE76_orig;
        time_orig_all(idx)     = time_orig;
    else
        % For when not running original algorithm
        deltaE00_orig = NaN;
        MSE_orig      = NaN;
        MAE_orig      = NaN;
        deltaE76_orig = NaN;
        time_orig     = NaN;
    end

    %% Run Modified Algorithm
    t0_mod = tic;
    [I_corr_modified, ~, ~, dynamic_k, fallback_flag] = model.correctImage(I_in, [], 0.25, USE_DYNAMIC_K, USE_FALLBACK, USE_FASTHIST);
    time_mod = toc(t0_mod);

    I_corr_modified = im2uint8(I_corr_modified); % Process

    [deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod] = ...
    evaluate_cc(I_corr_modified, gt, metadata.cc_mask_area, 4);

    if PERF
        fprintf('Modified Algorithm:\n');
        fprintf('DeltaE2000=%.2f, MSE=%.2f, MAE=%.2f, DeltaE76=%.2f, Time=%.4fs\n',...
            deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod, time_mod);
        fprintf('K: %d\n\n', dynamic_k);
    end

    deltaE00_mod_all(idx) = deltaE00_mod;
    MSE_mod_all(idx)       = MSE_mod;
    MAE_mod_all(idx)       = MAE_mod;
    deltaE76_mod_all(idx)  = deltaE76_mod;
    time_mod_all(idx)      = time_mod;
    fallback_used_all(idx) = fallback_flag;

    %% Append to results table
    results_table = [results_table; ...
        {imgin, ...
         deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig, ...
         deltaE00_mod,  MSE_mod,  MAE_mod,  deltaE76_mod, ...
         dynamic_k, time_orig, time_mod, fallback_flag}];
    
    %% Display images if 'DISP' 
    if DISP

        % Compute hists for both
        %hist_orig = model.RGB_UVhist(I_in);
        %hist_fast = model.fast_RGB_UVhist(I_in);
        
        % Visualize hists
        %visualizeHist3D(hist_orig, 'Original: ');
        %visualizeHist3D(hist_fast, 'Fast: ');

        figure(1); clf;
        subplot(2, 1 + RUN_ORIGINAL, 1); imshow(I_in); 
        title("Input");
        subplot(2, 1 + RUN_ORIGINAL, 2); imshow(gt);   
        title("Ground Truth");
        subplot(2, 1 + RUN_ORIGINAL, 3); imshow(I_corr_modified); 
        title(sprintf('Modified (DeltaE2000=%.2f)', deltaE00_mod));
        if RUN_ORIGINAL
            subplot(2, 2, 4); imshow(I_corr_original); 
            title(sprintf('Original (DeltaE2000=%.2f)', deltaE00_orig));
        end
        drawnow;
        
        % Pause until input
        fprintf('Press ENTER to proceed to the next img (or Ctrl+C to abort)\n');
        pause;
        close(gcf);
    end
end

%% Compute average performance
if RUN_ORIGINAL
    avg_deltaE00_orig = mean(deltaE00_orig_all, 'omitnan');
    avg_MSE_orig      = mean(MSE_orig_all,      'omitnan');
    avg_MAE_orig      = mean(MAE_orig_all,      'omitnan');
    avg_deltaE76_orig = mean(deltaE76_orig_all, 'omitnan');
    avg_time_orig     = mean(time_orig_all,     'omitnan');
end

avg_deltaE00_mod  = mean(deltaE00_mod_all, 'omitnan');
avg_MSE_mod       = mean(MSE_mod_all,      'omitnan');
avg_MAE_mod       = mean(MAE_mod_all,      'omitnan');
avg_deltaE76_mod  = mean(deltaE76_mod_all, 'omitnan');
avg_time_mod      = mean(time_mod_all,     'omitnan');

% Add AVERAGE metrics to bottom of table
if RUN_ORIGINAL
    avg_results_table = table({'AVERAGES'}, avg_deltaE00_orig, avg_MSE_orig, avg_MAE_orig, avg_deltaE76_orig, ...
                                         avg_deltaE00_mod,  avg_MSE_mod,  avg_MAE_mod,  avg_deltaE76_mod, ...
                                         NaN, avg_time_orig, avg_time_mod, false, ...
                              'VariableNames', results_table.Properties.VariableNames);
else
    avg_results_table = table({'AVERAGES'}, NaN, NaN, NaN, NaN, ...
                                         avg_deltaE00_mod, avg_MSE_mod, avg_MAE_mod, avg_deltaE76_mod, ...
                                         NaN, NaN, avg_time_mod, false, ...
                              'VariableNames', results_table.Properties.VariableNames);
end

results_table = [results_table; avg_results_table];

%% Write metrics to CSV file unless in 'LIST' mode
if ~strcmp(TEST_MODE, 'LIST') && ~strcmp(TEST_MODE, 'SINGLE')
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    output_filename = sprintf('Performance_Results_%s.csv', timestamp);

    folder_name = 'Performance';
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    
    output_file = fullfile(folder_name, output_filename);
    writetable(results_table, output_file);
    fprintf('Results saved to %s\n', output_file);
end

%% Compute total program runtime
elapsed_time = toc;
elapsed_time = elapsed_time / 3600.00; % Convert to hours
fprintf('\nThe entire program took %.2f hours to run.\n', elapsed_time);

profile off;
profile viewer;