clc
clear
addpath('evaluation')

%% Load Model
load(fullfile('models','WB_model.mat')); 
model.gamut_mapping = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example1 (RenderedWB_Set1)
disp('Example of evaluating on Set1 from the Rendered WB dataset');
dataset_name = 'RenderedWB_Set1';
imgin = 'Canon1DsMkIII_0087_F_P.png';
in_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set1', 'input');
gt_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set1', 'groundtruth');
metadata_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set1', 'metadata');

fprintf('Reading image: %s\n', imgin);
I_in = im2double(imread(fullfile(in_base, imgin))); % Input image
metadata = get_metadata(imgin, dataset_name, metadata_base); % Metadata
cc_mask = round(metadata.cc_mask);
cc_mask(cc_mask == 0) = cc_mask(cc_mask == 0) + 1;
gt = imread(fullfile(gt_base, metadata.gt_filename)); % Ground truth image

% Hide color chart in input and ground truth
I_in(cc_mask(2):cc_mask(2)+cc_mask(4), cc_mask(1):cc_mask(1)+cc_mask(3), :) = 0;
gt(cc_mask(2):cc_mask(2)+cc_mask(4), cc_mask(1):cc_mask(1)+cc_mask(3), :) = 0;

% Process with original and modified algorithms
fprintf('Processing image: %s\n', imgin);
I_corr_original = model.correctImage(I_in); % Original algorithm
I_corr_original = im2uint8(I_corr_original);

I_corr_modified = model.correctImage(I_in, [], 0.25, true); % Modified algorithm
I_corr_modified = im2uint8(I_corr_modified);

fprintf('Original Algorithm:\n');
[deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig] = ...
    evaluate_cc(I_corr_original, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig);

fprintf('Modified Algorithm:\n');
[deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod] = ...
    evaluate_cc(I_corr_modified, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod);

% Display images
figure('Name', 'Example1: RenderedWB_Set1');
subplot(1, 3, 1); imshow(gt); title('Ground Truth');
subplot(1, 3, 2); imshow(I_corr_original); title('Original Algorithm');
subplot(1, 3, 3); imshow(I_corr_modified); title('Modified Algorithm');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example2 (RenderedWB_Set2)
disp('Example of evaluating on Set2 from the Rendered WB dataset');
dataset_name = 'RenderedWB_Set2';
imgin = 'Mobile_00202.png';
in_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set2', 'input');
gt_base = fullfile('..', 'examples_from_datasets', 'RenderedWB_Set2', 'groundtruth');
metadata_base = ''; % No metadata required

fprintf('Reading image: %s\n', imgin);
I_in = im2double(imread(fullfile(in_base, imgin))); % Input image
metadata = get_metadata(imgin, dataset_name, metadata_base); % Metadata
gt = imread(fullfile(gt_base, metadata.gt_filename)); % Ground truth image

% Process with original and modified algorithms
fprintf('Processing image: %s\n', imgin);
I_corr_original = model.correctImage(I_in); % Original algorithm
I_corr_original = im2uint8(I_corr_original);

I_corr_modified = model.correctImage(I_in, [], 0.25, true); % Modified algorithm
I_corr_modified = im2uint8(I_corr_modified);

%% Evaluation 

fprintf('Original Algorithm:\n');
[deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig] = ...
    evaluate_cc(I_corr_original, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig);

fprintf('Modified Algorithm:\n');
[deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod] = ...
    evaluate_cc(I_corr_modified, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod);

% Display images
figure('Name', 'Example2: RenderedWB_Set2');
subplot(1, 3, 1); imshow(gt); title('Ground Truth');
subplot(1, 3, 2); imshow(I_corr_original); title('Original Algorithm');
subplot(1, 3, 3); imshow(I_corr_modified); title('Modified Algorithm');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example3 (Rendered_Cube+)
disp('Example of evaluating on the Rendered version of Cube+ dataset');
dataset_name = 'Rendered_Cube+';
imgin = '19_F.JPG';
in_base = fullfile('..', 'examples_from_datasets', 'Rendered_Cube+', 'input');
gt_base = fullfile('..', 'examples_from_datasets', 'Rendered_Cube+', 'groundtruth');
metadata_base = ''; % No metadata required

fprintf('Reading image: %s\n', imgin);
I_in = im2double(imread(fullfile(in_base, imgin))); % Input image
metadata = get_metadata(imgin, dataset_name, metadata_base); % Metadata
gt = imread(fullfile(gt_base, metadata.gt_filename)); % Ground truth image

% Process with original and modified algorithms
fprintf('Processing image: %s\n', imgin);
I_corr_original = model.correctImage(I_in); % Original algorithm
I_corr_original = im2uint8(I_corr_original);

I_corr_modified = model.correctImage(I_in, [], 0.25, true); % Modified algorithm
I_corr_modified = im2uint8(I_corr_modified);

%% Evaluation 

fprintf('Original Algorithm:\n');
[deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig] = ...
    evaluate_cc(I_corr_original, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00_orig, MSE_orig, MAE_orig, deltaE76_orig);

fprintf('Modified Algorithm:\n');
[deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod] = ...
    evaluate_cc(I_corr_modified, gt, metadata.cc_mask_area, 4);
fprintf('DeltaE 2000: %0.2f, MSE= %0.2f, MAE= %0.2f, DeltaE 76= %0.2f\n',...
    deltaE00_mod, MSE_mod, MAE_mod, deltaE76_mod);

% Display images
figure('Name', 'Example3: Rendered_Cube+');
subplot(1, 3, 1); imshow(gt); title('Ground Truth');
subplot(1, 3, 2); imshow(I_corr_original); title('Original Algorithm');
subplot(1, 3, 3); imshow(I_corr_modified); title('Modified Algorithm');
