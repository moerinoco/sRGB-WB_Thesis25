%% train_fallback_svm.m
clc; clear;

%% locate the most recent TrainingData CSV
trainFolder = 'TrainingData';
d = dir(fullfile(trainFolder,'TrainingData_*.csv'));
[~,idx] = max([d.datenum]);
csvFile = fullfile(trainFolder, d(idx).name);
fprintf("Loading training data from %s\n", d(idx).name);
T = readtable(csvFile);

%% build X and Y
featCols   = startsWith(T.Properties.VariableNames,'F');
extraCols  = ismember(T.Properties.VariableNames,{'MeanR','MeanG','MeanB','RGBratio'});
X = T{:, featCols | extraCols};    % predictor matrix
Y = T.BenefitsFallback;            % logical true/false

% Define parameter ranges
boxVals = logspace(-2, 2, 5);       % e.g., [0.01 0.1 1 10 100]
scaleVals = logspace(-2, 2, 5);     % e.g., same

bestAccuracy = 0;

for b = 1:length(boxVals)
    for s = 1:length(scaleVals)
        % Train model with current parameters
        t = templateSVM('KernelFunction','rbf', ...
                        'BoxConstraint',boxVals(b), ...
                        'KernelScale',scaleVals(s));
        CVSVM = fitcsvm(X, Y, ...
                        'KernelFunction','rbf', ...
                        'BoxConstraint',boxVals(b), ...
                        'KernelScale',scaleVals(s), ...
                        'CrossVal','on', ...
                        'KFold',5);  % 5-fold cross-validation
        
        % Evaluate cross-validation loss (lower is better)
        acc = 1 - kfoldLoss(CVSVM);
        fprintf('Box=%.3f, Scale=%.3f => Accuracy=%.2f%%\n', ...
                boxVals(b), scaleVals(s), acc*100);
        
        % Save best model
        if acc > bestAccuracy
            bestAccuracy = acc;
            bestModel = CVSVM;
            bestParams = [boxVals(b), scaleVals(s)];
        end
    end
end

fprintf('\nBest Parameters: Box=%.3f, Scale=%.3f (Accuracy=%.2f%%)\n', ...
        bestParams(1), bestParams(2), bestAccuracy*100);
