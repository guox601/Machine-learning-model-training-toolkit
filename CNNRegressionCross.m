function varargout = CNNRegressionCross(varargin)
% CNNREGRESSIONCROSS MATLAB code for CNNRegressionCross.fig
%      CNNREGRESSIONCROSS, by itself, creates a new CNNREGRESSIONCROSS or raises the existing
%      singleton*.
% 
%      H = CNNREGRESSIONCROSS returns the handle to a new CNNREGRESSIONCROSS or the handle to
%      the existing singleton*.
% 
%      CNNREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CNNREGRESSIONCROSS.M with the given input arguments.
% 
%      CNNREGRESSIONCROSS('Property','Value',...) creates a new CNNREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CNNRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CNNRegressionCross_OpeningFcn via varargin.
% 
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
% 
% See also: GUIDE, GUIDATA, GUIHANDLES

% Last Modified by GUIDE v2.5 06-Jun-2025 14:04:19

% --- Initialization code ---
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CNNRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @CNNRegressionCross_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% --- Executes just before CNNRegressionCross is made visible.
function CNNRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = CNNRegressionCross_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

function edit1_Callback(hObject, eventdata, handles)
function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit2_Callback(hObject, eventdata, handles)
function edit2_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit3_Callback(hObject, eventdata, handles)
function edit3_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    filename = get(handles.edit1, 'string');
    k = str2double(get(handles.edit3, 'string'));
    outdim = str2double(get(handles.edit2, 'string'));

    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);

    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);

    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1;
    Xnorm = (X - mu) ./ sigma;
    numFeatures = f_;
    X_cnn = reshape(Xnorm', [numFeatures, 1, 1, size(Xnorm,1)]);

    objFcn = @(params) kFoldEvalCNN(params, X_cnn, Y, outdim, k);

    optVars = [
        optimizableVariable('NumFilters', [8, 64], 'Type', 'integer')
        optimizableVariable('FilterSize', [2, 5], 'Type', 'integer')
        optimizableVariable('InitialLearnRate', [1e-4, 1e-2], 'Transform', 'log')
        optimizableVariable('MaxEpochs', [50, 300], 'Type', 'integer')
        optimizableVariable('MiniBatchSize', [8, 64], 'Type', 'integer')
    ];

    results = bayesopt(objFcn, optVars, ...
        'MaxObjectiveEvaluations', 20, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1, ...
        'UseParallel', false);

    bestParams = results.XAtMinObjective;

    fprintf('\n=== Final Training using %d-Fold Cross-Validation ===\n', k);
    cv = cvpartition(size(Y,1), 'KFold', k);
    allMetrics = repmat(struct('R2', [], 'RMSE', [], 'MAPE', []), k, 1);

    for i = 1:k
        fprintf('\n-- Fold %d/%d --\n', i, k);
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);

        Xtrain = X_cnn(:,:,:,trainIdx);
        Ytrain = Y(trainIdx, :);
        Xtest  = X_cnn(:,:,:,testIdx);
        Ytest  = Y(testIdx, :);

        layers = [
            imageInputLayer([numFeatures 1 1], 'Normalization', 'none')
            convolution2dLayer([bestParams.FilterSize 1], bestParams.NumFilters, 'Padding', 'same')
            reluLayer
            fullyConnectedLayer(outdim)
            regressionLayer
        ];

        options = trainingOptions('adam', ...
            'MaxEpochs', bestParams.MaxEpochs, ...
            'InitialLearnRate', bestParams.InitialLearnRate, ...
            'MiniBatchSize', bestParams.MiniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', 0, ...
            'Plots', 'none');

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Ypred = predict(net, Xtest);

        metrics = calc_metrics(Ytest, Ypred);
        allMetrics(i) = metrics;

        for d = 1:outdim
            fprintf('Dim %d: R² = %.4f | RMSE = %.4f | MAPE = %.2f%%\n', ...
                d, metrics.R2(d), metrics.RMSE(d), metrics.MAPE(d));
        end
    end

    fprintf('\n=== Average Performance Across %d Folds ===\n', k);
    for d = 1:outdim
        allR2 = arrayfun(@(x) x.R2(d), allMetrics);
        allRMSE = arrayfun(@(x) x.RMSE(d), allMetrics);
        allMAPE = arrayfun(@(x) x.MAPE(d), allMetrics);
        fprintf('Dim %d: R² = %.4f | RMSE = %.4f | MAPE = %.2f%%\n', ...
            d, mean(allR2), mean(allRMSE), mean(allMAPE));
    end

    assignin('base', 'TrainedCNNModel', net);
    assignin('base', 'Yall_true', Y);
    assignin('base', 'Yall_pred', predict(net, X_cnn));
    
function objective = kFoldEvalCNN(params, X_cnn, Y, outdim, k)
    cv = cvpartition(size(Y,1), 'KFold', k);
    rmse_all = zeros(k, outdim);

    for i = 1:k
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);

        Xtrain = X_cnn(:,:,:,trainIdx);
        Ytrain = Y(trainIdx,:);
        Xtest  = X_cnn(:,:,:,testIdx);
        Ytest  = Y(testIdx,:);

        layers = [
            imageInputLayer([size(X_cnn,1) 1 1], 'Normalization', 'none')
            convolution2dLayer([params.FilterSize 1], params.NumFilters, 'Padding', 'same')
            reluLayer
            fullyConnectedLayer(outdim)
            regressionLayer
        ];

        options = trainingOptions('adam', ...
            'MaxEpochs', params.MaxEpochs, ...
            'InitialLearnRate', params.InitialLearnRate, ...
            'MiniBatchSize', params.MiniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Ypred = predict(net, Xtest);

        for d = 1:outdim
            rmse_all(i,d) = sqrt(mean((Ypred(:,d) - Ytest(:,d)).^2));
        end
    end

    objective = mean(rmse_all, 'all');
function metrics = calc_metrics(Ytrue, Ypred)
    % 支持多维输出，逐维计算 R²、RMSE、MAPE
    outdim = size(Ytrue, 2);
    R2 = zeros(1, outdim);
    RMSE = zeros(1, outdim);
    MAPE = zeros(1, outdim);

    for i = 1:outdim
        yt = Ytrue(:, i);
        yp = Ypred(:, i);

        ss_res = sum((yt - yp).^2);
        ss_tot = sum((yt - mean(yt)).^2);
        R2(i) = 1 - ss_res / ss_tot;

        RMSE(i) = sqrt(mean((yt - yp).^2));

        if any(yt == 0)
            MAPE(i) = NaN; % 避免除零错误
        else
            MAPE(i) = mean(abs((yt - yp) ./ yt)) * 100;
        end
    end

    metrics.R2 = R2;
    metrics.RMSE = RMSE;
    metrics.MAPE = MAPE;
