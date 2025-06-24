function varargout = KNNRegressionCross(varargin)
% KNNREGRESSIONCROSS MATLAB code for KNNRegressionCross.fig
%      KNNREGRESSIONCROSS, by itself, creates a new KNNREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = KNNREGRESSIONCROSS returns the handle to a new KNNREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      KNNREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in KNNREGRESSIONCROSS.M with the given input arguments.
%
%      KNNREGRESSIONCROSS('Property','Value',...) creates a new KNNREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before KNNRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to KNNRegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Last Modified by GUIDE v2.5 05-Jun-2025 10:51:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @KNNRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @KNNRegressionCross_OutputFcn, ...
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
% End initialization code - DO NOT EDIT


% --- Executes just before KNNRegressionCross is made visible.
function KNNRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for KNNRegressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes KNNRegressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = KNNRegressionCross_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% No additional code needed here


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    % Read parameters
    filename = get(handles.edit1, 'string');          % Data file path
    kfold = str2double(get(handles.edit2, 'string')); % Number of folds for cross-validation
    outdim = str2double(get(handles.edit3, 'string')); % Number of output dimensions

    % Load data and shuffle
    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);

    % Split features and targets
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);

    % Normalize features
    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1; % avoid division by zero
    Xnorm = (X - mu) ./ sigma;

    % k-fold cross-validation partition
    cv = cvpartition(size(Y, 1), 'KFold', kfold);

    % Define hyperparameter search space for Bayesian optimization
    vars = optimizableVariable('k', [1, 20], 'Type', 'integer');

    % Objective function for Bayesian optimization
    objFcn = @(params) knnObjective(params.k, Xnorm, Y, cv, outdim);

    % Run Bayesian optimization, up to 30 iterations (adjust as needed)
    results = bayesopt(objFcn, vars, ...
        'MaxObjectiveEvaluations', 30, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 0);

    best_k = results.XAtMinObjective.k;
    fprintf('Bayesian optimization found best k = %d\n', best_k);

    % Perform final k-fold cross-validation evaluation using best k
    R2_all = zeros(kfold, outdim);
    MSE_all = zeros(kfold, outdim);
    RMSE_all = zeros(kfold, outdim);
    MAPE_all = zeros(kfold, outdim);

    Ytest_all = [];
    Ypred_all = [];

    for fold = 1:kfold
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        Xtrain = Xnorm(trainIdx, :);
        Ytrain = Y(trainIdx, :);
        Xtest = Xnorm(testIdx, :);
        Ytest = Y(testIdx, :);

        Ypred_fold = zeros(sum(testIdx), outdim);

        for dim = 1:outdim
            y_train = Ytrain(:, dim);
            y_test = Ytest(:, dim);
            y_pred = knn_regression_predict(Xtrain, y_train, Xtest, best_k);
            Ypred_fold(:, dim) = y_pred;

            R2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);
            MSE = mean((y_test - y_pred).^2);
            RMSE = sqrt(MSE);
            MAPE = mean(abs((y_test - y_pred) ./ (y_test + eps))) * 100;

            R2_all(fold, dim) = R2;
            MSE_all(fold, dim) = MSE;
            RMSE_all(fold, dim) = RMSE;
            MAPE_all(fold, dim) = MAPE;

            fprintf('Fold %d, Output %d | R²=%.4f | MSE=%.4f | RMSE=%.4f | MAPE=%.2f%%\n', ...
                fold, dim, R2, MSE, RMSE, MAPE);
        end

        Ytest_all = [Ytest_all; Ytest];
        Ypred_all = [Ypred_all; Ypred_fold];

        fprintf('\n');
    end

    fprintf('=== Average %d-Fold Metrics (k=%d) ===\n', kfold, best_k);
    for dim = 1:outdim
        fprintf('Output %d | R²=%.4f | MSE=%.4f | RMSE=%.4f | MAPE=%.2f%%\n', ...
            dim, mean(R2_all(:, dim)), mean(MSE_all(:, dim)), ...
            mean(RMSE_all(:, dim)), mean(MAPE_all(:, dim)));
    end

    % Save variables to base workspace for later use
    assignin('base', 'Ytest_all', Ytest_all);
    assignin('base', 'Ypred_all', Ypred_all);
    assignin('base', 'KNN_Xtrain', Xtrain);
    assignin('base', 'KNN_Ytrain', Ytrain);
    assignin('base', 'KNN_k', best_k);
    KNNModel = struct;
    KNNModel.k = best_k;
    KNNModel.Xtrain = Xtrain;
    KNNModel.Ytrain = Ytrain;
    KNNModel.mu = mu;
    KNNModel.sigma = sigma;

    assignin('base', 'TrainedKNNModel', KNNModel);

% Objective function for Bayesian optimization
function rmse = knnObjective(k, Xnorm, Y, cv, outdim)
    k = round(k); % Ensure k is an integer
    k = max(1, k);

    kfold = cv.NumTestSets;
    RMSE_all = zeros(kfold, outdim);

    for fold = 1:kfold
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        Xtrain = Xnorm(trainIdx, :);
        Ytrain = Y(trainIdx, :);
        Xtest = Xnorm(testIdx, :);
        Ytest = Y(testIdx, :);

        for dim = 1:outdim
            y_train = Ytrain(:, dim);
            y_test = Ytest(:, dim);

            y_pred = knn_regression_predict(Xtrain, y_train, Xtest, k);

            RMSE_all(fold, dim) = sqrt(mean((y_test - y_pred).^2));
        end
    end

    rmse = mean(RMSE_all, 'all'); % Minimize average RMSE


% KNN regression prediction function
function Ypred = knn_regression_predict(Xtrain, Ytrain, Xtest, k)
    num_test = size(Xtest, 1);
    Ypred = zeros(num_test, 1);
    for i = 1:num_test
        dists = sqrt(sum((Xtrain - Xtest(i, :)).^2, 2));
        [~, idx] = sort(dists);
        nearest_idx = idx(1:k);
        Ypred(i) = mean(Ytrain(nearest_idx));
    end



function edit2_Callback(hObject, eventdata, handles)
% No additional code needed here


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% No additional code needed here


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
