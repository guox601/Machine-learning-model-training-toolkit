function varargout = CNNRegressionHold(varargin)
% CNNREGRESSIONHOLD MATLAB code for CNNRegressionHold.fig
%      CNNREGRESSIONHOLD, by itself, creates a new CNNREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = CNNREGRESSIONHOLD returns the handle to a new CNNREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      CNNREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CNNREGRESSIONHOLD.M with the given input arguments.
%
%      CNNREGRESSIONHOLD('Property','Value',...) creates a new CNNREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CNNRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CNNRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CNNRegressionHold

% Last Modified by GUIDE v2.5 03-Jun-2025 19:43:29

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CNNRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @CNNRegressionHold_OutputFcn, ...
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


% --- Executes just before CNNRegressionHold is made visible.
function CNNRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CNNRegressionHold (see VARARGIN)

% Choose default command line output for CNNRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CNNRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CNNRegressionHold_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    filepath = get(handles.edit1, 'String');
    trainRatio = str2double(get(handles.edit2, 'String'));
    outdim = str2double(get(handles.edit3, 'String'));

    res = xlsread(filepath);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % shuffle

    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_ + 1:end);

    % Split dataset
    cv = cvpartition(size(Y,1), 'HoldOut', trainRatio);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalize
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    numFeatures = f_;

    % CNN reshape
    Xtrain_cnn = reshape(Xtrain', [numFeatures, 1, 1, size(Xtrain,1)]);
    Xtest_cnn = reshape(Xtest', [numFeatures, 1, 1, size(Xtest,1)]);

    % For bayesopt
    cv2 = cvpartition(size(Ytrain,1), 'HoldOut', 0.2);
    Xtrain_bayes = Xtrain(training(cv2), :);
    Ytrain_bayes = Ytrain(training(cv2), :);
    Xval_bayes = Xtrain(test(cv2), :);
    Yval_bayes = Ytrain(test(cv2), :);

    Xtrain_bayes_cnn = reshape(Xtrain_bayes', [numFeatures,1,1,size(Xtrain_bayes,1)]);
    Xval_bayes_cnn = reshape(Xval_bayes', [numFeatures,1,1,size(Xval_bayes,1)]);

    % Define optimization variables
    objFcn = @(params) trainEvalCNN(params, Xtrain_bayes_cnn, Ytrain_bayes, Xval_bayes_cnn, Yval_bayes, outdim);
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

    % Final model
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
        'Verbose', 1, ...
        'Plots', 'training-progress');

    net = trainNetwork(Xtrain_cnn, Ytrain, layers, options);

    % Prediction
    X_norm = (X - mu) ./ sigma;
    Xall_cnn = reshape(X_norm', [numFeatures, 1, 1, size(X_norm,1)]);

    Ytrain_pred = predict(net, Xtrain_cnn);
    Ytest_pred = predict(net, Xtest_cnn);
    Yall_pred = predict(net, Xall_cnn);

    % Per-dimension metrics
    for i = 1:outdim
        train_metrics = calc_metrics(Ytrain(:,i), Ytrain_pred(:,i));
        test_metrics  = calc_metrics(Ytest(:,i), Ytest_pred(:,i));

        fprintf('[Target %d - Training Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            i, train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);

        fprintf('[Target %d - Test Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n\n', ...
            i, test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);
    end

    % Plot only first dimension by default
for i = 1:outdim
    figure;
    scatter(Ytest(:, i), Ytest_pred(:, i), 40, 'filled');
    xlabel(sprintf('Actual Value (Target %d)', i));
    ylabel(sprintf('Predicted Value (Target %d)', i));
    title(sprintf('CNN Regression Results (Target %d - Test Set)', i));
    grid on;
    refline(1, 0); % y = x reference line
end

    % Store to base workspace
    assignin('base', 'CNN_Model', net);
    assignin('base', 'CNN_Y_Predict', Yall_pred);


function valLoss = trainEvalCNN(params, Xtrain, Ytrain, Xval, Yval, outdim)
    layers = [
        imageInputLayer([size(Xtrain,1) 1 1], 'Normalization', 'none')
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
        'Verbose', 0, ...
        'Plots', 'none', ...
        'ValidationData', {Xval, Yval}, ...
        'ValidationFrequency', 20);

    net = trainNetwork(Xtrain, Ytrain, layers, options);
    Yval_pred = predict(net, Xval);
    valLoss = sqrt(mean((Yval - Yval_pred).^2, 'all'));

function metrics = calc_metrics(Ytrue, Ypred)
    metrics.R2   = 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2);
    metrics.MSE  = mean((Ytrue - Ypred).^2);
    metrics.RMSE = sqrt(metrics.MSE);
    metrics.MAPE = mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100;

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
