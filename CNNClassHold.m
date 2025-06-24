function varargout = CNNClassHold(varargin)
% CNNCLASSHOLD MATLAB code for CNNClassHold.fig
%      CNNCLASSHOLD, by itself, creates a new CNNCLASSHOLD or raises the existing
%      singleton*.
%
%      H = CNNCLASSHOLD returns the handle to a new CNNCLASSHOLD or the handle to
%      the existing singleton*.
%
%      CNNCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CNNCLASSHOLD.M with the given input arguments.
%
%      CNNCLASSHOLD('Property','Value',...) creates a new CNNCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CNNClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CNNClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CNNClassHold

% Last Modified by GUIDE v2.5 04-Jun-2025 15:10:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CNNClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @CNNClassHold_OutputFcn, ...
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


% --- Executes just before CNNClassHold is made visible.
function CNNClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CNNClassHold (see VARARGIN)

% Choose default command line output for CNNClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CNNClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CNNClassHold_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
filename = get(handles.edit1, 'string');
proportion = str2double(get(handles.edit2, 'string'));
res = xlsread(filename);
outdim = numel(unique(res(:, end)));

num_samples = size(res, 1);
res = res(randperm(num_samples), :); 

f_ = size(res, 2) - 1;  
X = res(:, 1:f_);      
Y = res(:, f_ + 1:end); 

Y = categorical(Y);

%% Train/Test Split
cv = cvpartition(size(Y,1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

%% Normalize Features
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma==0) = 1;
Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest = (Xtest_raw - mu) ./ sigma;

numFeatures = f_;
Xtrain_cnn = reshape(Xtrain', [numFeatures, 1, 1, size(Xtrain,1)]);
Xtest_cnn = reshape(Xtest', [numFeatures, 1, 1, size(Xtest,1)]);

%% Bayesopt validation split
cv2 = cvpartition(size(Ytrain,1), 'HoldOut', 0.2);
Xtrain_bayes = Xtrain(training(cv2), :);
Ytrain_bayes = Ytrain(training(cv2), :);
Xval_bayes = Xtrain(test(cv2), :);
Yval_bayes = Ytrain(test(cv2), :);
Xtrain_bayes_cnn = reshape(Xtrain_bayes', [numFeatures,1,1,size(Xtrain_bayes,1)]);
Xval_bayes_cnn = reshape(Xval_bayes', [numFeatures,1,1,size(Xval_bayes,1)]);

%% Bayesian Optimization
objFcn = @(params) trainEvalCNN_class(params, Xtrain_bayes_cnn, Ytrain_bayes, Xval_bayes_cnn, Yval_bayes, outdim);
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

%% Final training
layers = [
    imageInputLayer([numFeatures 1 1], 'Normalization', 'none')
    convolution2dLayer([bestParams.FilterSize 1], bestParams.NumFilters, 'Padding', 'same')
    reluLayer
    fullyConnectedLayer(outdim)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', bestParams.MaxEpochs, ...
    'InitialLearnRate', bestParams.InitialLearnRate, ...
    'MiniBatchSize', bestParams.MiniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

net = trainNetwork(Xtrain_cnn, Ytrain, layers, options);

%% Prediction
Ytrain_pred = classify(net, Xtrain_cnn);
Ytest_pred = classify(net, Xtest_cnn);
Xall = [Xtrain; Xtest];
Yall = [Ytrain; Ytest];
Yall_pred = classify(net, reshape(Xall', [numFeatures,1,1,size(Xall,1)]));

%% Evaluation
[train_acc, train_precision, train_recall, train_f1] = classification_metrics(Ytrain, Ytrain_pred);
[test_acc, test_precision, test_recall, test_f1] = classification_metrics(Ytest, Ytest_pred);
[all_acc, all_precision, all_recall, all_f1] = classification_metrics(Yall, Yall_pred);

fprintf('[Training Set Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    train_acc, train_precision, train_recall, train_f1);
fprintf('[Test Set Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    test_acc, test_precision, test_recall, test_f1);
fprintf('[All Data Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    all_acc, all_precision, all_recall, all_f1);

%% Confusion Matrix
figure;
confusionchart(Ytest, Ytest_pred);
title('Confusion Matrix (Test Set)');

%% Save to base workspace
assignin('base', 'TrainedCNNModel', net);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest_pred', Ytest_pred);
assignin('base', 'Yall_pred', Yall_pred);

%% Local function for bayesopt
function valLoss = trainEvalCNN_class(params, Xtrain, Ytrain, Xval, Yval, outdim)
    layers = [
        imageInputLayer([size(Xtrain,1) 1 1], 'Normalization', 'none')
        convolution2dLayer([params.FilterSize 1], params.NumFilters, 'Padding', 'same')
        reluLayer
        fullyConnectedLayer(outdim)
        softmaxLayer
        classificationLayer];

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
    Yval_pred = classify(net, Xval);
    valLoss = 1 - mean(Yval_pred == Yval);

%% Classification metrics function
function [acc, precision, recall, f1] = classification_metrics(ytrue, ypred)
    classes = categories(ytrue);
    num_classes = numel(classes);
    confMat = confusionmat(ytrue, ypred);
    tp = diag(confMat);
    fp = sum(confMat,1)' - tp;
    fn = sum(confMat,2) - tp;
    precision = mean(tp ./ (tp + fp + eps));
    recall = mean(tp ./ (tp + fn + eps));
    f1 = 2 * precision * recall / (precision + recall + eps);
    acc = sum(tp) / sum(confMat(:));

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
