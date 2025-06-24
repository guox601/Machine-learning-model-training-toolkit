function varargout = BpANNClassHold(varargin)
% BPANNCLASSHOLD MATLAB code for BpANNClassHold.fig
%      BPANNCLASSHOLD, by itself, creates a new BPANNCLASSHOLD or raises the existing
%      singleton*.
%
%      H = BPANNCLASSHOLD returns the handle to a new BPANNCLASSHOLD or the handle to
%      the existing singleton*.
%
%      BPANNCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BPANNCLASSHOLD.M with the given input arguments.
%
%      BPANNCLASSHOLD('Property','Value',...) creates a new BPANNCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BpANNClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BpANNClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BpANNClassHold

% Last Modified by GUIDE v2.5 03-Jun-2025 23:07:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BpANNClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @BpANNClassHold_OutputFcn, ...
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


% --- Executes just before BpANNClassHold is made visible.
function BpANNClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BpANNClassHold (see VARARGIN)

% Choose default command line output for BpANNClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BpANNClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BpANNClassHold_OutputFcn(hObject, eventdata, handles) 
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
    outdim = 1; 
    res = xlsread(filename);
    num_samples = size(res, 1);
res = res(randperm(num_samples), :); 

f_ = size(res, 2) - outdim;
X = res(:, 1:f_);      
Yraw = res(:, f_ + 1:end);  
Y = categorical(Yraw);

numClasses = numel(categories(Y));

%% --- Train/Test Split ---
cv = cvpartition(size(Y,1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

%% --- Normalize Features ---
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma==0) = 1;

Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest = (Xtest_raw - mu) ./ sigma;

numFeatures = f_;

%% --- Split training set into train/validation for bayesopt ---
cv2 = cvpartition(size(Ytrain,1), 'HoldOut', 0.2);
Xtrain_bayes = Xtrain(training(cv2), :);
Ytrain_bayes = Ytrain(training(cv2), :);
Xval_bayes = Xtrain(test(cv2), :);
Yval_bayes = Ytrain(test(cv2), :);

%% --- Define objective function for bayesopt ---
objFcn = @(params) trainEvalANN(params, Xtrain_bayes, Ytrain_bayes, Xval_bayes, Yval_bayes, numFeatures, numClasses);

%% --- Define hyperparameters to optimize ---
optVars = [
    optimizableVariable('NumHiddenUnits', [10, 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4, 1e-2], 'Transform', 'log')
    optimizableVariable('MaxEpochs', [50, 300], 'Type', 'integer')
    optimizableVariable('MiniBatchSize', [8, 64], 'Type', 'integer')
];

%% --- Run Bayesian Optimization ---
results = bayesopt(objFcn, optVars, ...
    'MaxObjectiveEvaluations', 20, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'UseParallel', false);

bestParams = results.XAtMinObjective;

%% --- Train final model on full training set with best hyperparameters ---
layers = [
    featureInputLayer(numFeatures, 'Normalization', 'none')
    fullyConnectedLayer(bestParams.NumHiddenUnits)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', bestParams.MaxEpochs, ...
    'InitialLearnRate', bestParams.InitialLearnRate, ...
    'MiniBatchSize', bestParams.MiniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

net = trainNetwork(Xtrain, Ytrain, layers, options);

%% --- Predict ---
Ytrain_pred = classify(net, Xtrain);
Ytest_pred = classify(net, Xtest);

Xall = (X - mu) ./ sigma;
Yall_pred = classify(net, Xall);

%% --- Evaluation metrics ---
train_acc = sum(Ytrain_pred == Ytrain) / numel(Ytrain);
test_acc = sum(Ytest_pred == Ytest) / numel(Ytest);
all_acc = sum(Yall_pred == Y) / numel(Y);

fprintf('\n[Training Set]\nAccuracy: %.2f%%\n', train_acc*100);
fprintf('\n[Test Set]\nAccuracy: %.2f%%\n', test_acc*100);
fprintf('\n[Overall Dataset]\nAccuracy: %.2f%%\n', all_acc*100);

%% --- Plot Confusion Matrix ---
figure;
confusionchart(Ytest, Ytest_pred);
title('Confusion Matrix (Test Set)');

    assignin('base', 'TrainedBpANNModel', net);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest_pred', Ytest_pred);
    assignin('base', 'Yall_pred', Yall_pred);
    % === Calculate metrics ===
[train_acc, train_prec, train_rec, train_f1] = classificationMetrics(Ytrain, Ytrain_pred);
[test_acc, test_prec, test_rec, test_f1] = classificationMetrics(Ytest, Ytest_pred);
[all_acc, all_prec, all_rec, all_f1] = classificationMetrics(Y, Yall_pred);

% === Print metrics ===
fprintf('\n[Training Set Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    train_acc, train_prec, train_rec, train_f1);
fprintf('\n[Test Set Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    test_acc, test_prec, test_rec, test_f1);
fprintf('\n[All Data Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    all_acc, all_prec, all_rec, all_f1);

%% ======= Local function =======
function valLoss = trainEvalANN(params, Xtrain, Ytrain, Xval, Yval, numFeatures, numClasses)
    layers = [
        featureInputLayer(numFeatures, 'Normalization', 'none')
        fullyConnectedLayer(params.NumHiddenUnits)
        reluLayer
        fullyConnectedLayer(numClasses)
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
    
    valLoss = 1 - sum(Yval_pred == Yval) / numel(Yval);  % 用错误率作为目标函数
    
    
   
    % === Evaluation Metrics Function ===
function [acc, precision, recall, f1] = classificationMetrics(Ytrue, Ypred)
    classes = categories(Ytrue);
    numClasses = numel(classes);
    confMat = confusionmat(Ytrue, Ypred);
    
    TP = diag(confMat);
    FP = sum(confMat, 1)' - TP;
    FN = sum(confMat, 2) - TP;

    precision = mean(TP ./ max(TP + FP, eps));
    recall = mean(TP ./ max(TP + FN, eps));
    f1 = 2 * precision * recall / max(precision + recall, eps);
    acc = sum(TP) / sum(confMat(:));


% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
