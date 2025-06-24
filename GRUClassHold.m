function varargout = GRUClassHold(varargin)
% GRUCLASSHOLD MATLAB code for GRUClassHold.fig
%      GRUCLASSHOLD, by itself, creates a new GRUCLASSHOLD or raises the existing
%      singleton*.
%
%      H = GRUCLASSHOLD returns the handle to a new GRUCLASSHOLD or the handle to
%      the existing singleton*.
%
%      GRUCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GRUCLASSHOLD.M with the given input arguments.
%
%      GRUCLASSHOLD('Property','Value',...) creates a new GRUCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GRUClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GRUClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GRUClassHold

% Last Modified by GUIDE v2.5 04-Jun-2025 15:27:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GRUClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @GRUClassHold_OutputFcn, ...
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


% --- Executes just before GRUClassHold is made visible.
function GRUClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GRUClassHold (see VARARGIN)

% Choose default command line output for GRUClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GRUClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GRUClassHold_OutputFcn(hObject, eventdata, handles) 
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
res = res(randperm(num_samples), :);  % Shuffle rows

f_ = size(res, 2) - outdim;
X = res(:, 1:f_);
Y = res(:, f_ + 1:end);  

% Convert numeric labels to categorical for classification
Y = categorical(Y);
cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1;

Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest = (Xtest_raw - mu) ./ sigma;
Xall = (X - mu) ./ sigma;

% Prepar input format (cell array, each cell is feature x time)
Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
Xtest_seq = squeeze(num2cell(Xtest', [1]));
Xall_seq = squeeze(num2cell(Xall', [1]));

Ytrain_seq = Ytrain;
Ytest_seq = Ytest;
Yall_seq = Y;

numFeatures = f_;
numClasses = numel(categories(Y));  % Number of classes

% === Define hyperparameter search space ===
optimVars = [
    optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
];

% === Wrap objective for bayesopt ===
objFcn = @(optVars) trainAndEvaluate(optVars, Xtrain_seq, Ytrain_seq, Xtest_seq, Ytest_seq, numFeatures, numClasses);

% === Run Bayesian optimization ===
results = bayesopt(objFcn, optimVars, ...
    'MaxObjectiveEvaluations', 100, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

% === Show best hyperparameters ===
bestParams = results.XAtMinObjective;
fprintf('Best numHiddenUnits: %d\n', bestParams.numHiddenUnits);
fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);

% === Train final model with best hyperparameters ===
layers = [
    sequenceInputLayer(numFeatures)
    gruLayer(bestParams.numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', bestParams.InitialLearnRate, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

finalNet = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);

% === Predict ===
Ytrain_pred = classify(finalNet, Xtrain_seq);
Ytest_pred = classify(finalNet, Xtest_seq);
Yall_pred = classify(finalNet, Xall_seq);

% === Calculate accuracy ===
calc_acc = @(Ytrue, Ypred) mean(Ytrue == Ypred) * 100;
train_acc = calc_acc(Ytrain_seq, Ytrain_pred);
test_acc = calc_acc(Ytest_seq, Ytest_pred);
all_acc = calc_acc(Yall_seq, Yall_pred);

% === Calculate precision, recall, F1 ===
% Helper to calculate micro-averaged metrics
calc_metrics = @(Ytrue, Ypred) deal_metrics(Ytrue, Ypred);

[train_prec, train_recall, train_f1] = calc_metrics(Ytrain_seq, Ytrain_pred);
[test_prec, test_recall, test_f1] = calc_metrics(Ytest_seq, Ytest_pred);
[all_prec, all_recall, all_f1] = calc_metrics(Yall_seq, Yall_pred);

% === Display results ===
fprintf('\n[Training Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    train_acc / 100, train_prec, train_recall, train_f1);

fprintf('[Test Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    test_acc / 100, test_prec, test_recall, test_f1);

fprintf('[All Data Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    all_acc / 100, all_prec, all_recall, all_f1);

% === Plot confusion matrix ===
figure;
confusionchart(Ytest_seq, Ytest_pred);
title('GRU Classification Results (Test Set)');

% === Save to base workspace ===
assignin('base', 'TrainedGRUModel', finalNet);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest_pred', Ytest_pred);
assignin('base', 'Yall_pred', Yall_pred);

% === Objective function for bayesopt ===
function objective = trainAndEvaluate(optVars, Xtrain_seq, Ytrain_seq, Xtest_seq, Ytest_seq, numFeatures, numClasses)
    layers = [
        sequenceInputLayer(numFeatures)
        gruLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'InitialLearnRate', optVars.InitialLearnRate, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0);
    
    net = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);
    
    Ypred = classify(net, Xtest_seq);
    
    accuracy = mean(Ypred == Ytest_seq);
    % bayesopt minimizes, so return negative accuracy
    objective = -accuracy;
function [precision, recall, f1] = deal_metrics(Ytrue, Ypred)
    Ytrue = double(grp2idx(Ytrue));
    Ypred = double(grp2idx(Ypred));
    classes = unique([Ytrue; Ypred]);
    numClasses = numel(classes);
    TP = zeros(numClasses,1);
    FP = zeros(numClasses,1);
    FN = zeros(numClasses,1);
    for i = 1:numClasses
        TP(i) = sum(Ytrue == i & Ypred == i);
        FP(i) = sum(Ytrue ~= i & Ypred == i);
        FN(i) = sum(Ytrue == i & Ypred ~= i);
    end
    precision = sum(TP) / (sum(TP) + sum(FP) + eps);
    recall = sum(TP) / (sum(TP) + sum(FN) + eps);
    f1 = 2 * precision * recall / (precision + recall + eps);
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
