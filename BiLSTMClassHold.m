function varargout = BiLSTMClassHold(varargin)
% BILSTMCLASSHOLD MATLAB code for BiLSTMClassHold.fig
%      BILSTMCLASSHOLD, by itself, creates a new BILSTMCLASSHOLD or raises the existing
%      singleton*.
%
%      H = BILSTMCLASSHOLD returns the handle to a new BILSTMCLASSHOLD or the handle to
%      the existing singleton*.
%
%      BILSTMCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BILSTMCLASSHOLD.M with the given input arguments.
%
%      BILSTMCLASSHOLD('Property','Value',...) creates a new BILSTMCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BiLSTMClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BiLSTMClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BiLSTMClassHold

% Last Modified by GUIDE v2.5 04-Jun-2025 14:58:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BiLSTMClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @BiLSTMClassHold_OutputFcn, ...
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


% --- Executes just before BiLSTMClassHold is made visible.
function BiLSTMClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BiLSTMClassHold (see VARARGIN)

% Choose default command line output for BiLSTMClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BiLSTMClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BiLSTMClassHold_OutputFcn(hObject, eventdata, handles) 
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

cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1;

Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest  = (Xtest_raw - mu) ./ sigma;
Xall   = (X - mu) ./ sigma;

Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
Xtest_seq  = squeeze(num2cell(Xtest', [1]));
Xall_seq   = squeeze(num2cell(Xall', [1]));

Ytrain_seq = Ytrain;
Ytest_seq  = Ytest;
Yall_seq   = Y;

numFeatures = f_;
numClasses = numel(categories(Y));

optimVars = [
    optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
];

objFcn = @(optVars) trainAndEvaluateBiLSTM_class(optVars, Xtrain_seq, Ytrain_seq, Xtest_seq, Ytest_seq, numFeatures, numClasses);

results = bayesopt(objFcn, optimVars, ...
    'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

bestParams = results.XAtMinObjective;
fprintf('Best numHiddenUnits: %d\n', bestParams.numHiddenUnits);
fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);

layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(bestParams.numHiddenUnits, 'OutputMode', 'last')
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

Ytrain_pred = classify(finalNet, Xtrain_seq);
Ytest_pred  = classify(finalNet, Xtest_seq);
Yall_pred   = classify(finalNet, Xall_seq);

% === Evaluation function ===
evalMetrics = @(Y_true, Y_pred) struct( ...
    'Accuracy', mean(Y_pred == Y_true), ...
    'Precision', macroPrecision(Y_true, Y_pred), ...
    'Recall', macroRecall(Y_true, Y_pred), ...
    'F1', macroF1(Y_true, Y_pred));

train_metrics = evalMetrics(Ytrain_seq, Ytrain_pred);
test_metrics = evalMetrics(Ytest_seq, Ytest_pred);
all_metrics = evalMetrics(Yall_seq, Yall_pred);

fprintf('\n[Training Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    train_metrics.Accuracy, train_metrics.Precision, train_metrics.Recall, train_metrics.F1);

fprintf('\n[Test Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    test_metrics.Accuracy, test_metrics.Precision, test_metrics.Recall, test_metrics.F1);

fprintf('\n[All Data Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    all_metrics.Accuracy, all_metrics.Precision, all_metrics.Recall, all_metrics.F1);

% Confusion matrix for test set
figure;
confusionchart(Ytest_seq, Ytest_pred);
title('Confusion Matrix (Test Set)');

% Save to base workspace
assignin('base', 'TrainedBiLSTMModel', finalNet);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest_pred', Ytest_pred);
assignin('base', 'Yall_pred', Yall_pred);

% === Nested objective function ===
function objective = trainAndEvaluateBiLSTM_class(optVars, Xtrain_seq, Ytrain_seq, Xval_seq, Yval_seq, numFeatures, numClasses)
    layers = [
        sequenceInputLayer(numFeatures)
        bilstmLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', optVars.InitialLearnRate, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0);
    
    net = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);
    Ypred = classify(net, Xval_seq);
    acc = mean(Ypred == Yval_seq);
    objective = 1 - acc;


% === Supporting functions ===
function p = macroPrecision(ytrue, ypred)
    classes = categories(ytrue);
    p = 0;
    for i = 1:numel(classes)
        tp = sum(ypred == classes{i} & ytrue == classes{i});
        fp = sum(ypred == classes{i} & ytrue ~= classes{i});
        if tp + fp == 0
            prec = 0;
        else
            prec = tp / (tp + fp);
        end
        p = p + prec;
    end
    p = p / numel(classes);


function r = macroRecall(ytrue, ypred)
    classes = categories(ytrue);
    r = 0;
    for i = 1:numel(classes)
        tp = sum(ypred == classes{i} & ytrue == classes{i});
        fn = sum(ypred ~= classes{i} & ytrue == classes{i});
        if tp + fn == 0
            rec = 0;
        else
            rec = tp / (tp + fn);
        end
        r = r + rec;
    end
    r = r / numel(classes);


function f1 = macroF1(ytrue, ypred)
    p = macroPrecision(ytrue, ypred);
    r = macroRecall(ytrue, ypred);
    if p + r == 0
        f1 = 0;
    else
        f1 = 2 * p * r / (p + r);
    end
