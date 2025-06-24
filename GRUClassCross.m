function varargout = GRUClassCross(varargin)
% GRUCLASSCROSS MATLAB code for GRUClassCross.fig
%      GRUCLASSCROSS, by itself, creates a new GRUCLASSCROSS or raises the existing
%      singleton*.
%
%      H = GRUCLASSCROSS returns the handle to a new GRUCLASSCROSS or the handle to
%      the existing singleton*.
%
%      GRUCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GRUCLASSCROSS.M with the given input arguments.
%
%      GRUCLASSCROSS('Property','Value',...) creates a new GRUCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GRUClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GRUClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GRUClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 20:44:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GRUClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @GRUClassCross_OutputFcn, ...
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


% --- Executes just before GRUClassCross is made visible.
function GRUClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GRUClassCross (see VARARGIN)

% Choose default command line output for GRUClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GRUClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GRUClassCross_OutputFcn(hObject, eventdata, handles) 
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
% === 读取输入参数 ===
    filename = get(handles.edit1, 'string');            % Excel 文件路径
    k = str2double(get(handles.edit2, 'string'));   % 交叉验证折数
    outdim = 1;  % 输出维度数

    % === 读取并打乱数据 ===
    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % 打乱样本顺序

    % === 拆分特征和标签 ===
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);
numFeatures = f_;

classes = unique(Y);
numClasses = numel(classes);
Y = grp2idx(Y);  % 转换成整数类别标签（1, 2, 3, ...）

% === Hyperparameter search space ===
optimVars = [
    optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
];

% === Run Bayesian optimization with k-fold ===
objFcn = @(optVars) kfold_GRU_class_cv(X, Y, optVars, k, numClasses);

results = bayesopt(objFcn, optimVars, ...
    'MaxObjectiveEvaluations', 20, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

bestParams = results.XAtMinObjective;
fprintf('\nBest numHiddenUnits: %d\n', bestParams.numHiddenUnits);
fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);

% === Final k-Fold Training with Best Hyperparameters ===
fprintf('\n=== Final k-Fold Training with Best Hyperparameters ===\n');
cv_final = cvpartition(Y, 'KFold', k);
metrics_all = zeros(k, 4);  % Accuracy, Precision, Recall, F1

for fold = 1:k
    trainIdx = training(cv_final, fold);
    testIdx = test(cv_final, fold);

    Xtrain_raw = X(trainIdx, :);
    Xtest_raw = X(testIdx, :);
    Ytrain = Y(trainIdx);
    Ytest = Y(testIdx);

    % Z-score normalization
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw); sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;

    Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
    Xtest_seq = squeeze(num2cell(Xtest', [1]));

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
        'Verbose', 0);

    net = trainNetwork(Xtrain_seq, categorical(Ytrain), layers, options);
    Ypred = classify(net, Xtest_seq);
    Ytest_cat = categorical(Ytest);

    % Compute metrics
    [acc, prec, rec, f1] = classification_metrics(Ytest_cat, Ypred);
    metrics_all(fold, :) = [acc, prec, rec, f1];

    fprintf('Fold %d: Acc = %.4f, Precision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
        fold, acc, prec, rec, f1);
end

% === Average metrics ===
avg_metrics = mean(metrics_all, 1);
fprintf('\n[Final k-Fold Average Classification Metrics]\n');
fprintf('Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-score: %.4f\n', ...
    avg_metrics(1), avg_metrics(2), avg_metrics(3), avg_metrics(4));
assignin('base', 'TrainedGRUModel', net);

%% === Function: k-Fold Classification Objective ===
function objective = kfold_GRU_class_cv(X, Y, optVars, k, numClasses)
    cv = cvpartition(Y, 'KFold', k);
    acc_all = zeros(k, 1);

    for fold = 1:k
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        Xtrain = X(trainIdx, :);
        Xtest = X(testIdx, :);
        Ytrain = Y(trainIdx);
        Ytest = Y(testIdx);

        mu = mean(Xtrain); sigma = std(Xtrain); sigma(sigma == 0) = 1;
        Xtrain = (Xtrain - mu) ./ sigma;
        Xtest = (Xtest - mu) ./ sigma;

        Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
        Xtest_seq = squeeze(num2cell(Xtest', [1]));

        layers = [
            sequenceInputLayer(size(X, 2))
            gruLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
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

        net = trainNetwork(Xtrain_seq, categorical(Ytrain), layers, options);
        Ypred = classify(net, Xtest_seq);

        acc_all(fold) = mean(Ypred == categorical(Ytest));
    end

    objective = -mean(acc_all);  % Negative accuracy for minimization


%% === Function: Classification Metrics ===
function [accuracy, precision, recall, f1] = classification_metrics(Ytrue, Ypred)
    cm = confusionmat(Ytrue, Ypred);
    TP = diag(cm);
    FP = sum(cm, 1)' - TP;
    FN = sum(cm, 2) - TP;

    precision = mean(TP ./ (TP + FP + eps));
    recall = mean(TP ./ (TP + FN + eps));
    f1 = 2 * precision * recall / (precision + recall + eps);
    accuracy = sum(TP) / sum(cm(:));



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
