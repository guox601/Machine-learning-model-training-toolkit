function varargout = BiLSTMClassCross(varargin)
% BILSTMCLASSCROSS MATLAB code for BiLSTMClassCross.fig
%      BILSTMCLASSCROSS, by itself, creates a new BILSTMCLASSCROSS or raises the existing
%      singleton*.
%
%      H = BILSTMCLASSCROSS returns the handle to a new BILSTMCLASSCROSS or the handle to
%      the existing singleton*.
%
%      BILSTMCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BILSTMCLASSCROSS.M with the given input arguments.
%
%      BILSTMCLASSCROSS('Property','Value',...) creates a new BILSTMCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BiLSTMClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BiLSTMClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BiLSTMClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 20:31:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BiLSTMClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @BiLSTMClassCross_OutputFcn, ...
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


% --- Executes just before BiLSTMClassCross is made visible.
function BiLSTMClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BiLSTMClassCross (see VARARGIN)

% Choose default command line output for BiLSTMClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BiLSTMClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BiLSTMClassCross_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on mouse press over figure background.
function figure1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



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
    K = str2double(get(handles.edit2, 'string'));   % 交叉验证折数
    outdim = 1;  % 输出维度数

    % === 读取并打乱数据 ===
    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % 打乱样本顺序

    % === 拆分特征和标签 ===
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);
% 转为分类标签
Y = categorical(Y);
numFeatures = f_;

% === Define hyperparameter search space ===
optimVars = [
    optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
];

% === Number of classes ===
numClasses = numel(categories(Y));

% === Objective function (k-fold BiLSTM classification) ===
k = 5;
objFcn = @(optVars) kFoldBiLSTM_class(optVars, X, Y, k, numFeatures, numClasses);

% === Bayesian optimization ===
results = bayesopt(objFcn, optimVars, ...
    'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

% === Best hyperparameters ===
bestParams = results.XAtMinObjective;
fprintf('Best numHiddenUnits: %d\n', bestParams.numHiddenUnits);
fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);

% === Final model training on full dataset ===
% Normalize full data
mu = mean(X);
sigma = std(X);
sigma(sigma == 0) = 1;
Xnorm = (X - mu) ./ sigma;

Xall_seq = squeeze(num2cell(Xnorm', [1]));
Yall_seq = Y;

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

finalNet = trainNetwork(Xall_seq, Yall_seq, layers, options);

% === Final prediction and accuracy on full data ===
Yall_pred = classify(finalNet, Xall_seq);
final_acc = mean(Yall_pred == Yall_seq);

fprintf('\n[Final Model on All Data]\n');
fprintf('Accuracy: %.2f%%\n', final_acc*100);

% === Confusion matrix for full dataset ===
figure;
confusionchart(Yall_seq, Yall_pred);
title('Confusion Matrix (Full Dataset)');

% === ===
% === k-fold final training and evaluation using bestParams ===
fprintf('\n=== k-Fold Final Training and Evaluation ===\n');
cv = cvpartition(size(Y, 1), 'KFold', k);

accuracies = zeros(k,1);
precisions = zeros(k,1);
recalls = zeros(k,1);
f1scores = zeros(k,1);

for i = 1:k
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    Xtrain_raw = X(trainIdx, :);
    Ytrain = Y(trainIdx, :);
    Xtest_raw = X(testIdx, :);
    Ytest = Y(testIdx, :);
    
    % 标准化
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;
    
    Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
    Xtest_seq  = squeeze(num2cell(Xtest', [1]));
    
    % 网络结构
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
        'Verbose', 1);
    
    % 训练模型
    net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
    
    % 预测
    Ypred = classify(net, Xtest_seq);
    
    % 混淆矩阵
    C = confusionmat(Ytest, Ypred);
    
    % 计算指标
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;
    
    precision = TP ./ (TP + FP + eps);
    recall = TP ./ (TP + FN + eps);
    f1 = 2 * (precision .* recall) ./ (precision + recall + eps);
    
    precisions(i) = mean(precision);
    recalls(i) = mean(recall);
    f1scores(i) = mean(f1);
    accuracies(i) = mean(Ypred == Ytest);
    
    fprintf('Fold %d:\n', i);
    fprintf('  Accuracy: %.4f\n', accuracies(i));
    fprintf('  Precision: %.4f\n', precisions(i));
    fprintf('  Recall: %.4f\n', recalls(i));
    fprintf('  F1 Score: %.4f\n\n', f1scores(i));
end

fprintf('=== k-Fold Average Metrics ===\n');
fprintf('Accuracy: %.4f\n', mean(accuracies));
fprintf('Precision: %.4f\n', mean(precisions));
fprintf('Recall: %.4f\n', mean(recalls));
fprintf('F1 Score: %.4f\n', mean(f1scores));
assignin('base', 'TrainedBiLSTMModel', net);


% === k-Fold BiLSTM objective function for hyperparameter tuning ===
function objective = kFoldBiLSTM_class(optVars, X, Y, k, numFeatures, numClasses)
    cv = cvpartition(size(Y, 1), 'KFold', k);
    accs = zeros(k, 1);

    for i = 1:k
        trainIdx = training(cv, i);
        testIdx = test(cv, i);

        Xtrain_raw = X(trainIdx, :);
        Ytrain = Y(trainIdx, :);
        Xtest_raw = X(testIdx, :);
        Ytest = Y(testIdx, :);

        mu = mean(Xtrain_raw);
        sigma = std(Xtrain_raw);
        sigma(sigma == 0) = 1;
        Xtrain = (Xtrain_raw - mu) ./ sigma;
        Xtest  = (Xtest_raw - mu) ./ sigma;

        Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
        Xtest_seq  = squeeze(num2cell(Xtest', [1]));

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

        net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
        Ypred = classify(net, Xtest_seq);

        accs(i) = mean(Ypred == Ytest);
    end

    objective = 1 - mean(accs);


% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
