function varargout = BpANNClassCross(varargin)
% BPANNCLASSCROSS MATLAB code for BpANNClassCross.fig
%      BPANNCLASSCROSS, by itself, creates a new BPANNCLASSCROSS or raises the existing
%      singleton*.
%
%      H = BPANNCLASSCROSS returns the handle to a new BPANNCLASSCROSS or the handle to
%      the existing singleton*.
%
%      BPANNCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BPANNCLASSCROSS.M with the given input arguments.
%
%      BPANNCLASSCROSS('Property','Value',...) creates a new BPANNCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BpANNClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BpANNClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BpANNClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 20:23:32

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BpANNClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @BpANNClassCross_OutputFcn, ...
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


% --- Executes just before BpANNClassCross is made visible.
function BpANNClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BpANNClassCross (see VARARGIN)

% Choose default command line output for BpANNClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BpANNClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BpANNClassCross_OutputFcn(hObject, eventdata, handles) 
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
    kfold = str2double(get(handles.edit2, 'string'));   % 交叉验证折数
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

% 标准化特征
mu = mean(X);
sigma = std(X);
sigma(sigma == 0) = 1;
Xnorm = (X - mu) ./ sigma;

numFeatures = f_;


%% --- Bayesian Optimization ---
objFcn = @(params) kFoldLossANN_Classification(params, Xnorm, Y, numFeatures, kfold);

optVars = [
    optimizableVariable('NumHiddenUnits', [10, 200], 'Type', 'integer')
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

%% --- Final k-Fold Training with Best Parameters ---
cv = cvpartition(Y, 'KFold', kfold);
allMetrics = struct('Accuracy', [], 'Precision', [], 'Recall', [], 'F1', []);

for i = 1:kfold
    Xtrain = Xnorm(training(cv,i), :);
    Ytrain = Y(training(cv,i));
    Xval = Xnorm(test(cv,i), :);
    Yval = Y(test(cv,i));

    classes = categories(Y);
    numClasses = numel(classes);

    layers = [
        featureInputLayer(numFeatures, 'Normalization', 'none')
        fullyConnectedLayer(bestParams.NumHiddenUnits)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', bestParams.MaxEpochs, ...
        'InitialLearnRate', bestParams.InitialLearnRate, ...
        'MiniBatchSize', bestParams.MiniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false);

    net = trainNetwork(Xtrain, Ytrain, layers, options);
    Ypred = classify(net, Xval);

    [acc, prec, rec, f1] = classification_metrics(Yval, Ypred);
    allMetrics.Accuracy(i)  = acc;
    allMetrics.Precision(i) = prec;
    allMetrics.Recall(i)    = rec;
    allMetrics.F1(i)        = f1;

    % 打印当前折的结果
    fprintf('Fold %d - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1 Score: %.4f\n', ...
        i, acc, prec, rec, f1);

    % 可视化混淆矩阵
    figure;
    confusionchart(Yval, Ypred);
    title(sprintf('Fold %d - Confusion Matrix', i));
end
assignin('base', 'TrainedBpANNModel', net);
%% --- 输出平均性能指标 ---
fprintf('\n=== Final Model Classification Performance (k-Fold = %d) ===\n', kfold);
fprintf('Average Accuracy : %.4f\n', mean(allMetrics.Accuracy));
fprintf('Average Precision: %.4f\n', mean(allMetrics.Precision));
fprintf('Average Recall   : %.4f\n', mean(allMetrics.Recall));
fprintf('Average F1 Score : %.4f\n', mean(allMetrics.F1));

%% --- 函数：用于贝叶斯优化中的k折交叉验证目标函数 ---
function avgF1 = kFoldLossANN_Classification(params, X, Y, numFeatures, k)
    cv = cvpartition(Y, 'KFold', k);
    f1_all = zeros(k,1);

    for i = 1:k
        Xtrain = X(training(cv,i), :);
        Ytrain = Y(training(cv,i));
        Xval = X(test(cv,i), :);
        Yval = Y(test(cv,i));

        classes = categories(Y);
        numClasses = numel(classes);

        layers = [
            featureInputLayer(numFeatures, 'Normalization', 'none')
            fullyConnectedLayer(params.NumHiddenUnits)
            reluLayer
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer
        ];

        options = trainingOptions('adam', ...
            'MaxEpochs', params.MaxEpochs, ...
            'InitialLearnRate', params.InitialLearnRate, ...
            'MiniBatchSize', params.MiniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Ypred = classify(net, Xval);

        [~, ~, ~, f1] = classification_metrics(Yval, Ypred);
        f1_all(i) = f1;
    end

    avgF1 = -mean(f1_all);  % 负的 F1 作为最小化目标


%% --- 函数：分类性能评估指标 ---
function [acc, prec, rec, f1] = classification_metrics(Ytrue, Ypred)
    % 输入应为 categorical 类型
    confMat = confusionmat(Ytrue, Ypred);
    numClasses = size(confMat, 1);

    TP = diag(confMat);
    FP = sum(confMat, 1)' - TP;
    FN = sum(confMat, 2) - TP;

    precision = mean(TP ./ (TP + FP + eps));
    recall    = mean(TP ./ (TP + FN + eps));
    f1_score  = mean(2 * (precision .* recall) ./ (precision + recall + eps));
    acc       = sum(TP) / sum(confMat(:));

    prec = precision;
    rec = recall;
    f1 = f1_score;



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
