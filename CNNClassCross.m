function varargout = CNNClassCross(varargin)
% CNNCLASSCROSS MATLAB code for CNNClassCross.fig
%      CNNCLASSCROSS, by itself, creates a new CNNCLASSCROSS or raises the existing
%      singleton*.
%
%      H = CNNCLASSCROSS returns the handle to a new CNNCLASSCROSS or the handle to
%      the existing singleton*.
%
%      CNNCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CNNCLASSCROSS.M with the given input arguments.
%
%      CNNCLASSCROSS('Property','Value',...) creates a new CNNCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CNNClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CNNClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CNNClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 20:41:06

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CNNClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @CNNClassCross_OutputFcn, ...
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


% --- Executes just before CNNClassCross is made visible.
function CNNClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CNNClassCross (see VARARGIN)

% Choose default command line output for CNNClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CNNClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CNNClassCross_OutputFcn(hObject, eventdata, handles) 
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
% 转为分类标签
Y = categorical(Y);
%% --- Normalize Features ---
mu = mean(X);
sigma = std(X);
sigma(sigma == 0) = 1;
Xnorm = (X - mu) ./ sigma;
numFeatures = f_;
X_cnn = reshape(Xnorm', [numFeatures, 1, 1, size(Xnorm,1)]);

%% --- Define Objective Function for bayesopt ---
objFcn = @(params) kFoldEvalCNN_Classification(params, X_cnn, Y, k);

%% --- Define Hyperparameters ---
optVars = [
    optimizableVariable('NumFilters', [8, 64], 'Type', 'integer')
    optimizableVariable('FilterSize', [2, 5], 'Type', 'integer')
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

%% --- Final k-Fold Training and Evaluation ---
fprintf('\n=== Final Training using %d-Fold Cross-Validation ===\n', k);
cv = cvpartition(Y, 'KFold', k);
allMetrics = struct('Accuracy', [], 'Precision', [], 'Recall', [], 'F1', []);

for i = 1:k
    fprintf('\n-- Fold %d/%d --\n', i, k);
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);

    Xtrain = X_cnn(:,:,:,trainIdx);
    Ytrain = Y(trainIdx);
    Xtest  = X_cnn(:,:,:,testIdx);
    Ytest  = Y(testIdx);

    numClasses = numel(categories(Y));
    
    layers = [
        imageInputLayer([numFeatures 1 1], 'Normalization', 'none')
        convolution2dLayer([bestParams.FilterSize 1], bestParams.NumFilters, 'Padding', 'same')
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
        'Verbose', 0, ...
        'Plots', 'none');

    net = trainNetwork(Xtrain, Ytrain, layers, options);
    Ypred = classify(net, Xtest);

    [acc, prec, rec, f1] = classification_metrics(Ytest, Ypred);
    allMetrics.Accuracy(i)  = acc;
    allMetrics.Precision(i) = prec;
    allMetrics.Recall(i)    = rec;
    allMetrics.F1(i)        = f1;

    fprintf('Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n', acc, prec, rec, f1);

    % 可视化混淆矩阵
    figure;
    confusionchart(Ytest, Ypred);
    title(sprintf('Fold %d - Confusion Matrix', i));
end

%% --- Output Average Metrics ---
fprintf('\n=== Average Performance Across %d Folds ===\n', k);
fprintf('Average Accuracy : %.4f\n', mean(allMetrics.Accuracy));
fprintf('Average Precision: %.4f\n', mean(allMetrics.Precision));
fprintf('Average Recall   : %.4f\n', mean(allMetrics.Recall));
fprintf('Average F1 Score : %.4f\n', mean(allMetrics.F1));

assignin('base', 'TrainedCNNModel', net);
%% === Functions ===

function loss = kFoldEvalCNN_Classification(params, X, Y, k)
    c = cvpartition(Y, 'KFold', k);
    f1List = zeros(k,1);
    numClasses = numel(categories(Y));
    for i = 1:k
        trainIdx = training(c, i);
        valIdx = test(c, i);

        Xtrain = X(:,:,:,trainIdx);
        Ytrain = Y(trainIdx);
        Xval = X(:,:,:,valIdx);
        Yval = Y(valIdx);

        layers = [
            imageInputLayer([size(X,1) 1 1], 'Normalization', 'none')
            convolution2dLayer([params.FilterSize 1], params.NumFilters, 'Padding', 'same')
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
            'Verbose', 0, ...
            'Plots', 'none');

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Yval_pred = classify(net, Xval);

        [~, ~, ~, f1] = classification_metrics(Yval, Yval_pred);
        f1List(i) = f1;
    end
    loss = -mean(f1List);  % negative F1-score as loss


function [acc, prec, rec, f1] = classification_metrics(Ytrue, Ypred)
    confMat = confusionmat(Ytrue, Ypred);
    TP = diag(confMat);
    FP = sum(confMat,1)' - TP;
    FN = sum(confMat,2) - TP;

    precision = mean(TP ./ (TP + FP + eps));
    recall    = mean(TP ./ (TP + FN + eps));
    f1        = mean(2 * (precision .* recall) ./ (precision + recall + eps));
    acc       = sum(TP) / sum(confMat(:));

    prec = precision;
    rec = recall;


% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
