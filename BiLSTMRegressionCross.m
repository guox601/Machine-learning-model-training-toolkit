function varargout = BiLSTMRegressionCross(varargin)
% BILSTMREGRESSIONCROSS MATLAB code for BiLSTMRegressionCross.fig
%      BILSTMREGRESSIONCROSS, by itself, creates a new BILSTMREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = BILSTMREGRESSIONCROSS returns the handle to a new BILSTMREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      BILSTMREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BILSTMREGRESSIONCROSS.M with the given input arguments.
%
%      BILSTMREGRESSIONCROSS('Property','Value',...) creates a new BILSTMREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BiLSTMRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BiLSTMRegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BiLSTMRegressionCross

% Last Modified by GUIDE v2.5 05-Jun-2025 22:27:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BiLSTMRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @BiLSTMRegressionCross_OutputFcn, ...
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


% --- Executes just before BiLSTMRegressionCross is made visible.
function BiLSTMRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BiLSTMRegressionCross (see VARARGIN)

% Choose default command line output for BiLSTMRegressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BiLSTMRegressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BiLSTMRegressionCross_OutputFcn(hObject, eventdata, handles) 
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
function pushbutton1_Callback(hObject, eventdata, handles)
% 获取用户输入
filename = get(handles.edit1, 'string');
outdim = str2double(get(handles.edit2, 'string'));
k = str2double(get(handles.edit3, 'string'));

% 读取数据
res = xlsread(filename);
res = res(randperm(size(res,1)), :); % 打乱数据
X = res(:, 1:end-outdim);
Y = res(:, end-outdim+1:end);
numFeatures = size(X, 2);
Xraw = X; Yraw = Y;

% 贝叶斯优化超参数设置
optimVars = [
    optimizableVariable('numHiddenUnits', [50, 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4, 1e-2], 'Transform', 'log')
];

% 定义目标函数（k 折交叉验证）
ObjFcn = @(optVars) kFoldBiLSTM(optVars, Xraw, Yraw, k, numFeatures, outdim);

% 执行贝叶斯优化
results = bayesopt(ObjFcn, optimVars, ...
    'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'UseParallel', false);

% 获取最优参数
bestParams = results.XAtMinObjective;
fprintf('Best Parameters:\n  numHiddenUnits = %d\n  InitialLearnRate = %.5f\n', ...
    bestParams.numHiddenUnits, bestParams.InitialLearnRate);

% 标准化所有数据
mu = mean(Xraw); sigma = std(Xraw); sigma(sigma == 0) = 1;
Xnorm = (Xraw - mu) ./ sigma;

% 转换为序列输入
Xall_seq = squeeze(num2cell(Xnorm', [1]));
Yall_seq = Yraw;

% 构建 BiLSTM 网络
layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(bestParams.numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(outdim)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', bestParams.InitialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'Plots', 'none');

% 在全部数据上训练模型
net = trainNetwork(Xall_seq, Yall_seq, layers, options);
Ypred_all = predict(net, Xall_seq);

% 输出指标
for d = 1:outdim
    metrics = calc_metrics(Yall_seq(:,d), Ypred_all(:,d));
    fprintf('\n[Final Model - Output %d]\nR²: %.4f\nMSE: %.4f\nRMSE: %.4f\nMAPE: %.2f%%\n', ...
        d, metrics.R2, metrics.MSE, metrics.RMSE, metrics.MAPE);
end

% 绘图
figure;
for d = 1:outdim
    subplot(ceil(outdim/2), 2, d);
    scatter(Yall_seq(:,d), Ypred_all(:,d), 'filled');
    xlabel('Actual'); ylabel('Predicted');
    title(['Output Dim ', num2str(d)]);
    grid on; refline(1,0);
end

% 保存到 Workspace
assignin('base', 'TrainedBiLSTMModel', net);
assignin('base', 'Yall_true', Yall_seq);
assignin('base', 'Yall_pred', Ypred_all);

% === k 折交叉验证评估输出 ===
fprintf('\n=== %d-Fold Cross-Validation Metrics ===\n', k);
cv = cvpartition(size(Yraw,1), 'KFold', k);
for fold = 1:k
    trainIdx = training(cv, fold); testIdx = test(cv, fold);
    Xtrain = Xraw(trainIdx, :); Ytrain = Yraw(trainIdx, :);
    Xtest = Xraw(testIdx, :); Ytest = Yraw(testIdx, :);

    % 归一化
    mu = mean(Xtrain); sigma = std(Xtrain); sigma(sigma == 0) = 1;
    Xtrain = (Xtrain - mu) ./ sigma;
    Xtest  = (Xtest  - mu) ./ sigma;

    Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
    Xtest_seq  = squeeze(num2cell(Xtest', [1]));

    % 网络结构
    layers = [
        sequenceInputLayer(numFeatures)
        bilstmLayer(bestParams.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(outdim)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...
        'InitialLearnRate', bestParams.InitialLearnRate, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0);

    net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
    Ypred = predict(net, Xtest_seq);

    for d = 1:outdim
        m = calc_metrics(Ytest(:, d), Ypred(:, d));
        fprintf('Fold %d - Output %d: R² = %.4f, MSE = %.4f, RMSE = %.4f, MAPE = %.2f%%\n', ...
            fold, d, m.R2, m.MSE, m.RMSE, m.MAPE);
    end
end

function objective = kFoldBiLSTM(optVars, Xraw, Yraw, k, numFeatures, outdim)
% k 折交叉验证下 BiLSTM 的平均 MSE（目标函数供贝叶斯优化使用）
cv = cvpartition(size(Yraw,1), 'KFold', k);
mse_folds = zeros(k, outdim);  % 每折每维输出的 MSE

for fold = 1:k
    trainIdx = training(cv, fold);
    valIdx = test(cv, fold);

    Xtrain = Xraw(trainIdx, :); Ytrain = Yraw(trainIdx, :);
    Xval = Xraw(valIdx, :);     Yval = Yraw(valIdx, :);

    % 标准化
    mu = mean(Xtrain); sigma = std(Xtrain); sigma(sigma == 0) = 1;
    Xtrain = (Xtrain - mu) ./ sigma;
    Xval = (Xval - mu) ./ sigma;

    % 转换为序列
    Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
    Xval_seq   = squeeze(num2cell(Xval', [1]));

    % 构建 BiLSTM 网络
    layers = [
        sequenceInputLayer(numFeatures)
        bilstmLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(outdim)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', optVars.InitialLearnRate, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'Plots', 'none');

    try
        net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
        Ypred = predict(net, Xval_seq);

        for d = 1:outdim
            mse_folds(fold, d) = mean((Yval(:,d) - Ypred(:,d)).^2);
        end
    catch
        % 如果出现训练错误，惩罚当前参数组合
        objective = 1e6;
        return;
    end
end

% 返回平均 MSE（作为目标函数）
objective = mean(mse_folds(:));


% === Objective function used in Bayesian optimization ===
function m = calc_metrics(ytrue, ypred)
    m.R2 = 1 - sum((ytrue - ypred).^2) / sum((ytrue - mean(ytrue)).^2);
    m.MSE = mean((ytrue - ypred).^2);
    m.RMSE = sqrt(m.MSE);
    m.MAPE = mean(abs((ytrue - ypred) ./ (ytrue + eps))) * 100;



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
