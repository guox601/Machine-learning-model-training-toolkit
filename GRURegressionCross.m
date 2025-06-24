function varargout = GRURegressionCross(varargin)
% GRUREGRESSIONCROSS MATLAB code for GRURegressionCross.fig
%      GRUREGRESSIONCROSS, by itself, creates a new GRUREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = GRUREGRESSIONCROSS returns the handle to a new GRUREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      GRUREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GRUREGRESSIONCROSS.M with the given input arguments.
%
%      GRUREGRESSIONCROSS('Property','Value',...) creates a new GRUREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GRURegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GRURegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GRURegressionCross

% Last Modified by GUIDE v2.5 06-Jun-2025 14:04:16

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GRURegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @GRURegressionCross_OutputFcn, ...
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


% --- Executes just before GRURegressionCross is made visible.
function GRURegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);


% --- Outputs from this function are returned to the command line.
function varargout = GRURegressionCross_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


function edit1_Callback(hObject, eventdata, handles)


function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit2_Callback(hObject, eventdata, handles)


function edit2_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit3_Callback(hObject, eventdata, handles)


function edit3_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function pushbutton1_Callback(hObject, eventdata, handles)
    % 读取输入
    filename = get(handles.edit1, 'String');    % Excel文件路径
    k = str2double(get(handles.edit3, 'String'));   % 折数k
    outdim = str2double(get(handles.edit2, 'String'));  % 输出维度
    
    % 读取数据
    data = xlsread(filename);
    numSamples = size(data, 1);
    numFeatures = size(data, 2) - outdim;
    
    X = data(:, 1:numFeatures);
    Y = data(:, numFeatures+1:end);
    
    % 优化变量定义
    optimVars = [
        optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
        optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
    ];
    
    % 运行贝叶斯优化（优化单输出情况下取第1个输出维度）
    objFcn = @(optVars) kfold_GRU_cv(X, Y(:,1), optVars, k);
    
    results = bayesopt(objFcn, optimVars, ...
        'MaxObjectiveEvaluations', 30, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1);
    
    bestParams = results.XAtMinObjective;
    fprintf('\nBest numHiddenUnits: %d\n', bestParams.numHiddenUnits);
    fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);
    
    % 终极k折训练与评估 (多输出维度)
    fprintf('\n=== Final k-Fold Training with Best Hyperparameters ===\n');
    cv_final = cvpartition(numSamples, 'KFold', k);
    
    metrics_all = zeros(k, 4, outdim); % 维度：折数 x 指标数 x 输出维度
    % 指标顺序: R2, MSE, RMSE, MAPE
    
    for fold = 1:k
        trainIdx = training(cv_final, fold);
        testIdx = test(cv_final, fold);
        
        Xtrain_raw = X(trainIdx, :);
        Xtest_raw = X(testIdx, :);
        Ytrain = Y(trainIdx, :);
        Ytest = Y(testIdx, :);
        
        % 训练集归一化（均值方差）
        mu = mean(Xtrain_raw);
        sigma = std(Xtrain_raw);
        sigma(sigma==0) = 1;
        Xtrain = (Xtrain_raw - mu) ./ sigma;
        Xtest = (Xtest_raw - mu) ./ sigma;
        
        % 转成序列 cell array (每个cell是特征向量列)

        Xtrain_seq = cell(size(Xtrain,1), 1);
for i = 1:size(Xtrain,1)
    Xtrain_seq{i} = Xtrain(i,:)';
end

Xtest_seq = cell(size(Xtest,1), 1);
for i = 1:size(Xtest,1)
    Xtest_seq{i} = Xtest(i,:)';
end

        % 创建多输出模型层
        layers = [
            sequenceInputLayer(numFeatures)
            gruLayer(bestParams.numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(outdim)
            regressionLayer
        ];
        
        options = trainingOptions('adam', ...
            'MaxEpochs', 200, ...
            'InitialLearnRate', bestParams.InitialLearnRate, ...
            'MiniBatchSize', 32, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', 0);
        
        % 训练模型
        net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
        
        % 预测
        Ypred = predict(net, Xtest_seq);
        
        % 计算各输出维度指标
        for od = 1:outdim
            m = calc_metrics(Ytest(:, od), Ypred(:, od));
            metrics_all(fold, :, od) = [m.R2, m.MSE, m.RMSE, m.MAPE];
        end
        
        % 打印单折指标
        fprintf('Fold %d:\n', fold);
        for od = 1:outdim
            fprintf(' Output Dim %d: R²=%.4f, MSE=%.4f, RMSE=%.4f, MAPE=%.2f%%\n', ...
                od, metrics_all(fold,1,od), metrics_all(fold,2,od), ...
                metrics_all(fold,3,od), metrics_all(fold,4,od));
        end
    end
    
    % 输出各输出维度的平均指标
    fprintf('\n[Final k-Fold Average Metrics per Output Dimension]\n');
    avg_metrics = squeeze(mean(metrics_all, 1));
    for od = 1:outdim
        fprintf('Output Dim %d: R²=%.4f | MSE=%.4f | RMSE=%.4f | MAPE=%.2f%%\n', ...
            od, avg_metrics(1,od), avg_metrics(2,od), avg_metrics(3,od), avg_metrics(4,od));
    end
    
    % 将最终模型和指标保存到handles结构（可选）
    handles.net = net;
    handles.metrics_all = metrics_all;
    handles.avg_metrics = avg_metrics;
    guidata(hObject, handles);
    
assignin('base', 'TrainedGRUModel', net);



% --- k折交叉验证目标函数（只针对单输出维度，供bayesopt使用） ---
function objective = kfold_GRU_cv(X, Y, optVars, k)
    cv = cvpartition(size(Y,1), 'KFold', k);
    R2_all = zeros(k,1);
    numFeatures = size(X,2);
    
    for fold = 1:k
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);
        
        Xtrain_raw = X(trainIdx, :);
        Xtest_raw = X(testIdx, :);
        Ytrain = Y(trainIdx);
        Ytest = Y(testIdx);
        
        mu = mean(Xtrain_raw);
        sigma = std(Xtrain_raw);
        sigma(sigma==0) = 1;
        Xtrain = (Xtrain_raw - mu) ./ sigma;
        Xtest = (Xtest_raw - mu) ./ sigma;
        
        Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
        Xtest_seq = squeeze(num2cell(Xtest', [1]));
        
        layers = [
            sequenceInputLayer(numFeatures)
            gruLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
            fullyConnectedLayer(1)
            regressionLayer
        ];
        
        options = trainingOptions('adam', ...
            'MaxEpochs', 100, ...
            'InitialLearnRate', optVars.InitialLearnRate, ...
            'MiniBatchSize', 32, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', 0);
        
        net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
        Ypred = predict(net, Xtest_seq);
        
        R2 = 1 - sum((Ytest - Ypred).^2) / sum((Ytest - mean(Ytest)).^2);
        R2_all(fold) = R2;
    end
    objective = -mean(R2_all);  % 负平均R2供bayesopt最小化



% --- 计算指标函数 ---
function metrics = calc_metrics(Ytrue, Ypred)
    metrics = struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
    );
