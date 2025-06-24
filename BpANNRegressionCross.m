function varargout = BpANNRegressionCross(varargin)
% BPANNREGRESSIONCROSS MATLAB code for BpANNRegressionCross.fig
%      BPANNREGRESSIONCROSS, by itself, creates a new BPANNREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = BPANNREGRESSIONCROSS returns the handle to a new BPANNREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      BPANNREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BPANNREGRESSIONCROSS.M with the given input arguments.
%
%      BPANNREGRESSIONCROSS('Property','Value',...) creates a new BPANNREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BpANNRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BpANNRegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BpANNRegressionCross

% Last Modified by GUIDE v2.5 05-Jun-2025 21:58:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BpANNRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @BpANNRegressionCross_OutputFcn, ...
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


% --- Executes just before BpANNRegressionCross is made visible.
function BpANNRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BpANNRegressionCross (see VARARGIN)

% Choose default command line output for BpANNRegressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BpANNRegressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BpANNRegressionCross_OutputFcn(hObject, eventdata, handles) 
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
    % === 读取输入参数 ===
    filename = get(handles.edit1, 'string');            % Excel 文件路径
    kfold = str2double(get(handles.edit3, 'string'));   % 交叉验证折数
    outdim = str2double(get(handles.edit2, 'string'));  % 输出维度数

    % === 读取并打乱数据 ===
    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % 打乱样本顺序

    % === 拆分特征和标签 ===
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);

    % === 标准化特征数据 ===
    mu = mean(X);
    sigma = std(X);
    sigma(sigma == 0) = 1;  % 防止除以零
    Xnorm = (X - mu) ./ sigma;
    numFeatures = f_;

    % === 定义 bayesopt 的目标函数 ===
    objFcn = @(params) kFoldLossANN(params, Xnorm, Y, numFeatures, kfold);

    % === 设置待优化的超参数空间 ===
    optVars = [
        optimizableVariable('NumHiddenUnits', [10, 200], 'Type', 'integer')
        optimizableVariable('InitialLearnRate', [1e-4, 1e-2], 'Transform', 'log')
        optimizableVariable('MaxEpochs', [50, 300], 'Type', 'integer')
        optimizableVariable('MiniBatchSize', [8, 64], 'Type', 'integer')
    ];

    % === 执行贝叶斯优化 ===
    results = bayesopt(objFcn, optVars, ...
        'MaxObjectiveEvaluations', 20, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1, ...
        'UseParallel', false);

    bestParams = results.XAtMinObjective;

    % === 使用最佳超参数进行 k-fold 训练并记录指标 ===
    k = kfold;
    cv = cvpartition(size(Y,1), 'KFold', k);
    allMetrics.R2 = zeros(k, outdim);
    allMetrics.RMSE = zeros(k, outdim);
    allMetrics.MAPE = zeros(k, outdim);
    trainedModels = cell(k,1);  % 保存每一折的模型

    for i = 1:k
        Xtrain = Xnorm(training(cv,i), :);
        Ytrain = Y(training(cv,i), :);
        Xval = Xnorm(test(cv,i), :);
        Yval = Y(test(cv,i), :);

        layers = [
            featureInputLayer(numFeatures, 'Normalization', 'none')
            fullyConnectedLayer(bestParams.NumHiddenUnits)
            reluLayer
            fullyConnectedLayer(outdim)
            regressionLayer
        ];

        options = trainingOptions('adam', ...
            'MaxEpochs', bestParams.MaxEpochs, ...
            'InitialLearnRate', bestParams.InitialLearnRate, ...
            'MiniBatchSize', bestParams.MiniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'Plots', 'none');

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        trainedModels{i} = net;  % 保存当前模型

        Yval_pred = predict(net, Xval);
        metrics = calc_metrics(Yval, Yval_pred);

        allMetrics.R2(i,:) = metrics.R2;
        allMetrics.RMSE(i,:) = metrics.RMSE;
        allMetrics.MAPE(i,:) = metrics.MAPE;

        for d = 1:outdim
            fprintf('Fold %d - Output %d: R² = %.4f, RMSE = %.4f, MAPE = %.2f%%\n', ...
                i, d, metrics.R2(d), metrics.RMSE(d), metrics.MAPE(d));
        end
    end

    % === 输出平均性能指标 ===
    fprintf('\n=== Final Model Performance (k-Fold = %d) ===\n', k);
    for d = 1:outdim
        fprintf('Output %d - Avg R²: %.4f, Avg RMSE: %.4f, Avg MAPE: %.2f%%\n', ...
            d, mean(allMetrics.R2(:,d)), mean(allMetrics.RMSE(:,d)), mean(allMetrics.MAPE(:,d)));
    end

    % === 保存变量到工作区 ===
    assignin('base', 'TrainedBpANNModels', trainedModels);
    assignin('base', 'BpANNMetrics', allMetrics);
    assignin('base', 'BpANNBestParams', bestParams);
    assignin('base', 'BpANNNormalization', struct('mu', mu, 'sigma', sigma));

%% --- Evaluation Function for bayesopt ---
function avgRMSE = kFoldLossANN(params, X, Y, numFeatures, k)
    cv = cvpartition(size(Y,1), 'KFold', k);
    rmse_all = zeros(k,1);
    
    for i = 1:k
        Xtrain = X(training(cv,i), :);
        Ytrain = Y(training(cv,i), :);
        Xval = X(test(cv,i), :);
        Yval = Y(test(cv,i), :);

        layers = [
            featureInputLayer(numFeatures, 'Normalization', 'none')
            fullyConnectedLayer(params.NumHiddenUnits)
            reluLayer
            fullyConnectedLayer(size(Ytrain, 2))
            regressionLayer
        ];

        options = trainingOptions('adam', ...
            'MaxEpochs', params.MaxEpochs, ...
            'InitialLearnRate', params.InitialLearnRate, ...
            'MiniBatchSize', params.MiniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'Plots', 'none');

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Ypred = predict(net, Xval);
        
        % 多维输出情况下的平均 RMSE 作为目标值
        rmse_all(i) = mean(sqrt(mean((Yval - Ypred).^2, 1)));
    end
    
    avgRMSE = mean(rmse_all);  % 平均 k-fold 的 RMSE


%% --- Metrics Calculation Function ---
function metrics = calc_metrics(Ytrue, Ypred)
    n_outputs = size(Ytrue, 2);
    R2 = zeros(1, n_outputs);
    RMSE = zeros(1, n_outputs);
    MAPE = zeros(1, n_outputs);

    for i = 1:n_outputs
        y_true = Ytrue(:, i);
        y_pred = Ypred(:, i);
        SS_res = sum((y_true - y_pred).^2);
        SS_tot = sum((y_true - mean(y_true)).^2);
        R2(i) = 1 - SS_res / (SS_tot + eps);
        RMSE(i) = sqrt(mean((y_true - y_pred).^2));
        MAPE(i) = mean(abs((y_true - y_pred) ./ (y_true + eps))) * 100;
    end

    metrics = struct('R2', R2, 'RMSE', RMSE, 'MAPE', MAPE);



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
