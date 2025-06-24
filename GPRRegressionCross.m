function varargout = GPRRegressionCross(varargin)
% GPRREGRESSIONCROSS MATLAB code for GPRRegressionCross.fig
%      GPRREGRESSIONCROSS, by itself, creates a new GPRREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = GPRREGRESSIONCROSS returns the handle to a new GPRREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      GPRREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GPRREGRESSIONCROSS.M with the given input arguments.
%
%      GPRREGRESSIONCROSS('Property','Value',...) creates a new GPRREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GPRRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GPRRegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GPRRegressionCross

% Last Modified by GUIDE v2.5 05-Jun-2025 23:02:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GPRRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @GPRRegressionCross_OutputFcn, ...
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


% --- Executes just before GPRRegressionCross is made visible.
function GPRRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GPRRegressionCross (see VARARGIN)

% Choose default command line output for GPRRegressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GPRRegressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GPRRegressionCross_OutputFcn(hObject, eventdata, handles) 
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
    filename = get(handles.edit1, 'string');            % Excel 文件路径
    k = str2double(get(handles.edit3, 'string'));       % 交叉验证折数
    outdim = str2double(get(handles.edit2, 'string'));  % 输出维度数

    % === 读取并打乱数据 ===
    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % 打乱样本顺序

    % === 拆分特征和标签 ===
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);
    cv = cvpartition(num_samples, 'KFold', k);

    % === 初始化评估矩阵 ===
    R2s   = zeros(k, outdim);
    MSEs  = zeros(k, outdim);
    RMSEs = zeros(k, outdim);
    MAPEs = zeros(k, outdim);

    % === 交叉验证 ===
    for i = 1:k
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);

        Xtrain_raw = X(trainIdx, :);
        Ytrain     = Y(trainIdx, :);
        Xtest_raw  = X(testIdx, :);
        Ytest      = Y(testIdx, :);

        % --- 特征归一化 ---
        mu = mean(Xtrain_raw);
        sigma = std(Xtrain_raw);
        sigma(sigma == 0) = 1;
        Xtrain = (Xtrain_raw - mu) ./ sigma;
        Xtest  = (Xtest_raw - mu) ./ sigma;

        % --- 针对每个输出维度单独训练模型 ---
        Ypred = zeros(size(Ytest));  % 预测矩阵初始化
        for d = 1:outdim
            rng(1);  % 重现性
            Mdl = fitrgp(Xtrain, Ytrain(:, d), ...
                'Standardize', false, ...
                'KernelFunction', 'squaredexponential', ...
                'OptimizeHyperparameters', {'KernelScale', 'Sigma'}, ...
                'HyperparameterOptimizationOptions', struct( ...
                    'AcquisitionFunctionName', 'expected-improvement-plus', ...
                    'Verbose', 0 ...
                ));

            % --- 预测 ---
            Ypred(:, d) = predict(Mdl, Xtest);

            % --- 计算指标 ---
            Ytrue = Ytest(:, d);
            R2s(i, d)   = 1 - sum((Ytrue - Ypred(:, d)).^2) / sum((Ytrue - mean(Ytrue)).^2);
            MSEs(i, d)  = mean((Ytrue - Ypred(:, d)).^2);
            RMSEs(i, d) = sqrt(MSEs(i, d));
            MAPEs(i, d) = mean(abs((Ytrue - Ypred(:, d)) ./ (Ytrue + eps))) * 100;
        end

        % --- 可视化当前折 ---
        figure;
        for d = 1:outdim
            subplot(1, outdim, d);
            scatter(Ytest(:, d), Ypred(:, d), 'filled');
            xlabel('Actual'); ylabel('Predicted');
            title(sprintf('Fold %d - Output %d', i, d));
            grid on; refline(1, 0);
        end
    end

    % === 平均指标 ===
    fprintf('\n[Cross-Validation (%d-Fold) Results per Output Dimension]\n', k);
    for d = 1:outdim
        fprintf('--- Output Dimension %d ---\n', d);
        fprintf('Avg R²   = %.4f\n', mean(R2s(:, d)));
        fprintf('Avg MSE  = %.4f\n', mean(MSEs(:, d)));
        fprintf('Avg RMSE = %.4f\n', mean(RMSEs(:, d)));
        fprintf('Avg MAPE = %.2f%%\n\n', mean(MAPEs(:, d)));
    end

    % === 保存指标结果到工作区 ===
    assignin('base', 'TrainedGPRModel', Mdl);



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
