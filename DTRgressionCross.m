function varargout = DTRgressionCross(varargin)
% DTRGRESSIONCROSS MATLAB code for DTRgressionCross.fig
%      DTRGRESSIONCROSS, by itself, creates a new DTRGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = DTRGRESSIONCROSS returns the handle to a new DTRGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      DTRGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DTRGRESSIONCROSS.M with the given input arguments.
%
%      DTRGRESSIONCROSS('Property','Value',...) creates a new DTRGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DTRgressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DTRgressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DTRgressionCross

% Last Modified by GUIDE v2.5 05-Jun-2025 10:20:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DTRgressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @DTRgressionCross_OutputFcn, ...
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


% --- Executes just before DTRgressionCross is made visible.
function DTRgressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DTRgressionCross (see VARARGIN)

% Choose default command line output for DTRgressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DTRgressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DTRgressionCross_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    filename = get(handles.edit1, 'string');
    k = str2double(get(handles.edit2, 'string')); % K-fold
    outdim = str2double(get(handles.edit3, 'string')); % output dimensions

    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % shuffle

    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_ + 1:end);
    numOutputs = size(Y, 2);

    cv = cvpartition(size(Y, 1), 'KFold', k);
    all_metrics = cell(k, 1);  % 每一折的指标存储
    Ypred_all = cell(k, 1);    % 每一折的预测结果
    Ytest_all = cell(k, 1);    % 每一折的真实结果

    for i = 1:k
        trainIdx = training(cv, i);
        testIdx = test(cv, i);

        Xtrain_raw = X(trainIdx, :);
        Ytrain = Y(trainIdx, :);
        Xtest_raw = X(testIdx, :);
        Ytest = Y(testIdx, :);

        % 归一化
        mu = mean(Xtrain_raw);
        sigma = std(Xtrain_raw);
        sigma(sigma == 0) = 1;
        Xtrain = (Xtrain_raw - mu) ./ sigma;
        Xtest  = (Xtest_raw - mu) ./ sigma;

        % 初始化预测结果矩阵
        Ypred = zeros(size(Ytest));
        metrics_fold = struct('R2', [], 'MSE', [], 'RMSE', [], 'MAPE', []);

        fprintf('\nFold %d:\n', i);

        for j = 1:numOutputs
            y_train_j = Ytrain(:, j);
            rng(1); % 可重复性
            mdl_j = fitrtree(Xtrain, y_train_j, ...
                'OptimizeHyperparameters', {'MinLeafSize', 'MaxNumSplits', 'NumVariablesToSample'}, ...
                'HyperparameterOptimizationOptions', struct( ...
                    'AcquisitionFunctionName','expected-improvement-plus', ...
                    'Verbose', 0, ...
                    'ShowPlots', false));

            % 预测
            y_pred_j = predict(mdl_j, Xtest);
            Ypred(:, j) = y_pred_j;

            % 评估指标
            metrics = calc_metrics(Ytest(:, j), y_pred_j);
            metrics_fold.R2(j)   = metrics.R2;
            metrics_fold.MSE(j)  = metrics.MSE;
            metrics_fold.RMSE(j) = metrics.RMSE;
            metrics_fold.MAPE(j) = metrics.MAPE;

            % 打印当前输出维度结果
            fprintf('  Output %d: R² = %.4f, MSE = %.4f, RMSE = %.4f, MAPE = %.2f%%\n', ...
                j, metrics.R2, metrics.MSE, metrics.RMSE, metrics.MAPE);

            % 可视化
            figure;
            scatter(Ytest(:, j), y_pred_j, 'filled');
            xlabel(sprintf('Actual Output %d', j));
            ylabel(sprintf('Predicted Output %d', j));
            title(sprintf('Fold %d - Output %d - Decision Tree Regression', i, j));
            refline(1, 0); grid on;
        end

        % 保存当前折数据
        all_metrics{i} = metrics_fold;
        Ytest_all{i} = Ytest;
        Ypred_all{i} = Ypred;
        all_models = cell(k, numOutputs);
all_models{i, j} = mdl_j;
assignin('base', 'TrainedDTmodels', all_models);
    end

    % 平均指标统计
    fprintf('\n=== Average Performance Across %d Folds ===\n', k);
    for j = 1:numOutputs
        R2_all = cellfun(@(m) m.R2(j), all_metrics);
        MSE_all = cellfun(@(m) m.MSE(j), all_metrics);
        RMSE_all = cellfun(@(m) m.RMSE(j), all_metrics);
        MAPE_all = cellfun(@(m) m.MAPE(j), all_metrics);

        fprintf('Output %d: Avg R² = %.4f, MSE = %.4f, RMSE = %.4f, MAPE = %.2f%%\n', ...
            j, mean(R2_all), mean(MSE_all), mean(RMSE_all), mean(MAPE_all));
    end

    % 可选：将结果保存至工作区
    assignin('base', 'Ytest_all', Ytest_all);
    assignin('base', 'Ypred_all', Ypred_all);

%% --- Evaluation Metric Function ---
function metrics = calc_metrics(Ytrue, Ypred)
    metrics = struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
    );


% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


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
