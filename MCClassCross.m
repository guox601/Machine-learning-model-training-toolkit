function varargout = MCClassCross(varargin)
% MCCLASSCROSS MATLAB code for MCClassCross.fig
%      MCCLASSCROSS, by itself, creates a new MCCLASSCROSS or raises the existing
%      singleton*.
%
%      H = MCCLASSCROSS returns the handle to a new MCCLASSCROSS or the handle to
%      the existing singleton*.
%
%      MCCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MCCLASSCROSS.M with the given input arguments.
%
%      MCCLASSCROSS('Property','Value',...) creates a new MCCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MCClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MCClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MCClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 15:48:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MCClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @MCClassCross_OutputFcn, ...
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


% --- Executes just before MCClassCross is made visible.
function MCClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MCClassCross (see VARARGIN)

% Choose default command line output for MCClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MCClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MCClassCross_OutputFcn(hObject, eventdata, handles) 
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
% === Encode labels as integers ===
Y = grp2idx(Y);

% === Parameters ===
cv = cvpartition(size(Y, 1), 'KFold', kfold);

% === Initialize storage for metrics ===
acc_all = zeros(kfold, 1);
prec_all = zeros(kfold, 1);
rec_all = zeros(kfold, 1);
f1_all = zeros(kfold, 1);

Y_all_true = [];
Y_all_pred = [];

for fold = 1:kfold
    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    Xtrain_raw = X(trainIdx, :);
    Ytrain = Y(trainIdx);
    Xtest_raw  = X(testIdx, :);
    Ytest = Y(testIdx);

    % --- Z-score normalization ---
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;

    % --- 用fitcecoc训练多分类线性模型 ---
    t = templateLinear('Learner','logistic'); % 线性logistic回归作为基分类器
    Mdl = fitcecoc(Xtrain, Ytrain, 'Learners', t);

    % --- 预测 ---
    Ypred = predict(Mdl, Xtest);

    % --- 计算指标 ---
    [acc, prec, rec, f1] = classification_metrics(Ytest, Ypred);

    acc_all(fold) = acc;
    prec_all(fold) = prec;
    rec_all(fold) = rec;
    f1_all(fold) = f1;

    Y_all_true = [Y_all_true; Ytest];
    Y_all_pred = [Y_all_pred; Ypred];

    fprintf('[Fold %d] Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
        fold, acc, prec, rec, f1);
end

% === 平均指标（所有折平均）===
% fprintf('\n=== Average Metrics (mean of folds) ===\n');
% fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
%     mean(acc_all), mean(prec_all), mean(rec_all), mean(f1_all));

% === 综合指标（所有折合并）===
[acc_tot, prec_tot, rec_tot, f1_tot] = classification_metrics(Y_all_true, Y_all_pred);
fprintf('\n=== Overall Metrics (all folds combined) ===\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    acc_tot, prec_tot, rec_tot, f1_tot);

% === 画最终混淆矩阵 ===
figure;
cm = confusionmat(Y_all_true, Y_all_pred);
confusionchart(cm);
title('Final Confusion Matrix (All folds combined)');
assignin('base', 'TrainedMultilinearClassificationModel', Mdl);
%% --- 计算分类指标函数 ---
function [accuracy, precision, recall, f1] = classification_metrics(Ytrue, Ypred)
    cm = confusionmat(Ytrue, Ypred);
    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2) - TP;

    precision_per_class = TP ./ (TP + FP + eps);
    recall_per_class = TP ./ (TP + FN + eps);

    precision = mean(precision_per_class);
    recall = mean(recall_per_class);
    f1 = 2 * precision * recall / (precision + recall + eps);
    accuracy = sum(TP) / sum(cm(:));


% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
