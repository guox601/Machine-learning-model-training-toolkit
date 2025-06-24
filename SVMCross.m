function varargout = SVMCross(varargin)
% SVMCROSS MATLAB code for SVMCross.fig
%      SVMCROSS, by itself, creates a new SVMCROSS or raises the existing
%      singleton*.
%
%      H = SVMCROSS returns the handle to a new SVMCROSS or the handle to
%      the existing singleton*.
%
%      SVMCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SVMCROSS.M with the given input arguments.
%
%      SVMCROSS('Property','Value',...) creates a new SVMCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SVMCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SVMCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SVMCross

% Last Modified by GUIDE v2.5 06-Jun-2025 15:35:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SVMCross_OpeningFcn, ...
                   'gui_OutputFcn',  @SVMCross_OutputFcn, ...
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


% --- Executes just before SVMCross is made visible.
function SVMCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SVMCross (see VARARGIN)

% Choose default command line output for SVMCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SVMCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SVMCross_OutputFcn(hObject, eventdata, handles) 
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
% Convert Y to categorical if numeric
Y = Y(:);  
if isnumeric(Y)
    Y = categorical(Y);
end

% === K-fold cross-validation ===

cv = cvpartition(Y, 'KFold', K);
% Initialize metrics storage
accuracy_all = zeros(K,1);
precision_all = zeros(K,1);
recall_all = zeros(K,1);
f1_all = zeros(K,1);

% Store all true and predicted labels
Y_all_true = [];
Y_all_pred = [];

for fold = 1:K
    fprintf('=== Fold %d/%d ===\n', fold, K);

    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    Xtrain = X(trainIdx, :);
    Ytrain = Y(trainIdx);
    Xtest = X(testIdx, :);
    Ytest = Y(testIdx);

    % Normalize based on training set
    mu = mean(Xtrain);
    sigma = std(Xtrain);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain - mu) ./ sigma;
    Xtest = (Xtest - mu) ./ sigma;

    % Train multi-class SVM with hyperparameter optimization
    t = templateSVM('KernelFunction', 'rbf');
    Mdl = fitcecoc(Xtrain, Ytrain, ...
        'Learners', t, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'ShowPlots', false, ...
            'Verbose', 0, ...
            'MaxObjectiveEvaluations', 30));

    % Predict
    Ypred = predict(Mdl, Xtest);

    % Accumulate results
    Y_all_true = [Y_all_true; Ytest];
    Y_all_pred = [Y_all_pred; Ypred];

    % Compute confusion matrix and metrics
    confMat = confusionmat(Ytest, Ypred);
    TP = diag(confMat);
    FP = sum(confMat,1)' - TP;
    FN = sum(confMat,2) - TP;

    precision = mean(TP ./ max(TP + FP, 1));
    recall = mean(TP ./ max(TP + FN, 1));
    f1 = mean(2 * (precision .* recall) ./ max(precision + recall, 1));
    accuracy = sum(Ytest == Ypred) / numel(Ytest);

    % Save metrics
    accuracy_all(fold) = accuracy;
    precision_all(fold) = precision;
    recall_all(fold) = recall;
    f1_all(fold) = f1;

    fprintf('Fold %d Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1-score: %.4f\n', ...
        fold, accuracy, precision, recall, f1);
end

% Overall metrics
confMat_total = confusionmat(Y_all_true, Y_all_pred);
TP_all = diag(confMat_total);
FP_all = sum(confMat_total,1)' - TP_all;
FN_all = sum(confMat_total,2) - TP_all;

precision_all_total = mean(TP_all ./ max(TP_all + FP_all, 1));
recall_all_total = mean(TP_all ./ max(TP_all + FN_all, 1));
f1_all_total = mean(2 * (precision_all_total .* recall_all_total) ./ max(precision_all_total + recall_all_total, 1));
accuracy_all_total = sum(Y_all_true == Y_all_pred) / numel(Y_all_true);

fprintf('\n=== Overall Cross-Validation Performance ===\n');
fprintf('Accuracy: %.4f\nPrecision: %.4f\nRecall: %.4f\nF1-score: %.4f\n', ...
    accuracy_all_total, precision_all_total, recall_all_total, f1_all_total);

% Plot overall confusion matrix
figure;
confusionchart(confMat_total);
title('Overall Confusion Matrix');
assignin('base', 'TrainedSVMModel', Mdl);
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
