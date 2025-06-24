function varargout = RFClassCross(varargin)
% RFCLASSCROSS MATLAB code for RFClassCross.fig
%      RFCLASSCROSS, by itself, creates a new RFCLASSCROSS or raises the existing
%      singleton*.
%
%      H = RFCLASSCROSS returns the handle to a new RFCLASSCROSS or the handle to
%      the existing singleton*.
%
%      RFCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RFCLASSCROSS.M with the given input arguments.
%
%      RFCLASSCROSS('Property','Value',...) creates a new RFCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RFClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RFClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RFClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 14:09:51

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RFClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @RFClassCross_OutputFcn, ...
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


% --- Executes just before RFClassCross is made visible.
function RFClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RFClassCross (see VARARGIN)

% Choose default command line output for RFClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RFClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = RFClassCross_OutputFcn(hObject, eventdata, handles) 
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
k = str2double(get(handles.edit3, 'string'));   % 交叉验证折数
outdim = 1;

% === 读取并打乱数据 ===
res = xlsread(filename);
num_samples = size(res, 1);
res = res(randperm(num_samples), :);  % 打乱样本顺序
f_ = size(res, 2) - outdim;
X = res(:, 1:f_);
Y = res(:, f_ + 1:end);

% Convert labels to categorical for classification
Y = categorical(Y);

% Number of folds
cv = cvpartition(num_samples, 'KFold', k);

% Preallocate metrics
accuracy_fold = zeros(k, 1);
precision_fold = zeros(k, 1);
recall_fold = zeros(k, 1);
f1_fold = zeros(k, 1);

% Store all predictions for overall metrics
Y_all_true = [];
Y_all_pred = [];

classes = categories(Y);

for i = 1:k
    % Training and test indices
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    Xtrain_raw = X(trainIdx, :);
    Ytrain = Y(trainIdx, :);
    Xtest_raw = X(testIdx, :);
    Ytest = Y(testIdx, :);
    
    % Normalize features (Z-score using train stats)
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    
    % Train classification ensemble (Random Forest)
    rng(1); % reproducibility
    
    Mdl = fitcensemble(Xtrain, Ytrain, ...
        'Method', 'Bag', ...
        'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'Verbose', 0 ...
        ));
    
    % Predict class labels
    Ypred = predict(Mdl, Xtest);
    
    % Store all predictions and true labels
    Y_all_true = [Y_all_true; Ytest];
    Y_all_pred = [Y_all_pred; Ypred];
    
    % Calculate metrics for this fold
    [acc, prec, rec, f1] = classification_metrics(Ytest, Ypred, classes);
    
    accuracy_fold(i) = acc;
    precision_fold(i) = prec;
    recall_fold(i) = rec;
    f1_fold(i) = f1;
    
    fprintf('[Fold %d] Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
        i, acc, prec, rec, f1);
end

% Overall metrics
[acc_all, prec_all, rec_all, f1_all] = classification_metrics(Y_all_true, Y_all_pred, classes);

fprintf('\n=== Overall Metrics ===\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', acc_all, prec_all, rec_all, f1_all);

% Plot final confusion matrix
figure;
cm = confusionmat(Y_all_true, Y_all_pred);
confusionchart(cm, classes);
title('Final Confusion Matrix (All folds combined)');
assignin('base', 'TrainedRFModel', Mdl);

%% Helper function to calculate classification metrics
function [accuracy, precision, recall, f1] = classification_metrics(Ytrue, Ypred, classes)
    cm = confusionmat(Ytrue, Ypred);
    TP = diag(cm);
    FP = sum(cm, 1)' - TP;
    FN = sum(cm, 2) - TP;

    precision_per_class = TP ./ (TP + FP + eps);
    recall_per_class = TP ./ (TP + FN + eps);

    precision = mean(precision_per_class);
    recall = mean(recall_per_class);
    f1 = 2 * precision * recall / (precision + recall + eps);
    accuracy = sum(TP) / sum(cm(:));



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
