function varargout = RFClassHold(varargin)
% RFCLASSHOLD MATLAB code for RFClassHold.fig
%      RFCLASSHOLD, by itself, creates a new RFCLASSHOLD or raises the existing
%      singleton*.
%
%      H = RFCLASSHOLD returns the handle to a new RFCLASSHOLD or the handle to
%      the existing singleton*.
%
%      RFCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RFCLASSHOLD.M with the given input arguments.
%
%      RFCLASSHOLD('Property','Value',...) creates a new RFCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RFClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RFClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RFClassHold

% Last Modified by GUIDE v2.5 03-Jun-2025 21:58:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RFClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @RFClassHold_OutputFcn, ...
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


% --- Executes just before RFClassHold is made visible.
function RFClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RFClassHold (see VARARGIN)

% Choose default command line output for RFClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RFClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = RFClassHold_OutputFcn(hObject, eventdata, handles) 
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
filename = get(handles.edit1, 'string');
outdim = 1;
proportion = str2double(get(handles.edit2, 'string'));

% 读取数据
res = xlsread(filename);
num_samples = size(res, 1);
f_ = size(res, 2) - outdim;
X = res(:, 1:f_);
Y = res(:, f_ + 1:end);
Y = categorical(Y);

% 划分训练/测试集
cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

% 数据标准化
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1;
Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest  = (Xtest_raw - mu) ./ sigma;
Xall   = (X - mu) ./ sigma;

% 训练随机森林分类模型（带超参数优化）
rng(1);
Mdl = fitcensemble(Xtrain, Ytrain, ...
    'Method', 'Bag', ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits'}, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName','expected-improvement-plus', ...
                                                'ShowPlots', true, ...
                                                'Verbose', 1));

% 预测
Ytrain_pred = predict(Mdl, Xtrain);
Ytest_pred  = predict(Mdl, Xtest);
Yall_pred   = predict(Mdl, Xall);

% 输出每组数据的指标
printMetrics(Ytrain, Ytrain_pred, 'Training Set');
printMetrics(Ytest, Ytest_pred, 'Test Set');
printMetrics(Y, Yall_pred, 'All Data');

% 混淆矩阵
figure;
confusionchart(Ytest, Ytest_pred);
title('Confusion Matrix (Test Set)');

% 保存到工作区
assignin('base', 'TrainedRFModel', Mdl);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest_pred', Ytest_pred);
assignin('base', 'Yall_pred', Yall_pred);


function printMetrics(Ytrue, Ypred, setName)
% 计算并打印分类性能指标
[precision, recall, f1, accuracy] = classificationMetrics(Ytrue, Ypred);
avg_precision = mean(precision);
avg_recall = mean(recall);
avg_f1 = mean(f1);
acc = accuracy / 100;

fprintf('\n[%s Evaluation]\n', setName);
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', acc, avg_precision, avg_recall, avg_f1);

function [precision, recall, f1, accuracy] = classificationMetrics(Ytrue, Ypred)
classes = categories(Ytrue);
nClasses = numel(classes);
precision = zeros(nClasses, 1);
recall = zeros(nClasses, 1);
f1 = zeros(nClasses, 1);

for i = 1:nClasses
    cls = classes{i};
    TP = sum(Ypred == cls & Ytrue == cls);
    FP = sum(Ypred == cls & Ytrue ~= cls);
    FN = sum(Ypred ~= cls & Ytrue == cls);
    
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
end

accuracy = sum(Ytrue == Ypred) / numel(Ytrue) * 100;
