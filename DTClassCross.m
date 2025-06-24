function varargout = DTClassCross(varargin)
% DTCLASSCROSS MATLAB code for DTClassCross.fig
%      DTCLASSCROSS, by itself, creates a new DTCLASSCROSS or raises the existing
%      singleton*.
%
%      H = DTCLASSCROSS returns the handle to a new DTCLASSCROSS or the handle to
%      the existing singleton*.
%
%      DTCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DTCLASSCROSS.M with the given input arguments.
%
%      DTCLASSCROSS('Property','Value',...) creates a new DTCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DTClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DTClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DTClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 15:44:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DTClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @DTClassCross_OutputFcn, ...
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


% --- Executes just before DTClassCross is made visible.
function DTClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DTClassCross (see VARARGIN)

% Choose default command line output for DTClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DTClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DTClassCross_OutputFcn(hObject, eventdata, handles) 
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
Y = categorical(Y);  % 转换为分类标签

% --- k-Fold Cross-Validation ---
cv = cvpartition(Y, 'KFold', k);
all_metrics = struct('Accuracy', [], 'Precision', [], 'Recall', [], 'F1', []);

% --- Loop over each fold ---
for i = 1:k
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    Xtrain_raw = X(trainIdx, :);
    Ytrain = Y(trainIdx);
    Xtest_raw = X(testIdx, :);
    Ytest = Y(testIdx);

    % Normalize (z-score based on training set)
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;

    % --- Train Decision Tree Classifier with Hyperparameter Optimization ---
    rng(1);
    Mdl = fitctree(Xtrain, Ytrain, ...
        'OptimizeHyperparameters', {'MinLeafSize', 'MaxNumSplits', 'SplitCriterion'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'Verbose', 0, ...
            'ShowPlots', false));

    % --- Predict ---
    Ypred = predict(Mdl, Xtest);

    % --- Evaluate Classification Metrics ---
    [acc, prec, rec, f1] = classification_metrics(Ytest, Ypred);
    all_metrics.Accuracy(i)  = acc;
    all_metrics.Precision(i) = prec;
    all_metrics.Recall(i)    = rec;
    all_metrics.F1(i)        = f1;

    fprintf('Fold %d - Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n', ...
        i, acc, prec, rec, f1);

    % 
    figure;
    confusionchart(Ytest, Ypred);
    title(sprintf('Fold %d - Confusion Matrix', i));
end

% --- Display Average Performance ---
fprintf('\n=== k-Fold Cross-Validation Results (k = %d) ===\n', k);
fprintf('Average Accuracy : %.4f\n', mean(all_metrics.Accuracy));
fprintf('Average Precision: %.4f\n', mean(all_metrics.Precision));
fprintf('Average Recall   : %.4f\n', mean(all_metrics.Recall));
fprintf('Average F1 Score : %.4f\n', mean(all_metrics.F1));

assignin('base', 'TrainedDTModel', Mdl);

%% --- Classification Metrics Function ---
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
