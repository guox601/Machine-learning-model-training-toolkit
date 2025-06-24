function varargout = SVMHold(varargin)
% SVMHOLD MATLAB code for SVMHold.fig
%      SVMHOLD, by itself, creates a new SVMHOLD or raises the existing
%      singleton*.
%
%      H = SVMHOLD returns the handle to a new SVMHOLD or the handle to
%      the existing singleton*.
%
%      SVMHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SVMHOLD.M with the given input arguments.
%
%      SVMHOLD('Property','Value',...) creates a new SVMHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SVMHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SVMHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SVMHold

% Last Modified by GUIDE v2.5 03-Jun-2025 23:31:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SVMHold_OpeningFcn, ...
                   'gui_OutputFcn',  @SVMHold_OutputFcn, ...
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


% --- Executes just before SVMHold is made visible.
function SVMHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SVMHold (see VARARGIN)

% Choose default command line output for SVMHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SVMHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SVMHold_OutputFcn(hObject, eventdata, handles) 
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


function pushbutton1_Callback(hObject, eventdata, handles)
filename = get(handles.edit1, 'string');
outdim = 1;
proportion = str2double(get(handles.edit2, 'string'));

% Read data from Excel
res = xlsread(filename);
num_samples = size(res, 1);
f_ = size(res, 2) - outdim;
X = res(:, 1:f_);
Y = res(:, f_ + 1:end);
Y = categorical(Y);

% ------------------【Split Training and Test Sets】------------------
cv = cvpartition(size(Y, 1), 'HoldOut', 1 - proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

% ------------------【Feature Normalization】------------------
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1;  % Avoid division by zero
Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest  = (Xtest_raw - mu) ./ sigma;
Xall   = (X - mu) ./ sigma;

% ------------------【Train Multi-class SVM Model with Hyperparameter Optimization】------------------
rng(1);  % For reproducibility
Mdl = fitcecoc(Xtrain, Ytrain, ...
    'Learners', templateSVM('KernelFunction', 'rbf', 'Standardize', false), ...
    'Coding', 'onevsall', ...
    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ShowPlots', true, ...
        'Verbose', 1 ...
    ));

% ------------------【Predict】------------------
Ytrain_pred = predict(Mdl, Xtrain);
Ytest_pred  = predict(Mdl, Xtest);
Yall_pred   = predict(Mdl, Xall);

% ------------------【Evaluation Metrics with Precision, Recall, F1】------------------
metrics_train = calc_class_metrics(Ytrain, Ytrain_pred);
metrics_test = calc_class_metrics(Ytest, Ytest_pred);
metrics_all = calc_class_metrics(Y, Yall_pred);

% Save trained model to workspace
assignin('base', 'TrainedSVMModel', Mdl);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest_pred', Ytest_pred);
assignin('base', 'Yall_pred', Yall_pred);

% ------------------【Print metrics in requested format】------------------
fprintf('\n[Training Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    metrics_train.Accuracy, metrics_train.Precision, metrics_train.Recall, metrics_train.F1);

fprintf('\n[Test Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    metrics_test.Accuracy, metrics_test.Precision, metrics_test.Recall, metrics_test.F1);

fprintf('\n[All Data Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    metrics_all.Accuracy, metrics_all.Precision, metrics_all.Recall, metrics_all.F1);

% ------------------【Plot Confusion Matrix】------------------
figure;
confusionchart(Ytest, Ytest_pred);
title('Confusion Matrix - Test Set');


% ===== Nested function to calculate evaluation metrics =====
function metrics = calc_class_metrics(Ytrue, Ypred)
    C = confusionmat(Ytrue, Ypred);
    TP = diag(C);
    FP = sum(C,1)' - TP;
    FN = sum(C,2) - TP;

    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);
    f1 = 2 * (precision .* recall) ./ (precision + recall);

    % Handle NaN when denominator is zero
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1(isnan(f1)) = 0;

    % Macro average
    metrics.Accuracy = mean(Ytrue == Ypred);
    metrics.Precision = mean(precision);
    metrics.Recall = mean(recall);
    metrics.F1 = mean(f1);






    
