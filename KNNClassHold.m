function varargout = KNNClassHold(varargin)
% KNNCLASSHOLD MATLAB code for KNNClassHold.fig
%      KNNCLASSHOLD, by itself, creates a new KNNCLASSHOLD or raises the existing
%      singleton*.
%
%      H = KNNCLASSHOLD returns the handle to a new KNNCLASSHOLD or the handle to
%      the existing singleton*.
%
%      KNNCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in KNNCLASSHOLD.M with the given input arguments.
%
%      KNNCLASSHOLD('Property','Value',...) creates a new KNNCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before KNNClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to KNNClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help KNNClassHold

% Last Modified by GUIDE v2.5 03-Jun-2025 23:31:55

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @KNNClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @KNNClassHold_OutputFcn, ...
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


% --- Executes just before KNNClassHold is made visible.
function KNNClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to KNNClassHold (see VARARGIN)

% Choose default command line output for KNNClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes KNNClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = KNNClassHold_OutputFcn(hObject, eventdata, handles) 
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
    proportion = str2double(get(handles.edit2, 'string'));
    outdim = 1; 

    res = xlsread(filename);

    num_features = size(res, 2) - outdim;
    X = res(:, 1:num_features);
    Y = res(:, num_features+1:end);
    Y = categorical(Y); 
% Split training and test sets
cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

% Normalize features (z-score normalization)
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1;
Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest  = (Xtest_raw - mu) ./ sigma;
Xall   = (X - mu) ./ sigma;

% Train Linear Discriminant Analysis (LDA) model
Mdl = fitcdiscr(Xtrain, Ytrain);

% Predict on training, test, and all data
Ytrain_pred = predict(Mdl, Xtrain);
Ytest_pred  = predict(Mdl, Xtest);
Yall_pred   = predict(Mdl, Xall);

% Calculate classification accuracy
calc_acc = @(Ytrue, Ypred) mean(Ytrue == Ypred) * 100;

train_acc = calc_acc(Ytrain, Ytrain_pred);
test_acc  = calc_acc(Ytest, Ytest_pred);
all_acc   = calc_acc(Y, Yall_pred);

% Display accuracies
fprintf('\n[Training Set Accuracy] %.2f%%\n', train_acc);
fprintf('[Test Set Accuracy] %.2f%%\n', test_acc);
fprintf('[All Data Accuracy] %.2f%%\n', all_acc);

% Plot confusion matrix for test set
figure;
confusionchart(Ytest, Ytest_pred);
title('Confusion Matrix (Test Set)');
% Calculate and print metrics
[precision_train, recall_train, f1_train] = compute_metrics(Ytrain, Ytrain_pred);
[precision_test, recall_test, f1_test] = compute_metrics(Ytest, Ytest_pred);
[precision_all, recall_all, f1_all] = compute_metrics(Y, Yall_pred);

fprintf('\n[Training Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    train_acc/100, precision_train, recall_train, f1_train);

fprintf('[Test Set Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    test_acc/100, precision_test, recall_test, f1_test);

fprintf('[All Data Evaluation]\n');
fprintf('Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
    all_acc/100, precision_all, recall_all, f1_all);
    
    assignin('base', 'TrainedKNNModel', Mdl);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest_pred', Ytest_pred);
    assignin('base', 'Yall_pred', Yall_pred);
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Function to compute precision, recall, and F1 for multi-class
function [precision, recall, f1] = compute_metrics(Ytrue, Ypred)
    confMat = confusionmat(Ytrue, Ypred);
    TP = diag(confMat);
    FP = sum(confMat, 1)' - TP;
    FN = sum(confMat, 2) - TP;
    precision_class = TP ./ (TP + FP);
    recall_class = TP ./ (TP + FN);
    f1_class = 2 * (precision_class .* recall_class) ./ (precision_class + recall_class);

    % Handle NaNs
    precision_class(isnan(precision_class)) = 0;
    recall_class(isnan(recall_class)) = 0;
    f1_class(isnan(f1_class)) = 0;

    precision = mean(precision_class);
    recall = mean(recall_class);
    f1 = mean(f1_class);
