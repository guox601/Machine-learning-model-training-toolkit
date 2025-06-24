function varargout = DTClassHold(varargin)
% DTCLASSHOLD MATLAB code for DTClassHold.fig
%      DTCLASSHOLD, by itself, creates a new DTCLASSHOLD or raises the existing
%      singleton*.
%
%      H = DTCLASSHOLD returns the handle to a new DTCLASSHOLD or the handle to
%      the existing singleton*.
%
%      DTCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DTCLASSHOLD.M with the given input arguments.
%
%      DTCLASSHOLD('Property','Value',...) creates a new DTCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DTClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DTClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Last Modified by GUIDE v2.5 03-Jun-2025 23:31:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DTClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @DTClassHold_OutputFcn, ...
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


% --- Executes just before DTClassHold is made visible.
function DTClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for DTClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DTClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DTClassHold_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;


function edit1_Callback(hObject, eventdata, handles)
% Callback for edit1


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit2_Callback(hObject, eventdata, handles)
% Callback for edit2


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
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

    cv = cvpartition(size(Y,1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1; 

    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    Xall = (X - mu) ./ sigma;

    rng(1); 
    Mdl = fitctree(Xtrain, Ytrain, ...
        'OptimizeHyperparameters', {'MinLeafSize', 'MaxNumSplits', 'NumVariablesToSample'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'ShowPlots', true, ...
            'Verbose', 1));
    Ytrain_pred = predict(Mdl, Xtrain);
    Ytest_pred = predict(Mdl, Xtest);
    Yall_pred = predict(Mdl, Xall);

    calc_acc = @(Ytrue, Ypred) sum(Ytrue == Ypred) / numel(Ytrue);

    train_acc = calc_acc(Ytrain, Ytrain_pred);
    test_acc = calc_acc(Ytest, Ytest_pred);
    all_acc = calc_acc(Y, Yall_pred);

    [train_prec, train_rec, train_f1] = calc_prf(Ytrain, Ytrain_pred);
    [test_prec, test_rec, test_f1] = calc_prf(Ytest, Ytest_pred);
    [all_prec, all_rec, all_f1] = calc_prf(Y, Yall_pred);

    fprintf('\n[Training Set Evaluation] \nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f \n', ...
        train_acc, train_prec, train_rec, train_f1);
    fprintf('[Test Set Evaluation] \nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f \n', ...
        test_acc, test_prec, test_rec, test_f1);
    fprintf('[All Data Evaluation] \nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f \n', ...
        all_acc, all_prec, all_rec, all_f1);


    figure;
    confusionchart(Ytest, Ytest_pred);
    title('Decision Tree Classification Confusion Matrix (Test Set)');

 
    handles.Mdl = Mdl;
    handles.mu = mu;
    handles.sigma = sigma;
    handles.train_acc = train_acc;
    handles.test_acc = test_acc;
    handles.all_acc = all_acc;
    handles.train_prec = train_prec;
    handles.train_rec = train_rec;
    handles.train_f1 = train_f1;
    handles.test_prec = test_prec;
    handles.test_rec = test_rec;
    handles.test_f1 = test_f1;
    handles.all_prec = all_prec;
    handles.all_rec = all_rec;
    handles.all_f1 = all_f1;
    handles.Ytrain_pred = Ytrain_pred;
    handles.Ytest_pred = Ytest_pred;
    handles.Yall_pred = Yall_pred;
    handles.Ytrain = Ytrain;
    handles.Ytest = Ytest;
    handles.Yall = Y;

    guidata(hObject, handles);

    assignin('base', 'TrainedDTModel', Mdl);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest_pred', Ytest_pred);
    assignin('base', 'Yall_pred', Yall_pred);



% --- 局部子函数：计算Precision, Recall, F1 ---
function [prec, rec, f1] = calc_prf(Ytrue, Ypred)
    C = confusionmat(Ytrue, Ypred);
    numClasses = size(C,1);
    precision = zeros(numClasses,1);
    recall = zeros(numClasses,1);
    f1score = zeros(numClasses,1);
    for i = 1:numClasses
        TP = C(i,i);
        FP = sum(C(:,i)) - TP;
        FN = sum(C(i,:)) - TP;
        precision(i) = TP / (TP + FP + eps);
        recall(i) = TP / (TP + FN + eps);
        f1score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
    end
    
    prec = mean(precision);
    rec = mean(recall);
    f1 = mean(f1score);
