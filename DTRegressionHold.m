function varargout = DTRegressionHold(varargin)
% DTREGRESSIONHOLD MATLAB code for DTRegressionHold.fig
%      DTREGRESSIONHOLD, by itself, creates a new DTREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = DTREGRESSIONHOLD returns the handle to a new DTREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      DTREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DTREGRESSIONHOLD.M with the given input arguments.
%
%      DTREGRESSIONHOLD('Property','Value',...) creates a new DTREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DTRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DTRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DTRegressionHold

% Last Modified by GUIDE v2.5 30-May-2025 14:11:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DTRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @DTRegressionHold_OutputFcn, ...
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


% --- Executes just before DTRegressionHold is made visible.
function DTRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DTRegressionHold (see VARARGIN)

% Choose default command line output for DTRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes DTRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = DTRegressionHold_OutputFcn(hObject, eventdata, handles) 
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
name1 = get(handles.edit1, 'string');
outdim1 = get(handles.edit3, 'string');
proportion1 = get(handles.edit2, 'string');

outdim = str2double(outdim1);
proportion = str2double(proportion1);

res = xlsread(name1);
f_ = size(res, 2) - outdim;
X = res(:, 1:f_);         % Features
Y = res(:, f_+1:end);     % Multiple target variables

% Hold-out data splitting
cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
Xtrain_raw = X(training(cv), :);
Ytrain_raw = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest_raw = Y(test(cv), :);

% Normalize features
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1; % Prevent divide by zero

Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest  = (Xtest_raw - mu) ./ sigma;
Xall   = (X - mu) ./ sigma;

% Initialize storage
models = cell(outdim,1);
train_metrics_list = cell(outdim,1);
test_metrics_list = cell(outdim,1);

for i = 1:outdim
    % Extract single target column
    Ytrain = Ytrain_raw(:, i);
    Ytest  = Ytest_raw(:, i);
    Yall   = Y(:, i);

    % Train model
    rng(1);
    Mdl = fitrtree(Xtrain, Ytrain, ...
        'OptimizeHyperparameters', {'MinLeafSize', 'MaxNumSplits', 'NumVariablesToSample'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'ShowPlots', false, ...
            'Verbose', 0));

    % Store model
    models{i} = Mdl;

    % Predict
    Ytrain_pred = predict(Mdl, Xtrain);
    Ytest_pred = predict(Mdl, Xtest);
    Yall_pred = predict(Mdl, Xall);

    % Evaluation function
    calc_metrics = @(Ytrue, Ypred) struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ max(abs(Ytrue), 1e-6))) * 100 ...
    );

    % Metrics
    train_metrics = calc_metrics(Ytrain, Ytrain_pred);
    test_metrics  = calc_metrics(Ytest, Ytest_pred);

    train_metrics_list{i} = train_metrics;
    test_metrics_list{i} = test_metrics;

    % Display
    fprintf('\n[Target %d - Training Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        i, train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);
    fprintf('[Target %d - Test Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        i, test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);

    % Plot predictions vs actual for test set
    figure;
    scatter(Ytest, Ytest_pred, 'filled');
    xlabel('Actual Value'); ylabel('Predicted Value');
    title(sprintf('DT Regression: Target %d', i));
    grid on; 
    refline(1, 0); % 45-degree line
end

% Assign to base workspace
assignin('base', 'DT_Models', models);
assignin('base', 'TrainMetricsList', train_metrics_list);
assignin('base', 'TestMetricsList', test_metrics_list);
assignin('base', 'Xtest', Xtest);
assignin('base', 'Ytest_raw', Ytest_raw);


