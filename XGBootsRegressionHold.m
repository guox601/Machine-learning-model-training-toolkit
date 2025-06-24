function varargout = XGBootsRegressionHold(varargin)
% XGBOOTSREGRESSIONHOLD MATLAB code for XGBootsRegressionHold.fig
%      XGBOOTSREGRESSIONHOLD, by itself, creates a new XGBOOTSREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = XGBOOTSREGRESSIONHOLD returns the handle to a new XGBOOTSREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      XGBOOTSREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in XGBOOTSREGRESSIONHOLD.M with the given input arguments.
%
%      XGBOOTSREGRESSIONHOLD('Property','Value',...) creates a new XGBOOTSREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before XGBootsRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to XGBootsRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help XGBootsRegressionHold

% Last Modified by GUIDE v2.5 30-May-2025 13:42:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @XGBootsRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @XGBootsRegressionHold_OutputFcn, ...
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


% --- Executes just before XGBootsRegressionHold is made visible.
function XGBootsRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to XGBootsRegressionHold (see VARARGIN)

% Choose default command line output for XGBootsRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes XGBootsRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = XGBootsRegressionHold_OutputFcn(hObject, eventdata, handles) 
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
% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    % Read inputs from GUI
    filename = get(handles.edit1, 'String');
    outdim_str = get(handles.edit3, 'String');
    proportion_str = get(handles.edit2, 'String');

    outdim = str2double(outdim_str);
    proportion = str2double(proportion_str);

    % Load data
    data = xlsread(filename);

    % Shuffle rows
    num_samples = size(data, 1);
    data = data(randperm(num_samples), :);

    % Split features and targets
    num_features = size(data, 2) - outdim;
    X = data(:, 1:num_features);
    Y = data(:, num_features+1:end);

    % Split into training and test sets (Hold-out)
    cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalize features based on training data
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;

    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    Xall = (X - mu) ./ sigma;

    num_targets = size(Ytrain, 2);
    Models = cell(num_targets, 1);
    Ytrain_pred = zeros(size(Ytrain));
    Ytest_pred = zeros(size(Ytest));
    Yall_pred = zeros(size(Y));

    % Train one model per target
    for i = 1:num_targets
        fprintf('Training model for target %d...\n', i);

        Models{i} = fitrensemble(Xtrain, Ytrain(:, i), ...
            'Method', 'LSBoost', ...
            'OptimizeHyperparameters', {'NumLearningCycles', 'LearnRate', 'MinLeafSize', 'MaxNumSplits'}, ...
            'HyperparameterOptimizationOptions', struct( ...
                'AcquisitionFunctionName', 'expected-improvement-plus', ...
                'ShowPlots', false, ...
                'Verbose', 0 ...
            ));

        % Predict
        Ytrain_pred(:, i) = predict(Models{i}, Xtrain);
        Ytest_pred(:, i) = predict(Models{i}, Xtest);
        Yall_pred(:, i) = predict(Models{i}, Xall);
    end

    % Metrics calculation function
    calc_metrics = @(Ytrue, Ypred) struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) ./ sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
    );

    % Print metrics per target
    for i = 1:num_targets
        train_metrics = calc_metrics(Ytrain(:, i), Ytrain_pred(:, i));
        test_metrics = calc_metrics(Ytest(:, i), Ytest_pred(:, i));

        fprintf('\n[Target %d - Training Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);

        fprintf('[Target %d - Test Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);

        % Plot predicted vs actual for test set of each target
        figure;
        scatter(Ytest(:, i), Ytest_pred(:, i), 'filled');
        xlabel('Actual Value');
        ylabel('Predicted Value');
        title(sprintf('XGBoost Regression Results (Target %d - Test Set)', i));
        grid on;
        refline(1,0);  % 45-degree line
    end

    % Assign variables to base workspace for further use
    assignin('base', 'Models', Models);
    assignin('base', 'Ytrain', Ytrain);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest', Ytest);
    assignin('base', 'Ytest_pred', Ytest_pred);




% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
