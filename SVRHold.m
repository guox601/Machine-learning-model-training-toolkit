function varargout = SVRHold(varargin)
% SVRHOLD MATLAB code for SVRHold.fig
%      SVRHOLD, by itself, creates a new SVRHOLD or raises the existing
%      singleton*.
%
%      H = SVRHOLD returns the handle to a new SVRHOLD or the handle to
%      the existing singleton*.
%
%      SVRHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SVRHOLD.M with the given input arguments.
%
%      SVRHOLD('Property','Value',...) creates a new SVRHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SVRHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SVRHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SVRHold

% Last Modified by GUIDE v2.5 30-May-2025 13:53:38

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SVRHold_OpeningFcn, ...
                   'gui_OutputFcn',  @SVRHold_OutputFcn, ...
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


% --- Executes just before SVRHold is made visible.
function SVRHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SVRHold (see VARARGIN)

% Choose default command line output for SVRHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SVRHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SVRHold_OutputFcn(hObject, eventdata, handles) 
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
    name1= get(handles.edit1, 'string');
    outdim1= get(handles.edit3, 'string');
    proportion1 = get(handles.edit2, 'string');

    outdim = str2num(outdim1);
    proportion = str2num(proportion1);

    res = xlsread(name1);
    num_samples = size(res, 1); 
    res = res(randperm(num_samples), :);   % Shuffle rows
    f_ = size(res, 2) - outdim;     
    X = res(:, 1:f_);      % Features
    Y = res(:, f_ + 1:end);  % Targets

    % Split training and test sets
    cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Data normalization
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;
    Xall   = (X - mu) ./ sigma;

    % Initialize storage for models and predictions
    models = cell(outdim,1);
    Ytrain_pred = zeros(size(Ytrain));
    Ytest_pred = zeros(size(Ytest));
    Yall_pred = zeros(size(Y));

    % Define evaluation function for single dimension
    calc_metrics = @(Ytrue, Ypred) struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ Ytrue)) * 100 ...
    );

    % Train models for each output dimension
for d = 1:outdim
    fprintf('Training SVR for output dimension %d/%d\n', d, outdim);
    rng(1);  % for reproducibility
    models{d} = fitrsvm(Xtrain, Ytrain(:,d), ...
        'KernelFunction', 'gaussian', ...
        'Standardize', false, ...
        'IterationLimit', 1000, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'Epsilon', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct( ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'ShowPlots', false, ...
            'Verbose', 0 ...
        ));

    % Predict for this dimension
    Ytrain_pred(:,d) = predict(models{d}, Xtrain);
    Ytest_pred(:,d) = predict(models{d}, Xtest);
    Yall_pred(:,d) = predict(models{d}, Xall);

    % Calculate metrics for train and test sets
    train_metrics = calc_metrics(Ytrain(:,d), Ytrain_pred(:,d));
    test_metrics = calc_metrics(Ytest(:,d), Ytest_pred(:,d));

    % Print with desired format
    fprintf('\n[Target %d - Training Set]\n', d);
    fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);

    fprintf('[Target %d - Test Set]\n', d);
    fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n\n', ...
        test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);

    % Plot predicted vs actual for this output dimension (Test set)
    figure;
    scatter(Ytest(:,d), Ytest_pred(:,d), 'filled');
    hold on;
    plot([min(Ytest(:,d)) max(Ytest(:,d))], [min(Ytest(:,d)) max(Ytest(:,d))], 'r--', 'LineWidth', 1.5);
    xlabel('Actual Value');
    ylabel('Predicted Value');
    title(sprintf('SVR Predicted vs Actual - Output Dimension %d (Test Set)', d));
    grid on;
    hold off;
end

    % Calculate and display averaged metrics for all outputs
    avg_metrics = struct( ...
        'R2', mean(arrayfun(@(i) calc_metrics(Ytest(:,i), Ytest_pred(:,i)).R2, 1:outdim)), ...
        'MSE', mean(arrayfun(@(i) calc_metrics(Ytest(:,i), Ytest_pred(:,i)).MSE, 1:outdim)), ...
        'RMSE', mean(arrayfun(@(i) calc_metrics(Ytest(:,i), Ytest_pred(:,i)).RMSE, 1:outdim)), ...
        'MAPE', mean(arrayfun(@(i) calc_metrics(Ytest(:,i), Ytest_pred(:,i)).MAPE, 1:outdim)) ...
    );


    % Save models and predictions to base workspace if needed
    assignin('base', 'TrainedSVRModels', models);
    assignin('base', 'Ytrain', Ytrain);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest', Ytest);
    assignin('base', 'Ytest_pred', Ytest_pred);

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
