function varargout = GPRRegressionHold(varargin)
% GPRREGRESSIONHOLD MATLAB code for GPRRegressionHold.fig
%      GPRREGRESSIONHOLD, by itself, creates a new GPRREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = GPRREGRESSIONHOLD returns the handle to a new GPRREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      GPRREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GPRREGRESSIONHOLD.M with the given input arguments.
%
%      GPRREGRESSIONHOLD('Property','Value',...) creates a new GPRREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GPRRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GPRRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GPRRegressionHold

% Last Modified by GUIDE v2.5 03-Jun-2025 20:41:06

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GPRRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @GPRRegressionHold_OutputFcn, ...
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


% --- Executes just before GPRRegressionHold is made visible.
function GPRRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GPRRegressionHold (see VARARGIN)

% Choose default command line output for GPRRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GPRRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GPRRegressionHold_OutputFcn(hObject, eventdata, handles) 
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
    filepath = get(handles.edit1, 'String');
    trainRatio = str2double(get(handles.edit2, 'String'));
    outdim = str2double(get(handles.edit3, 'String'));

    res = xlsread(filepath);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_ + 1:end);

    cv = cvpartition(size(Y, 1), 'HoldOut', trainRatio);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalization
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;
    Xall   = (X - mu) ./ sigma;

    % Train GPR for each output dimension
    rng(1);
    Mdl = cell(1, outdim);
    Ytrain_pred = zeros(size(Ytrain));
    Ytest_pred  = zeros(size(Ytest));
    Yall_pred   = zeros(size(Y));

    for i = 1:outdim
        Mdl{i} = fitrgp(Xtrain, Ytrain(:, i), ...
            'Standardize', false, ...
            'KernelFunction', 'squaredexponential', ...
            'OptimizeHyperparameters', {'KernelScale', 'Sigma'}, ...
            'HyperparameterOptimizationOptions', struct( ...
                'AcquisitionFunctionName', 'expected-improvement-plus', ...
                'ShowPlots', false, ...
                'Verbose', 0 ...
            ));
        Ytrain_pred(:, i) = predict(Mdl{i}, Xtrain);
        Ytest_pred(:, i)  = predict(Mdl{i}, Xtest);
        Yall_pred(:, i)   = predict(Mdl{i}, Xall);
    end

    % Evaluation
    calc_metrics = @(Ytrue, Ypred) struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
    );

    for i = 1:outdim
        train_metrics = calc_metrics(Ytrain(:, i), Ytrain_pred(:, i));
        test_metrics  = calc_metrics(Ytest(:, i), Ytest_pred(:, i));

        fprintf('[Target %d - Training Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);
        fprintf('[Target %d - Test Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n\n', ...
            test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);

        % Scatter plot
        figure;
        scatter(Ytest(:, i), Ytest_pred(:, i), 'filled');
        xlabel('Actual Value'); ylabel('Predicted Value');
        title(sprintf('GPR Prediction (Target %d - Test Set)', i));
        grid on;
        refline(1, 0);
    end

    % Save results to workspace
    assignin('base', 'TrainedGPRModel', Mdl);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest_pred', Ytest_pred);
    assignin('base', 'Yall_pred', Yall_pred);

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
