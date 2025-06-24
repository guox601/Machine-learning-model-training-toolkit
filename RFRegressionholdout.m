function varargout = RFRegressionholdout(varargin)
% RFREGRESSIONHOLDOUT MATLAB code for RFRegressionholdout.fig
%      RFREGRESSIONHOLDOUT, by itself, creates a new RFREGRESSIONHOLDOUT or raises the existing
%      singleton*.
%
%      H = RFREGRESSIONHOLDOUT returns the handle to a new RFREGRESSIONHOLDOUT or the handle to
%      the existing singleton*.
%
%      RFREGRESSIONHOLDOUT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RFREGRESSIONHOLDOUT.M with the given input arguments.
%
%      RFREGRESSIONHOLDOUT('Property','Value',...) creates a new RFREGRESSIONHOLDOUT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RFRegressionholdout_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RFRegressionholdout_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RFRegressionholdout

% Last Modified by GUIDE v2.5 30-May-2025 22:26:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RFRegressionholdout_OpeningFcn, ...
                   'gui_OutputFcn',  @RFRegressionholdout_OutputFcn, ...
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


% --- Executes just before RFRegressionholdout is made visible.
function RFRegressionholdout_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RFRegressionholdout (see VARARGIN)

% Choose default command line output for RFRegressionholdout
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RFRegressionholdout wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = RFRegressionholdout_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function edit1_Callback(hObject, eventdata, handles)
   


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
    % Load inputs
    filename = get(handles.edit1, 'string');
    outdim_str = get(handles.edit3, 'string');
    proportion_str = get(handles.edit2, 'string');

    outdim = str2double(outdim_str);
    proportion = str2double(proportion_str);

    % Read data
    data = xlsread(filename);
    data = data(randperm(size(data,1)), :); % Shuffle rows

    num_features = size(data, 2) - outdim;
    X = data(:, 1:num_features);
    Y = data(:, num_features+1:end);

    % Split
    cv = cvpartition(size(Y,1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalize
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    Xall = (X - mu) ./ sigma;

    numLearningCycles = 100;
    num_targets = size(Ytrain, 2);

    Models = cell(num_targets, 1);
    Ytrain_pred = zeros(size(Ytrain));
    Ytest_pred = zeros(size(Ytest));
    Yall_pred = zeros(size(Y));

    rng(1); % reproducible

    for i = 1:num_targets
        fprintf('Training model for target %d...\n', i);
        Models{i} = fitrensemble(Xtrain, Ytrain(:, i), ...
            'Method', 'Bag', ...
            'NumLearningCycles', numLearningCycles, ...
            'OptimizeHyperparameters', {'MinLeafSize', 'MaxNumSplits'}, ...
            'HyperparameterOptimizationOptions', struct(...
                'AcquisitionFunctionName','expected-improvement-plus',...
                'ShowPlots', false,...
                'Verbose', 0));
        Ytrain_pred(:, i) = predict(Models{i}, Xtrain);
        Ytest_pred(:, i) = predict(Models{i}, Xtest);
        Yall_pred(:, i) = predict(Models{i}, Xall);
    end

    calc_metrics = @(Ytrue,Ypred) struct(...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100);

    for i = 1:num_targets
        train_metrics = calc_metrics(Ytrain(:, i), Ytrain_pred(:, i));
        test_metrics = calc_metrics(Ytest(:, i), Ytest_pred(:, i));

        fprintf('\n[Target %d - Training Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);

        fprintf('[Target %d - Test Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);

        figure;
        scatter(Ytest(:, i), Ytest_pred(:, i), 'filled');
        xlabel('Actual Value');
        ylabel('Predicted Value');
        title(sprintf('Random Forest Regression (Target %d - Test Set)', i));
        grid on;
        refline(1, 0);
    end

    assignin('base', 'Models', Models);
    assignin('base', 'Ytrain', Ytrain);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest', Ytest);
    assignin('base', 'Ytest_pred', Ytest_pred);



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
