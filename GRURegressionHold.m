function varargout = GRURegressionHold(varargin)
% GRUREGRESSIONHOLD MATLAB code for GRURegressionHold.fig
%      GRUREGRESSIONHOLD, by itself, creates a new GRUREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = GRUREGRESSIONHOLD returns the handle to a new GRUREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      GRUREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GRUREGRESSIONHOLD.M with the given input arguments.
%
%      GRUREGRESSIONHOLD('Property','Value',...) creates a new GRUREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GRURegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GRURegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GRURegressionHold

% Last Modified by GUIDE v2.5 03-Jun-2025 20:20:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GRURegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @GRURegressionHold_OutputFcn, ...
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


% --- Executes just before GRURegressionHold is made visible.
function GRURegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GRURegressionHold (see VARARGIN)

% Choose default command line output for GRURegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GRURegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GRURegressionHold_OutputFcn(hObject, eventdata, handles) 
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

    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;

    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    Xall = (X - mu) ./ sigma;

    % Format input for sequence input
    Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
    Xtest_seq  = squeeze(num2cell(Xtest', [1]));
    Xall_seq   = squeeze(num2cell(Xall', [1]));

    numFeatures = f_;

    % Bayesian optimization
    optimVars = [
        optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
        optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
    ];

    objFcn = @(optVars) trainAndEvaluate(optVars, Xtrain_seq, Ytrain, Xtest_seq, Ytest, numFeatures, outdim);

    results = bayesopt(objFcn, optimVars, ...
        'MaxObjectiveEvaluations', 30, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1);

    bestParams = results.XAtMinObjective;
    fprintf('Best numHiddenUnits: %d\n', bestParams.numHiddenUnits);
    fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);

    % Final model
    layers = [
        sequenceInputLayer(numFeatures)
        gruLayer(bestParams.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(outdim)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...
        'InitialLearnRate', bestParams.InitialLearnRate, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1, ...
        'Plots', 'training-progress');

    finalNet = trainNetwork(Xtrain_seq, Ytrain, layers, options);

    % Predict
    Ytrain_pred = predict(finalNet, Xtrain_seq);
    Ytest_pred = predict(finalNet, Xtest_seq);
    Yall_pred = predict(finalNet, Xall_seq);

    % Evaluate and plot per dimension
    for i = 1:outdim
        % Evaluation function
        calc_metrics = @(Ytrue, Ypred) struct( ...
            'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
            'MSE', mean((Ytrue - Ypred).^2), ...
            'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
            'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
        );

        % Actual vs predicted for i-th target
        train_metrics = calc_metrics(Ytrain(:, i), Ytrain_pred(:, i));
        test_metrics = calc_metrics(Ytest(:, i), Ytest_pred(:, i));

        % Print
        fprintf('\n[Target %d - Training Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);

        fprintf('[Target %d - Test Set]\n', i);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);

        % Plot
        figure;
        scatter(Ytest(:, i), Ytest_pred(:, i), 'filled');
        xlabel('Actual Value');
        ylabel('Predicted Value');
        title(sprintf('GRU Regression - Target %d', i));
        grid on;
        refline(1, 0);
    end

    assignin('base', 'GRU_finalNet', finalNet);
    assignin('base', 'GRU_Ytest', Ytest);
    assignin('base', 'GRU_Ytest_pred', Ytest_pred);
    assignin('base', 'GRU_Ytrain', Ytrain);
    assignin('base', 'GRU_Ytrain_pred', Ytrain_pred);


function objective = trainAndEvaluate(optVars, Xtrain_seq, Ytrain, Xtest_seq, Ytest, numFeatures, outdim)
    layers = [
        sequenceInputLayer(numFeatures)
        gruLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(outdim)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', optVars.InitialLearnRate, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0);

    net = trainNetwork(Xtrain_seq, Ytrain, layers, options);
    Ypred = predict(net, Xtest_seq);

    R2 = 1 - sum((Ytest(:) - Ypred(:)).^2) / sum((Ytest(:) - mean(Ytest(:))).^2);
    objective = -R2;


    
    % shuffle
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
