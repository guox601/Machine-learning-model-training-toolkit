function varargout = BpANNegressionHold(varargin)
% BPANNEGRESSIONHOLD MATLAB code for BpANNegressionHold.fig
%      BPANNEGRESSIONHOLD, by itself, creates a new BPANNEGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = BPANNEGRESSIONHOLD returns the handle to a new BPANNEGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      BPANNEGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BPANNEGRESSIONHOLD.M with the given input arguments.
%
%      BPANNEGRESSIONHOLD('Property','Value',...) creates a new BPANNEGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BpANNegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BpANNegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BpANNegressionHold

% Last Modified by GUIDE v2.5 03-Jun-2025 16:19:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BpANNegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @BpANNegressionHold_OutputFcn, ...
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


% --- Executes just before BpANNegressionHold is made visible.
function BpANNegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BpANNegressionHold (see VARARGIN)

% Choose default command line output for BpANNegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BpANNegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BpANNegressionHold_OutputFcn(hObject, eventdata, handles) 
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
    % Load inputs from GUI
    filename = get(handles.edit1, 'string');
    outdim = str2double(get(handles.edit3, 'string'));
    proportion = str2double(get(handles.edit2, 'string'));
    
    % Load and shuffle data
    res = xlsread(filename);  
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :); 

    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);      
    Y = res(:, f_ + 1:end); 

    % Hold-out split
    cv = cvpartition(size(Y,1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalize features
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma==0) = 1;

    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;

    numFeatures = f_;

    % Split training set into train/val for bayesopt
    cv2 = cvpartition(size(Ytrain,1), 'HoldOut', 0.2);
    Xtrain_bayes = Xtrain(training(cv2), :);
    Ytrain_bayes = Ytrain(training(cv2), :);
    Xval_bayes = Xtrain(test(cv2), :);
    Yval_bayes = Ytrain(test(cv2), :);

    % Define objective function for bayesopt
    objFcn = @(params) trainEvalANN(params, Xtrain_bayes, Ytrain_bayes, Xval_bayes, Yval_bayes, numFeatures);

    % Hyperparameter space
    optVars = [
        optimizableVariable('NumHiddenUnits', [10, 200], 'Type', 'integer')
        optimizableVariable('InitialLearnRate', [1e-4, 1e-2], 'Transform', 'log')
        optimizableVariable('MaxEpochs', [50, 300], 'Type', 'integer')
        optimizableVariable('MiniBatchSize', [8, 64], 'Type', 'integer')
    ];

    % Run Bayesian Optimization
    results = bayesopt(objFcn, optVars, ...
        'MaxObjectiveEvaluations', 20, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1, ...
        'UseParallel', false);

    bestParams = results.XAtMinObjective;

    % Train final model
    layers = [
        featureInputLayer(numFeatures, 'Normalization', 'none')
        fullyConnectedLayer(bestParams.NumHiddenUnits)
        reluLayer
        fullyConnectedLayer(outdim)
        regressionLayer];

    options = trainingOptions('adam', ...
        'MaxEpochs', bestParams.MaxEpochs, ...
        'InitialLearnRate', bestParams.InitialLearnRate, ...
        'MiniBatchSize', bestParams.MiniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1, ...
        'Plots', 'training-progress');

    net = trainNetwork(Xtrain, Ytrain, layers, options);

    % Prediction
    Ytrain_pred = predict(net, Xtrain);
    Ytest_pred = predict(net, Xtest);

    % Save predictions and model
    assignin('base', 'TrainedANNModel', net);
    assignin('base', 'Ytrain', Ytrain);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest', Ytest);
    assignin('base', 'Ytest_pred', Ytest_pred);

    % Evaluation metrics
    calc_metrics = @(Ytrue, Ypred) struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) ./ sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
    );

    n_targets = size(Y,2);
    for i = 1:n_targets
        train_metrics = calc_metrics(Ytrain(:,i), Ytrain_pred(:,i));
        test_metrics = calc_metrics(Ytest(:,i), Ytest_pred(:,i));

        fprintf('\n[Target %d - Training Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            i, train_metrics.R2, train_metrics.MSE, train_metrics.RMSE, train_metrics.MAPE);
        fprintf('[Target %d - Test Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            i, test_metrics.R2, test_metrics.MSE, test_metrics.RMSE, test_metrics.MAPE);
    end

    % Plot
    if n_targets == 1
        figure;
        scatter(Ytest, Ytest_pred, 'filled');
        xlabel('Actual Value');
        ylabel('Predicted Value');
        title('Feedforward ANN Regression Results (Test Set)');
        grid on;
        refline(1,0);
    end

    %% ======= Local training function =======
    function valLoss = trainEvalANN(params, Xtrain, Ytrain, Xval, Yval, numFeatures)
        layers = [
            featureInputLayer(numFeatures, 'Normalization', 'none')
            fullyConnectedLayer(params.NumHiddenUnits)
            reluLayer
            fullyConnectedLayer(size(Ytrain,2))
            regressionLayer];

        options = trainingOptions('adam', ...
            'MaxEpochs', params.MaxEpochs, ...
            'InitialLearnRate', params.InitialLearnRate, ...
            'MiniBatchSize', params.MiniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', 0, ...
            'Plots', 'none', ...
            'ValidationData', {Xval, Yval}, ...
            'ValidationFrequency', 20);

        net = trainNetwork(Xtrain, Ytrain, layers, options);
        Yval_pred = predict(net, Xval);
        valLoss = sqrt(mean((Yval - Yval_pred).^2, 'all'));
   



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of 