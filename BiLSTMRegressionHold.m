function varargout = BiLSTMRegressionHold(varargin)
% BILSTMREGRESSIONHOLD MATLAB code for BiLSTMRegressionHold.fig
%      BILSTMREGRESSIONHOLD, by itself, creates a new BILSTMREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = BILSTMREGRESSIONHOLD returns the handle to a new BILSTMREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      BILSTMREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BILSTMREGRESSIONHOLD.M with the given input arguments.
%
%      BILSTMREGRESSIONHOLD('Property','Value',...) creates a new BILSTMREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before BiLSTMRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to BiLSTMRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help BiLSTMRegressionHold

% Last Modified by GUIDE v2.5 03-Jun-2025 20:48:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BiLSTMRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @BiLSTMRegressionHold_OutputFcn, ...
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


% --- Executes just before BiLSTMRegressionHold is made visible.
function BiLSTMRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BiLSTMRegressionHold (see VARARGIN)

% Choose default command line output for BiLSTMRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BiLSTMRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BiLSTMRegressionHold_OutputFcn(hObject, eventdata, handles) 
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

% Split into training and test sets
cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
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

% Convert to sequences
Xtrain_seq = squeeze(num2cell(Xtrain', [1]));
Xtest_seq  = squeeze(num2cell(Xtest', [1]));
Xall_seq   = squeeze(num2cell(Xall', [1]));

Ytrain_seq = Ytrain;
Ytest_seq = Ytest;
Yall_seq = Y;

numFeatures = f_;

% Define hyperparameter search space
optimVars = [
    optimizableVariable('numHiddenUnits', [50 200], 'Type', 'integer')
    optimizableVariable('InitialLearnRate', [1e-4 1e-2], 'Transform', 'log')
];

% Use only the first output for Bayesian optimization
objFcn = @(optVars) trainAndEvaluateBiLSTM(optVars, Xtrain_seq, Ytrain_seq(:, 1), Xtest_seq, Ytest_seq(:, 1), numFeatures);
results = bayesopt(objFcn, optimVars, ...
    'MaxObjectiveEvaluations', 30, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

% Best hyperparameters
bestParams = results.XAtMinObjective;
fprintf('Best numHiddenUnits: %d\n', bestParams.numHiddenUnits);
fprintf('Best InitialLearnRate: %.6f\n', bestParams.InitialLearnRate);

% Define layers for multi-output
layers = [
    sequenceInputLayer(numFeatures)
    bilstmLayer(bestParams.numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(outdim)
    regressionLayer
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', bestParams.InitialLearnRate, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

% Train final model
finalNet = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);

% Predict
Ytrain_pred = predict(finalNet, Xtrain_seq);
Ytest_pred = predict(finalNet, Xtest_seq);
Yall_pred = predict(finalNet, Xall_seq);

% Metrics function
calc_metrics = @(Ytrue, Ypred) struct( ...
    'R2', 1 - sum((Ytrue - Ypred).^2) ./ sum((Ytrue - mean(Ytrue)).^2), ...
    'MSE', mean((Ytrue - Ypred).^2), ...
    'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
    'MAPE', mean(abs((Ytrue - Ypred) ./ (Ytrue + eps))) * 100 ...
);

% Print metrics for each output dimension
for i = 1:outdim
    train_m = calc_metrics(Ytrain_seq(:, i), Ytrain_pred(:, i));
    test_m  = calc_metrics(Ytest_seq(:, i), Ytest_pred(:, i));
    
    fprintf('\n[Target %d - Training Set]\n', i);
    fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        train_m.R2, train_m.MSE, train_m.RMSE, train_m.MAPE);
    
    fprintf('[Target %d - Test Set]\n', i);
    fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        test_m.R2, test_m.MSE, test_m.RMSE, test_m.MAPE);
end

% Plot predictions of first target
% Plot predictions in subplots
figure;
for i = 1:outdim
    subplot(ceil(outdim / 2), 2, i);  % 每行最多2个子图，自适应行数
    scatter(Ytest_seq(:, i), Ytest_pred(:, i), 'filled');
    xlabel('Actual');
    ylabel('Predicted');
    title(sprintf('Target %d', i));
    grid on;
    refline(1, 0);
end
sgtitle('Bi-LSTM Regression Results (Test Set)');


% Export to workspace
assignin('base', 'TrainedBiLSTMModel', finalNet);
assignin('base', 'Ytrain', Ytrain_seq);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest', Ytest_seq);
assignin('base', 'Ytest_pred', Ytest_pred);
assignin('base', 'Yall_pred', Yall_pred);

% === Bi-LSTM objective function ===
function objective = trainAndEvaluateBiLSTM(optVars, Xtrain_seq, Ytrain_seq, Xtest_seq, Ytest_seq, numFeatures)
    layers = [
        sequenceInputLayer(numFeatures)
        bilstmLayer(optVars.numHiddenUnits, 'OutputMode', 'last')
        fullyConnectedLayer(1)
        regressionLayer
    ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'InitialLearnRate', optVars.InitialLearnRate, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0);
    
    net = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);
    Ypred = predict(net, Xtest_seq);
    
    R2 = 1 - sum((Ytest_seq - Ypred).^2) / sum((Ytest_seq - mean(Ytest_seq)).^2);
    objective = -R2;  % For bayesopt: minimize negative R²

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA) 
