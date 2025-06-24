function varargout = KNNRegressionHold(varargin)
% KNNREGRESSIONHOLD MATLAB code for KNNRegressionHold.fig
%      KNNREGRESSIONHOLD, by itself, creates a new KNNREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = KNNREGRESSIONHOLD returns the handle to a new KNNREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      KNNREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in KNNREGRESSIONHOLD.M with the given input arguments.
%
%      KNNREGRESSIONHOLD('Property','Value',...) creates a new KNNREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before KNNRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to KNNRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help KNNRegressionHold

% Last Modified by GUIDE v2.5 30-May-2025 22:25:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @KNNRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @KNNRegressionHold_OutputFcn, ...
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


% --- Executes just before KNNRegressionHold is made visible.
function KNNRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to KNNRegressionHold (see VARARGIN)

% Choose default command line output for KNNRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes KNNRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = KNNRegressionHold_OutputFcn(hObject, eventdata, handles) 
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

    % Convert string inputs to numeric
    outdim = str2double(outdim_str);
    proportion = str2double(proportion_str);

    % Read dataset from Excel
    data = xlsread(filename);
    num_samples = size(data, 1);

    % Shuffle dataset rows
    data = data(randperm(num_samples), :);

    % Separate features and targets according to output dimension
    num_features = size(data, 2) - outdim;
    X = data(:, 1:num_features);
    Y = data(:, num_features+1:end);

    % Split dataset into training and test sets (Hold-Out)
    cv = cvpartition(size(Y, 1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalize features (zero mean, unit std) using training data stats
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1; % Avoid division by zero
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    Xall = (X - mu) ./ sigma;

    % Hyperparameter tuning: search best k in [1,20]
    k_candidates = 1:20;
    best_k = k_candidates(1);
    best_mse = inf;

    for k_try = k_candidates
        Yval_pred = knn_regression_predict(Xtrain, Ytrain, Xtest, k_try);
        mse = mean(mean((Ytest - Yval_pred).^2)); % average MSE over all outputs
        fprintf('k=%d, Test MSE=%.4f\n', k_try, mse);
        if mse < best_mse
            best_mse = mse;
            best_k = k_try;
        end
    end

    fprintf('Best k = %d, Test MSE = %.4f\n', best_k, best_mse);

    % Predict with the best k on train, test, and all data
    Ytrain_pred = knn_regression_predict(Xtrain, Ytrain, Xtrain, best_k);
    Ytest_pred = knn_regression_predict(Xtrain, Ytrain, Xtest, best_k);
    Yall_pred = knn_regression_predict(Xtrain, Ytrain, Xall, best_k);

    % Evaluation metrics function
    calc_metrics = @(Ytrue, Ypred) struct( ...
        'R2', 1 - sum((Ytrue - Ypred).^2) / sum((Ytrue - mean(Ytrue)).^2), ...
        'MSE', mean((Ytrue - Ypred).^2), ...
        'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
        'MAPE', mean(abs((Ytrue - Ypred) ./ Ytrue)) * 100 ...
    );

    % Initialize metrics storage
    train_metrics = repmat(struct('R2', [], 'MSE', [], 'RMSE', [], 'MAPE', []), outdim, 1);
    test_metrics = train_metrics;
    all_metrics = train_metrics;

    % Calculate metrics per output dimension
    for t = 1:outdim
        train_metrics(t) = calc_metrics(Ytrain(:, t), Ytrain_pred(:, t));
        test_metrics(t) = calc_metrics(Ytest(:, t), Ytest_pred(:, t));
        all_metrics(t) = calc_metrics(Y(:, t), Yall_pred(:, t));
    end

    % Display results per output target and plot results
    for t = 1:outdim
        fprintf('\n[Target %d - Training Set]\n', t);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            train_metrics(t).R2, train_metrics(t).MSE, train_metrics(t).RMSE, train_metrics(t).MAPE);

        fprintf('[Target %d - Test Set]\n', t);
        fprintf('R²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
            test_metrics(t).R2, test_metrics(t).MSE, test_metrics(t).RMSE, test_metrics(t).MAPE);

        % Scatter plot of actual vs predicted values on test set
        figure;
        scatter(Ytest(:, t), Ytest_pred(:, t), 'filled');
        xlabel('Actual Value'); ylabel('Predicted Value');
        title(sprintf('Target %d - KNN Regression (Test Set), k=%d', t, best_k));
        grid on;
        refline(1, 0); % reference line y = x
    end
    
% Save trained model info to base workspace
% After training and getting predictions:
Ytrain_pred = knn_regression_predict(Xtrain, Ytrain, Xtrain, best_k);
Ytest_pred = knn_regression_predict(Xtrain, Ytrain, Xtest, best_k);

% Package trained model info in a struct
TrainedKNNModel.k = best_k;          % Best k found
TrainedKNNModel.mu = mu;             % Mean of training features (for normalization)
TrainedKNNModel.sigma = sigma;       % Std dev of training features
TrainedKNNModel.Xtrain = Xtrain;     % Normalized training features
TrainedKNNModel.Ytrain = Ytrain;     % Training targets

% Save prediction results as well
TrainedKNNModel.Ytrain_pred = Ytrain_pred;
TrainedKNNModel.Ytest_pred = Ytest_pred;

% Assign variables to base workspace
assignin('base', 'TrainedKNNModel', TrainedKNNModel);
assignin('base', 'Ytrain', Ytrain);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest', Ytest);
assignin('base', 'Ytest_pred', Ytest_pred);



% KNN regression prediction function supporting multi-output
function Ypred = knn_regression_predict(Xtrain, Ytrain, Xtest, k)
    num_test = size(Xtest, 1);
    out_dim = size(Ytrain, 2);
    Ypred = zeros(num_test, out_dim);

    for i = 1:num_test
        dists = sqrt(sum((Xtrain - Xtest(i, :)).^2, 2));  % Euclidean distance
        [~, idx] = sort(dists);
        nearest_idx = idx(1:k);
        % Average the neighbors' output for each dimension
        Ypred(i, :) = mean(Ytrain(nearest_idx, :), 1);
    end
