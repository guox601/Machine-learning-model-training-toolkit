function varargout = MLRegressionCross(varargin)
% MLREGRESSIONCROSS MATLAB code for MLRegressionCross.fig
%      MLREGRESSIONCROSS, by itself, creates a new MLREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = MLREGRESSIONCROSS returns the handle to a new MLREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      MLREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MLREGRESSIONCROSS.M with the given input arguments.
%
%      MLREGRESSIONCROSS('Property','Value',...) creates a new MLREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MLRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MLRegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MLRegressionCross

% Last Modified by GUIDE v2.5 05-Jun-2025 10:43:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MLRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @MLRegressionCross_OutputFcn, ...
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


% --- Executes just before MLRegressionCross is made visible.
function MLRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MLRegressionCross (see VARARGIN)

% Choose default command line output for MLRegressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MLRegressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MLRegressionCross_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    filename = get(handles.edit1, 'string');
    kfold = str2double(get(handles.edit2, 'string')); % K-fold
    outdim = str2double(get(handles.edit3, 'string')); % output dimensions

    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % shuffle

    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_ + 1:end);
    numOutputs = size(Y, 2);
    cv = cvpartition(size(Y, 1), 'KFold', kfold);

    % Initialize storage
    R2_all = zeros(kfold, numOutputs);
    MSE_all = zeros(kfold, numOutputs);
    RMSE_all = zeros(kfold, numOutputs);
    MAPE_all = zeros(kfold, numOutputs);

    Y_all_true = cell(1, numOutputs);
    Y_all_pred = cell(1, numOutputs);

    Models = cell(numOutputs, kfold); % Store models

    % Cross-validation
    for fold = 1:kfold
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        Xtrain_raw = X(trainIdx, :);
        Ytrain = Y(trainIdx, :);
        Xtest_raw  = X(testIdx, :);
        Ytest = Y(testIdx, :);

        % Z-score normalization
        mu = mean(Xtrain_raw);
        sigma = std(Xtrain_raw);
        sigma(sigma == 0) = 1;
        Xtrain = (Xtrain_raw - mu) ./ sigma;
        Xtest  = (Xtest_raw - mu) ./ sigma;

        fprintf('[Fold %d]\n', fold);

        for dim = 1:numOutputs
            y_train = Ytrain(:, dim);
            y_test = Ytest(:, dim);

            % Train linear regression
            Mdl = fitlm(Xtrain, y_train);
            Models{dim, fold} = Mdl;

            % Predict
            y_pred = predict(Mdl, Xtest);

            % Evaluation
            R2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);
            MSE = mean((y_test - y_pred).^2);
            RMSE = sqrt(MSE);
            MAPE = mean(abs((y_test - y_pred) ./ (y_test + eps))) * 100;

            R2_all(fold, dim)   = R2;
            MSE_all(fold, dim)  = MSE;
            RMSE_all(fold, dim) = RMSE;
            MAPE_all(fold, dim) = MAPE;

            Y_all_true{dim} = [Y_all_true{dim}; y_test];
            Y_all_pred{dim} = [Y_all_pred{dim}; y_pred];

            fprintf('  Output %d | R² = %.4f | MSE = %.4f | RMSE = %.4f | MAPE = %.2f%%\n', ...
                dim, R2, MSE, RMSE, MAPE);
        end
    end

    % Summary
    fprintf('\n=== Average k-Fold Metrics per Output Dimension ===\n');
    for dim = 1:numOutputs
        fprintf('Output %d | R² = %.4f | MSE = %.4f | RMSE = %.4f | MAPE = %.2f%%\n', ...
            dim, mean(R2_all(:, dim)), mean(MSE_all(:, dim)), ...
            mean(RMSE_all(:, dim)), mean(MAPE_all(:, dim)));
    end

    % Plot predicted vs actual
    for dim = 1:numOutputs
        figure;
        scatter(Y_all_true{dim}, Y_all_pred{dim}, 25, 'filled');
        xlabel('Actual Value');
        ylabel('Predicted Value');
        title(sprintf('Output %d: Predicted vs Actual (k=%d)', dim, kfold));
        grid on;
        refline(1, 0);
    end

    % Export to base workspace
    assignin('base', 'Linear_Models', Models);
    assignin('base', 'Linear_Ytrue_all', Y_all_true);
    assignin('base', 'Linear_Ypred_all', Y_all_pred);





    
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



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
