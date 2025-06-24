function varargout = MLRegressionHold(varargin)
% MLREGRESSIONHOLD MATLAB code for MLRegressionHold.fig
%      MLREGRESSIONHOLD, by itself, creates a new MLREGRESSIONHOLD or raises the existing
%      singleton*.
%
%      H = MLREGRESSIONHOLD returns the handle to a new MLREGRESSIONHOLD or the handle to
%      the existing singleton*.
%
%      MLREGRESSIONHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MLREGRESSIONHOLD.M with the given input arguments.
%
%      MLREGRESSIONHOLD('Property','Value',...) creates a new MLREGRESSIONHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MLRegressionHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MLRegressionHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MLRegressionHold

% Last Modified by GUIDE v2.5 30-May-2025 22:26:19

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MLRegressionHold_OpeningFcn, ...
                   'gui_OutputFcn',  @MLRegressionHold_OutputFcn, ...
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


% --- Executes just before MLRegressionHold is made visible.
function MLRegressionHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to MLRegressionHold (see VARARGIN)

% Choose default command line output for MLRegressionHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes MLRegressionHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = MLRegressionHold_OutputFcn(hObject, eventdata, handles) 
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
% hObject    handle to pushbutton1
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data

% --- Read inputs from GUI
filename = get(handles.edit1, 'string');        % Excel file name
outdim_str = get(handles.edit3, 'string');      % Output dimension
testRatio_str = get(handles.edit2, 'string');   % Test set proportion

% --- Convert strings to numeric
outdim = str2double(outdim_str);
testRatio = str2double(testRatio_str);

% --- Load and shuffle dataset
data = xlsread(filename);
num_samples = size(data, 1);
data = data(randperm(num_samples), :);  % Shuffle data

% --- Split into features and target
f_ = size(data, 2) - outdim;   % Number of features
X = data(:, 1:f_);
Y = data(:, f_ + 1:end);

% --- Train/test split
cv = cvpartition(size(Y, 1), 'HoldOut', testRatio);
Xtrain_raw = X(training(cv), :);
Ytrain = Y(training(cv), :);
Xtest_raw = X(test(cv), :);
Ytest = Y(test(cv), :);

% --- Standardize features using training set stats
mu = mean(Xtrain_raw);
sigma = std(Xtrain_raw);
sigma(sigma == 0) = 1;  % Prevent division by zero
Xtrain = (Xtrain_raw - mu) ./ sigma;
Xtest = (Xtest_raw - mu) ./ sigma;
Xall = (X - mu) ./ sigma;

% --- Train linear models for each output dimension
Mdl = cell(1, outdim);
for i = 1:outdim
    Mdl{i} = fitlm(Xtrain, Ytrain(:, i));
end

% --- Predict
Ytrain_pred = zeros(size(Ytrain));
Ytest_pred = zeros(size(Ytest));
Yall_pred = zeros(size(Y));
for i = 1:outdim
    Ytrain_pred(:, i) = predict(Mdl{i}, Xtrain);
    Ytest_pred(:, i) = predict(Mdl{i}, Xtest);
    Yall_pred(:, i) = predict(Mdl{i}, Xall);
end

% --- Define evaluation metrics
calc_metrics = @(Ytrue, Ypred) struct( ...
    'R2', 1 - sum((Ytrue - Ypred).^2) ./ sum((Ytrue - mean(Ytrue)).^2), ...
    'MSE', mean((Ytrue - Ypred).^2), ...
    'RMSE', sqrt(mean((Ytrue - Ypred).^2)), ...
    'MAPE', mean(abs((Ytrue - Ypred) ./ Ytrue), 'omitnan') * 100 ...
);

% --- Calculate metrics
train_metrics = calc_metrics(Ytrain, Ytrain_pred);
test_metrics = calc_metrics(Ytest, Ytest_pred);
all_metrics = calc_metrics(Y, Yall_pred);

% --- Display metrics for each output dimension
% --- Display metrics for each output dimension with new format
for i = 1:outdim
    fprintf('\n[Target %d - Training Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        i, train_metrics.R2(i), train_metrics.MSE(i), train_metrics.RMSE(i), train_metrics.MAPE(i));

    fprintf('[Target %d - Test Set]\nR²: %.4f | MSE: %.4f | RMSE: %.4f | MAPE: %.2f%%\n', ...
        i, test_metrics.R2(i), test_metrics.MSE(i), test_metrics.RMSE(i), test_metrics.MAPE(i));
end


% --- Plot actual vs predicted for the first output dimension
for i = 1:outdim
    figure;
    scatter(Ytest(:, i), Ytest_pred(:, i), 'filled');
    xlabel('Actual Value');
    ylabel('Predicted Value');
    title(sprintf('Linear Regression: Test Set Prediction (Output %d)', i));
    grid on;
    refline(1, 0);  % 45-degree reference line
end

% --- Export variables to MATLAB workspace
assignin('base', 'TrainedMLModel', Mdl);
assignin('base', 'Ytrain', Ytrain);
assignin('base', 'Ytrain_pred', Ytrain_pred);
assignin('base', 'Ytest', Ytest);
assignin('base', 'Ytest_pred', Ytest_pred);
