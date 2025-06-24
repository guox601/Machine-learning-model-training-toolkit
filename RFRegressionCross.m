function varargout = RFRegressionCross(varargin)
% RFREGRESSIONCROSS MATLAB code for RFRegressionCross.fig
%      RFREGRESSIONCROSS, by itself, creates a new RFREGRESSIONCROSS or raises the existing
%      singleton*.
%
%      H = RFREGRESSIONCROSS returns the handle to a new RFREGRESSIONCROSS or the handle to
%      the existing singleton*.
%
%      RFREGRESSIONCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RFREGRESSIONCROSS.M with the given input arguments.
%
%      RFREGRESSIONCROSS('Property','Value',...) creates a new RFREGRESSIONCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RFRegressionCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RFRegressionCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RFRegressionCross

% Last Modified by GUIDE v2.5 04-Jun-2025 21:54:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RFRegressionCross_OpeningFcn, ...
                   'gui_OutputFcn',  @RFRegressionCross_OutputFcn, ...
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


% --- Executes just before RFRegressionCross is made visible.
function RFRegressionCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RFRegressionCross (see VARARGIN)

% Choose default command line output for RFRegressionCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RFRegressionCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = RFRegressionCross_OutputFcn(hObject, eventdata, handles) 
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
k = str2double(get(handles.edit3, 'string'));
outdim = str2double(get(handles.edit4, 'string'));

res = xlsread(filename);
num_samples = size(res, 1);
res = res(randperm(num_samples), :);

f_ = size(res, 2) - outdim;
X = res(:, 1:f_);
Y = res(:, f_ + 1:end);

cv = cvpartition(num_samples, 'KFold', k);

% Initialize storage
R2s = zeros(k, outdim);
MSEs = zeros(k, outdim);
RMSEs = zeros(k, outdim);
MAPEs = zeros(k, outdim);

Ytest_all = cell(k, 1);
Ypred_all = cell(k, 1);

for i = 1:k
    trainIdx = training(cv, i);
    testIdx  = test(cv, i);
    
    Xtrain_raw = X(trainIdx, :);
    Ytrain = Y(trainIdx, :);
    Xtest_raw  = X(testIdx, :);
    Ytest = Y(testIdx, :);
    
    % Normalize features
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest  = (Xtest_raw - mu) ./ sigma;
    
    % Train one model per output dimension
    Ypred = zeros(size(Ytest));
    
    for j = 1:outdim
        rng(1);
        Mdl = fitrensemble(Xtrain, Ytrain(:, j), ...
            'Method', 'Bag', ...
            'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits'}, ...
            'HyperparameterOptimizationOptions', struct( ...
                'AcquisitionFunctionName', 'expected-improvement-plus', ...
                'Verbose', 0 ...
            ));

        Ypred(:, j) = predict(Mdl, Xtest);
        
        % Metrics
        yt = Ytest(:, j);
        yp = Ypred(:, j);
        R2s(i, j) = 1 - sum((yt - yp).^2) / sum((yt - mean(yt)).^2);
        MSEs(i, j) = mean((yt - yp).^2);
        RMSEs(i, j) = sqrt(MSEs(i, j));
        MAPEs(i, j) = mean(abs((yt - yp) ./ (yt + eps))) * 100;
    end

    % Store for export
    Ytest_all{i} = Ytest;
    Ypred_all{i} = Ypred;

    % Print fold metrics
    fprintf('\n[Fold %d Evaluation]\n', i);
    for j = 1:outdim
        fprintf('Output %d -> R²=%.4f  MSE=%.4f  RMSE=%.4f  MAPE=%.2f%%\n', ...
            j, R2s(i,j), MSEs(i,j), RMSEs(i,j), MAPEs(i,j));
    end
end

% Average metrics
fprintf('\n[Average Evaluation across %d folds]\n', k);
for j = 1:outdim
    fprintf('Output %d -> Avg R²=%.4f  Avg MSE=%.4f  Avg RMSE=%.4f  Avg MAPE=%.2f%%\n', ...
        j, mean(R2s(:,j)), mean(MSEs(:,j)), mean(RMSEs(:,j)), mean(MAPEs(:,j)));
end

% Export to base workspace
assignin('base', 'RF_Ytest_all', Ytest_all);
assignin('base', 'RF_Ypred_all', Ypred_all);
assignin('base', 'TrainedRFModel', Mdl);



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
