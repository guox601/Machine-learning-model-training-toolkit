function varargout = SVRCross(varargin)
% SVRCROSS MATLAB code for SVRCross.fig
%      SVRCROSS, by itself, creates a new SVRCROSS or raises the existing
%      singleton*.
%
%      H = SVRCROSS returns the handle to a new SVRCROSS or the handle to
%      the existing singleton*.
%
%      SVRCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SVRCROSS.M with the given input arguments.
%
%      SVRCROSS('Property','Value',...) creates a new SVRCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SVRCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SVRCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SVRCross

% Last Modified by GUIDE v2.5 05-Jun-2025 10:11:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SVRCross_OpeningFcn, ...
                   'gui_OutputFcn',  @SVRCross_OutputFcn, ...
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


% --- Executes just before SVRCross is made visible.
function SVRCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SVRCross (see VARARGIN)

% Choose default command line output for SVRCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SVRCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SVRCross_OutputFcn(hObject, eventdata, handles) 
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
K = str2double(get(handles.edit2, 'string')); % K-fold
outdim = str2double(get(handles.edit3, 'string')); % output dimensions

res = xlsread(filename);
num_samples = size(res, 1);
res = res(randperm(num_samples), :);  % shuffle

f_ = size(res, 2) - outdim;
X = res(:, 1:f_);
Y = res(:, f_ + 1:end);

cv = cvpartition(num_samples, 'KFold', K);

% 存储每一折每个输出维度的指标
R2s = zeros(K, outdim);
MSEs = zeros(K, outdim);
RMSEs = zeros(K, outdim);
MAPEs = zeros(K, outdim);

Ytrue_all = cell(K, 1);
Ypred_all = cell(K, 1);

for fold = 1:K
    fprintf('\n=== Fold %d/%d ===\n', fold, K);
    
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    
    Xtrain_raw = X(trainIdx, :);
    Ytrain = Y(trainIdx, :);
    Xtest_raw = X(testIdx, :);
    Ytest = Y(testIdx, :);
    
    % 标准化
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1;
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    
    Ypred = zeros(size(Ytest));
    
    for j = 1:outdim
        % 超参数优化配置
        opts = struct( ...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'ShowPlots', false, ...
            'Verbose', 0 ...
        );

        rng(1);
        Mdl = fitrsvm(Xtrain, Ytrain(:, j), ...
            'KernelFunction', 'gaussian', ...
            'Standardize', false, ...
            'IterationLimit', 1000, ...
            'OptimizeHyperparameters', 'auto', ...
            'HyperparameterOptimizationOptions', opts);
        
        Ypred(:, j) = predict(Mdl, Xtest);
        
        yt = Ytest(:, j);
        yp = Ypred(:, j);
        R2s(fold, j) = 1 - sum((yt - yp).^2) / sum((yt - mean(yt)).^2);
        MSEs(fold, j) = mean((yt - yp).^2);
        RMSEs(fold, j) = sqrt(MSEs(fold, j));
        MAPEs(fold, j) = mean(abs((yt - yp) ./ (yt + eps))) * 100;
        
        fprintf('Output %d -> R² = %.4f, MSE = %.4f, RMSE = %.4f, MAPE = %.2f%%\n', ...
            j, R2s(fold, j), MSEs(fold, j), RMSEs(fold, j), MAPEs(fold, j));
    end
    
    Ytrue_all{fold} = Ytest;
    Ypred_all{fold} = Ypred;
end

% 平均指标
fprintf('\n=== Average Metrics Across %d Folds ===\n', K);
for j = 1:outdim
    fprintf('Output %d -> Avg R² = %.4f, Avg MSE = %.4f, Avg RMSE = %.4f, Avg MAPE = %.2f%%\n', ...
        j, mean(R2s(:, j)), mean(MSEs(:, j)), mean(RMSEs(:, j)), mean(MAPEs(:, j)));
end

% 保存结果到 base 工作区
assignin('base', 'SVR_Ytrue_all', Ytrue_all);
assignin('base', 'SVR_Ypred_all', Ypred_all);
assignin('base', 'TrainedSVRModel', Mdl);




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
