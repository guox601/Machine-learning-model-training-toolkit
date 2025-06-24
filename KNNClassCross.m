function varargout = KNNClassCross(varargin)
% KNNCLASSCROSS MATLAB code for KNNClassCross.fig
%      KNNCLASSCROSS, by itself, creates a new KNNCLASSCROSS or raises the existing
%      singleton*.
%
%      H = KNNCLASSCROSS returns the handle to a new KNNCLASSCROSS or the handle to
%      the existing singleton*.
%
%      KNNCLASSCROSS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in KNNCLASSCROSS.M with the given input arguments.
%
%      KNNCLASSCROSS('Property','Value',...) creates a new KNNCLASSCROSS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before KNNClassCross_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to KNNClassCross_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help KNNClassCross

% Last Modified by GUIDE v2.5 06-Jun-2025 15:51:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @KNNClassCross_OpeningFcn, ...
                   'gui_OutputFcn',  @KNNClassCross_OutputFcn, ...
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


% --- Executes just before KNNClassCross is made visible.
function KNNClassCross_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to KNNClassCross (see VARARGIN)

% Choose default command line output for KNNClassCross
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes KNNClassCross wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = KNNClassCross_OutputFcn(hObject, eventdata, handles) 
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


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
 % === 读取输入参数 ===
    filename = get(handles.edit1, 'string');            % Excel 文件路径
   kfold = str2double(get(handles.edit2, 'string'));   % 交叉验证折数
    outdim = 1;  % 输出维度数

    % === 读取并打乱数据 ===
    res = xlsread(filename);
    num_samples = size(res, 1);
    res = res(randperm(num_samples), :);  % 打乱样本顺序

    % === 拆分特征和标签 ===
    f_ = size(res, 2) - outdim;
    X = res(:, 1:f_);
    Y = res(:, f_+1:end);
% === Encode labels as integers ===
Y = grp2idx(Y);  % 1, 2, 3... for classification
classes = unique(Y);
num_classes = length(classes);

% === Normalize features ===
mu = mean(X);
sigma = std(X);
sigma(sigma == 0) = 1;
Xnorm = (X - mu) ./ sigma;

% === k-Fold Cross-Validation ===

k_candidates = 1:20;
num_ks = length(k_candidates);
all_metrics = zeros(num_ks, 4); % Accuracy, Precision, Recall, F1

cv = cvpartition(size(Y,1), 'KFold', kfold);

fprintf('\n%-6s %-10s %-10s %-10s %-10s\n', 'k', 'Accuracy', 'Precision', 'Recall', 'F1-score');
fprintf('%s\n', repmat('-', 1, 50));

for k_idx = 1:num_ks
    k_try = k_candidates(k_idx);
    acc_all = zeros(kfold,1);
    prec_all = zeros(kfold,1);
    rec_all = zeros(kfold,1);
    f1_all = zeros(kfold,1);

    fprintf('\nEvaluating k = %d\n', k_try);
    for fold = 1:kfold
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);

        Xtrain = Xnorm(trainIdx, :);
        Ytrain = Y(trainIdx);
        Xtest  = Xnorm(testIdx, :);
        Ytest  = Y(testIdx);

        Ypred = knn_class_predict(Xtrain, Ytrain, Xtest, k_try);

        % Calculate metrics
        [acc, prec, rec, f1] = classification_metrics(Ytest, Ypred);
        acc_all(fold) = acc;
        prec_all(fold) = prec;
        rec_all(fold) = rec;
        f1_all(fold) = f1;

        % Print metrics for this fold
        fprintf('  Fold %d | Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
                fold, acc, prec, rec, f1);
    end

    all_metrics(k_idx, :) = [mean(acc_all), mean(prec_all), mean(rec_all), mean(f1_all)];
    fprintf('Mean for k=%d: Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
        k_try, all_metrics(k_idx,1), all_metrics(k_idx,2), all_metrics(k_idx,3), all_metrics(k_idx,4));
end

% === Find best k (based on highest F1) ===
[~, best_idx] = max(all_metrics(:,4));
best_k = k_candidates(best_idx);

fprintf('\n[Best k = %d]\n', best_k);
fprintf('Average Accuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f\n', ...
    all_metrics(best_idx,1), all_metrics(best_idx,2), ...
    all_metrics(best_idx,3), all_metrics(best_idx,4));

% === Plot k vs F1-score ===
figure;
plot(k_candidates, all_metrics(:,4), '-o', 'LineWidth', 2);
xlabel('k'); ylabel('Average F1-score (5-Fold)');
title('KNN Cross-Validation: k vs F1-score');
grid on;

% === Compute and visualize average confusion matrix for best k ===
avg_cm = zeros(num_classes);
cv = cvpartition(size(Y,1), 'KFold', kfold);  % Recreate for consistency

for fold = 1:kfold
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    Xtrain = Xnorm(trainIdx, :);
    Ytrain = Y(trainIdx);
    Xtest  = Xnorm(testIdx, :);
    Ytest  = Y(testIdx);

    Ypred = knn_class_predict(Xtrain, Ytrain, Xtest, best_k);

    cm = confusionmat(Ytest, Ypred, 'Order', classes);
    avg_cm = avg_cm + cm;
end

avg_cm = avg_cm / kfold;

% Visualize average confusion matrix (rounded)
figure('Name', 'Average Confusion Matrix');
confusionchart(round(avg_cm), string(classes), 'Title', ...
    sprintf('Average Confusion Matrix (Best k = %d)', best_k));
final_model.Xtrain = Xnorm;
final_model.Ytrain = Y;
final_model.k = best_k;
final_model.mu = mu;
final_model.sigma = sigma;
assignin('base', 'TrainedKNNModel', final_model);
%% --- Function: KNN classification prediction ---
function Ypred = knn_class_predict(Xtrain, Ytrain, Xtest, k)
    num_test = size(Xtest, 1);
    Ypred = zeros(num_test, 1);
    for i = 1:num_test
        dists = sqrt(sum((Xtrain - Xtest(i, :)).^2, 2));
        [~, idx] = sort(dists);
        nearest_idx = idx(1:k);
        nearest_labels = Ytrain(nearest_idx);
        Ypred(i) = mode(nearest_labels);
    end


%% --- Function: Classification metrics ---
function [accuracy, precision, recall, f1] = classification_metrics(Ytrue, Ypred)
    cm = confusionmat(Ytrue, Ypred);
    TP = diag(cm);
    FP = sum(cm,1)' - TP;
    FN = sum(cm,2) - TP;

    precision = mean(TP ./ (TP + FP + eps));
    recall = mean(TP ./ (TP + FN + eps));
    f1 = 2 * precision * recall / (precision + recall + eps);
    accuracy = sum(TP) / sum(cm(:));



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
