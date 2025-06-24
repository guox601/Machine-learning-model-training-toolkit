function varargout = LCClassHold(varargin)
% LCCLASSHOLD MATLAB code for LCClassHold.fig
%      LCCLASSHOLD, by itself, creates a new LCCLASSHOLD or raises the existing
%      singleton*.
%
%      H = LCCLASSHOLD returns the handle to a new LCCLASSHOLD or the handle to
%      the existing singleton*.
%
%      LCCLASSHOLD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LCCLASSHOLD.M with the given input arguments.
%
%      LCCLASSHOLD('Property','Value',...) creates a new LCCLASSHOLD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LCClassHold_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LCClassHold_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LCClassHold

% Last Modified by GUIDE v2.5 03-Jun-2025 22:37:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LCClassHold_OpeningFcn, ...
                   'gui_OutputFcn',  @LCClassHold_OutputFcn, ...
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


% --- Executes just before LCClassHold is made visible.
function LCClassHold_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to LCClassHold (see VARARGIN)

% Choose default command line output for LCClassHold
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes LCClassHold wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = LCClassHold_OutputFcn(hObject, eventdata, handles) 
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
    % Get file name from edit1
    filename = get(handles.edit1, 'string');
    outdim = 1; % Adjust based on the number of output columns
    proportion = str2double(get(handles.edit2, 'string')); % Train-test split ratio

    % Read data from Excel file
    data = xlsread(filename);
    num_samples = size(data, 1);
    num_features = size(data, 2) - outdim;
    X = data(:, 1:num_features);
    Y = data(:, num_features + 1:end);
    Y = categorical(Y); % Convert labels to categorical for classification

    % Create hold-out partition for training and testing
    cv = cvpartition(size(Y,1), 'HoldOut', proportion);
    Xtrain_raw = X(training(cv), :);
    Ytrain = Y(training(cv), :);
    Xtest_raw = X(test(cv), :);
    Ytest = Y(test(cv), :);

    % Normalize features using z-score normalization (based on training data)
    mu = mean(Xtrain_raw);
    sigma = std(Xtrain_raw);
    sigma(sigma == 0) = 1; % Prevent division by zero
    Xtrain = (Xtrain_raw - mu) ./ sigma;
    Xtest = (Xtest_raw - mu) ./ sigma;
    Xall = (X - mu) ./ sigma;

    % Train Linear Discriminant Analysis (LDA) classifier
    Mdl = fitcdiscr(Xtrain, Ytrain);

    % Predict labels on training, testing, and all data
    Ytrain_pred = predict(Mdl, Xtrain);
    Ytest_pred = predict(Mdl, Xtest);
    Yall_pred = predict(Mdl, Xall);

    % Save the trained model and predictions to the base workspace
    assignin('base', 'TrainedMCModel', Mdl);
    assignin('base', 'Ytrain_pred', Ytrain_pred);
    assignin('base', 'Ytest_pred', Ytest_pred);
    assignin('base', 'Yall_pred', Yall_pred);

    % Define a helper function to calculate accuracy
    calc_acc = @(Ytrue, Ypred) mean(Ytrue == Ypred);
    % Calculate metrics for training set
    train_acc = calc_acc(Ytrain, Ytrain_pred);
    [train_precision, train_recall, train_f1] = calc_prf(Ytrain, Ytrain_pred);

    % Calculate metrics for test set
    test_acc = calc_acc(Ytest, Ytest_pred);
    [test_precision, test_recall, test_f1] = calc_prf(Ytest, Ytest_pred);

    % Calculate metrics for all data
    all_acc = calc_acc(Y, Yall_pred);
    [all_precision, all_recall, all_f1] = calc_prf(Y, Yall_pred);

    % Display the evaluation results in Command Window
    fprintf('\n[Training Set Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
        train_acc, train_precision, train_recall, train_f1);
    fprintf('[Test Set Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
        test_acc, test_precision, test_recall, test_f1);
    fprintf('[All Data Evaluation]\nAccuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f\n', ...
        all_acc, all_precision, all_recall, all_f1);

    % Plot confusion matrix for test data
    figure;
    confusionchart(Ytest, Ytest_pred);
    title('Confusion Matrix (Test Set)');

    % Helper function to calculate macro-averaged precision, recall, and F1-score
    function [precision, recall, f1] = calc_prf(y_true, y_pred)
        classes = categories(y_true);
        nClass = numel(classes);
        precision_c = zeros(nClass,1);
        recall_c = zeros(nClass,1);
        f1_c = zeros(nClass,1);

        for i = 1:nClass
            c = classes{i};
            TP = sum(y_pred == c & y_true == c);
            FP = sum(y_pred == c & y_true ~= c);
            FN = sum(y_pred ~= c & y_true == c);

            if TP + FP == 0
                precision_c(i) = 0;
            else
                precision_c(i) = TP / (TP + FP);
            end

            if TP + FN == 0
                recall_c(i) = 0;
            else
                recall_c(i) = TP / (TP + FN);
            end

            if precision_c(i) + recall_c(i) == 0
                f1_c(i) = 0;
            else
                f1_c(i) = 2 * (precision_c(i) * recall_c(i)) / (precision_c(i) + recall_c(i));
            end
        end

        % Macro-average metrics
        precision = mean(precision_c);
        recall = mean(recall_c);
        f1 = mean(f1_c);
    



% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
