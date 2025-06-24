function varargout = GMM(varargin)
% GMM MATLAB code for GMM.fig
%      GMM, by itself, creates a new GMM or raises the existing
%      singleton*.
%
%      H = GMM returns the handle to a new GMM or the handle to
%      the existing singleton*.
%
%      GMM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GMM.M with the given input arguments.
%
%      GMM('Property','Value',...) creates a new GMM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GMM_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GMM_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GMM

% Last Modified by GUIDE v2.5 06-Jun-2025 21:24:43

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GMM_OpeningFcn, ...
                   'gui_OutputFcn',  @GMM_OutputFcn, ...
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


% --- Executes just before GMM is made visible.
function GMM_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GMM (see VARARGIN)

% Choose default command line output for GMM
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GMM wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GMM_OutputFcn(hObject, eventdata, handles) 
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
data = xlsread(filename);

% === 2. Z-score Standardization ===
mu = mean(data);
sigma = std(data);
sigma(sigma == 0) = 1;
X = (data - mu) ./ sigma;

% === 3. Automatically select best k using silhouette score ===
maxK = 10;
silhouette_avg = zeros(maxK - 1, 1);
gmModels = cell(maxK, 1);  % store GMMs
idxList = cell(maxK, 1);   % store cluster labels

fprintf('Evaluating silhouette scores for k = 2 to %d...\n', maxK);
for k = 2:maxK
    gm = fitgmdist(X, k, 'Replicates', 5, 'RegularizationValue', 1e-5, ...
        'Options', statset('MaxIter', 500, 'Display','off'));
    idx_tmp = cluster(gm, X);
    s = silhouette(X, idx_tmp);
    silhouette_avg(k - 1) = mean(s);
    gmModels{k} = gm;
    idxList{k} = idx_tmp;
end

% === 4. Determine best k ===
[~, best_k_idx] = max(silhouette_avg);
best_k = best_k_idx + 1;
fprintf('Best number of clusters based on silhouette: %d\n', best_k);
gmBest = gmModels{best_k};
idx = idxList{best_k};

% === 5. Cluster size output ===
fprintf('\nSamples per Cluster:\n');
tabulate(idx)

% === 6. Evaluation Metrics ===
% Log-likelihood (similar to inertia for GMM)
negLogLikelihood = -gmBest.NegativeLogLikelihood;

% Silhouette score
silhouette_vals = silhouette(X, idx);
avg_silhouette = mean(silhouette_vals);

% CH 和 DB 指数
ch = evalclusters(X, @(X,k)cluster(fitgmdist(X,k,'RegularizationValue',1e-5,'Replicates',3), X), 'CalinskiHarabasz', 'klist', best_k);
db = evalclusters(X, @(X,k)cluster(fitgmdist(X,k,'RegularizationValue',1e-5,'Replicates',3), X), 'DaviesBouldin', 'klist', best_k);

fprintf('\n=== Evaluation Metrics ===\n');
fprintf('Negative Log-Likelihood (lower is better): %.4f\n', negLogLikelihood);
fprintf('Average Silhouette Score: %.4f\n', avg_silhouette);
fprintf('Calinski-Harabasz Index: %.4f\n', ch.CriterionValues);
fprintf('Davies-Bouldin Index: %.4f\n', db.CriterionValues);

% === 7. Scatter Plot (first 2 features) ===
figure;
hold on;
colors = lines(best_k);
for i = 1:best_k
    scatter(X(idx==i,1), X(idx==i,2), 36, colors(i,:), 'filled', 'DisplayName', sprintf('Cluster %d', i));
end
title(sprintf('GMM Clustering (k = %d)', best_k));
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');
grid on;

% === 8. Silhouette Plot ===
figure;
silhouette(X, idx);
title(sprintf('Silhouette Plot for GMM (k = %d)', best_k));

% === 9. Silhouette Scores by k ===
figure;
plot(2:maxK, silhouette_avg, '-o', 'LineWidth', 2);
xlabel('Number of Clusters (k)');
ylabel('Average Silhouette Score');
title('Silhouette Score vs. Number of Clusters (GMM)');
grid on;

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
