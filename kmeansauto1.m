function varargout = kmeansauto1(varargin)
% KMEANSAUTO1 MATLAB code for kmeansauto1.fig
%      KMEANSAUTO1, by itself, creates a new KMEANSAUTO1 or raises the existing
%      singleton*.
%
%      H = KMEANSAUTO1 returns the handle to a new KMEANSAUTO1 or the handle to
%      the existing singleton*.
%
%      KMEANSAUTO1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in KMEANSAUTO1.M with the given input arguments.
%
%      KMEANSAUTO1('Property','Value',...) creates a new KMEANSAUTO1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before kmeansauto1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to kmeansauto1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help kmeansauto1

% Last Modified by GUIDE v2.5 06-Jun-2025 21:19:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @kmeansauto1_OpeningFcn, ...
                   'gui_OutputFcn',  @kmeansauto1_OutputFcn, ...
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


% --- Executes just before kmeansauto1 is made visible.
function kmeansauto1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to kmeansauto1 (see VARARGIN)

% Choose default command line output for kmeansauto1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes kmeansauto1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = kmeansauto1_OutputFcn(hObject, eventdata, handles) 
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
% === 2. Preprocessing: Standardize the data (Z-score normalization) ===
mu = mean(data);
sigma = std(data);
sigma(sigma == 0) = 1;
X = (data - mu) ./ sigma;

% === 3. Set the number of clusters ===
k = 5;  % You can change this manually or use silhouette to optimize

% === 4. Perform K-means clustering ===
opts = statset('Display','final');
[idx, C, sumd] = kmeans(X, k, ...
    'Replicates', 10, ...
    'MaxIter', 300, ...
    'Distance', 'sqeuclidean', ...
    'Options', opts);

% === 5. Output number of samples in each cluster ===
fprintf('\nSamples per Cluster:\n');
tabulate(idx)

% === 6. Evaluation Metrics ===
inertia = sum(sumd);  % Inertia (within-cluster sum of squares)
silhouette_vals = silhouette(X, idx);  % Silhouette coefficients
avg_silhouette = mean(silhouette_vals);

% Calinski-Harabasz and Davies-Bouldin require "evalclusters"
ch_idx = evalclusters(X, idx, 'CalinskiHarabasz').CriterionValues;
db_idx = evalclusters(X, idx, 'DaviesBouldin').CriterionValues;

fprintf('\n=== Evaluation Metrics ===\n');
fprintf('Inertia (within-cluster SSE): %.4f\n', inertia);
fprintf('Average Silhouette Score: %.4f\n', avg_silhouette);
fprintf('Calinski-Harabasz Index: %.4f\n', ch_idx);
fprintf('Davies-Bouldin Index: %.4f\n', db_idx);

% === 7. Visualization (First 2 features only) ===
figure;
hold on;
colors = lines(k);
for i = 1:k
    cluster_points = X(idx == i, 1:2);
    scatter(cluster_points(:,1), cluster_points(:,2), 36, ...
        'MarkerEdgeColor', colors(i,:), ...
        'DisplayName', sprintf('Cluster %d', i));
end
scatter(C(:,1), C(:,2), 100, 'kx', 'LineWidth', 2, 'DisplayName', 'Centroids');
title(sprintf('K-means Clustering (k = %d)', k));
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');
grid on;

% === 8. Silhouette Plot ===
figure;
silhouette(X, idx);
title(sprintf('Silhouette Plot (k = %d)', k));

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
