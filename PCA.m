function varargout = PCA(varargin)
% PCA MATLAB code for PCA.fig
%      PCA, by itself, creates a new PCA or raises the existing
%      singleton*.
%
%      H = PCA returns the handle to a new PCA or the handle to
%      the existing singleton*.
%
%      PCA('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PCA.M with the given input arguments.
%
%      PCA('Property','Value',...) creates a new PCA or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PCA_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PCA_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help PCA

% Last Modified by GUIDE v2.5 06-Jun-2025 21:27:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PCA_OpeningFcn, ...
                   'gui_OutputFcn',  @PCA_OutputFcn, ...
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


% --- Executes just before PCA is made visible.
function PCA_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to PCA (see VARARGIN)

% Choose default command line output for PCA
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes PCA wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = PCA_OutputFcn(hObject, eventdata, handles) 
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

% === 2. Standardize data (Z-score normalization) ===
mu = mean(data);
sigma = std(data);
sigma(sigma == 0) = 1;  % Prevent division by zero
X = (data - mu) ./ sigma;

% === 3. Perform PCA ===
[coeff, score, ~, ~, explained] = pca(X);

% === 4. Select number of components to retain (>=90% cumulative variance) ===
cumVar = cumsum(explained);
N = find(cumVar >= 90, 1);

% === 5. Get reduced data ===
X_reduced = score(:, 1:N);
assignin('base', 'X_reduced', X_reduced);

% === 6. Reconstruct data from reduced components ===
X_reconstructed_std = X_reduced * coeff(:, 1:N)'; % in standardized space

% === 7. Inverse standardization to original scale ===
X_reconstructed = X_reconstructed_std .* sigma + mu;

% === 8. Calculate reconstruction error (MSE) ===
mse_reconstruction = mean(mean((data - X_reconstructed).^2));
fprintf('Reconstruction MSE: %.6f\n', mse_reconstruction);

% === 9. Plot cumulative explained variance ===
figure;
plot(cumVar, '-o', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA Cumulative Explained Variance');
grid on;

% === 10. Plot original vs reconstructed for first feature (example) ===
figure;
plot(data(:,1), 'b-', 'DisplayName', 'Original Feature 1');
hold on;
plot(X_reconstructed(:,1), 'r--', 'DisplayName', 'Reconstructed Feature 1');
xlabel('Sample Index');
ylabel('Value');
title('Original vs Reconstructed Data (Feature 1)');
legend('Location', 'best');
grid on;

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
