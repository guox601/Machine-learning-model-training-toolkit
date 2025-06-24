function varargout = unsupervisedmodel(varargin)
% UNSUPERVISEDMODEL MATLAB code for unsupervisedmodel.fig
%      UNSUPERVISEDMODEL, by itself, creates a new UNSUPERVISEDMODEL or raises the existing
%      singleton*.
%
%      H = UNSUPERVISEDMODEL returns the handle to a new UNSUPERVISEDMODEL or the handle to
%      the existing singleton*.
%
%      UNSUPERVISEDMODEL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UNSUPERVISEDMODEL.M with the given input arguments.
%
%      UNSUPERVISEDMODEL('Property','Value',...) creates a new UNSUPERVISEDMODEL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before unsupervisedmodel_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to unsupervisedmodel_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help unsupervisedmodel

% Last Modified by GUIDE v2.5 06-Jun-2025 20:56:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @unsupervisedmodel_OpeningFcn, ...
                   'gui_OutputFcn',  @unsupervisedmodel_OutputFcn, ...
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


% --- Executes just before unsupervisedmodel is made visible.
function unsupervisedmodel_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to unsupervisedmodel (see VARARGIN)

% Choose default command line output for unsupervisedmodel
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes unsupervisedmodel wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = unsupervisedmodel_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
isRadio1Selected = get(handles.radiobutton1, 'Value');
isRadio3Selected = get(handles.radiobutton3, 'Value');
isRadio4Selected = get(handles.radiobutton4, 'Value');
isRadio5Selected = get(handles.radiobutton5, 'Value');

if isRadio1Selected
    try
        kmeansauto1;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open kmeansauto1.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio3Selected
    try
        Hera;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open Hera.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio4Selected
    try
        GMM;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open GMM.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio5Selected
    try
        PCA;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open PCA.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in radiobutton3.
function radiobutton3_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton3


% --- Executes on button press in radiobutton4.
function radiobutton4_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton4


% --- Executes on button press in radiobutton5.
function radiobutton5_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton5
