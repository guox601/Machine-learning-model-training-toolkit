function varargout = supervisedmodel(varargin)
% SUPERVISEDMODEL MATLAB code for supervisedmodel.fig
%      SUPERVISEDMODEL, by itself, creates a new SUPERVISEDMODEL or raises the existing
%      singleton*.
%
%      H = SUPERVISEDMODEL returns the handle to a new SUPERVISEDMODEL or the handle to
%      the existing singleton*.
%
%      SUPERVISEDMODEL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SUPERVISEDMODEL.M with the given input arguments.
%
%      SUPERVISEDMODEL('Property','Value',...) creates a new SUPERVISEDMODEL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before supervisedmodel_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to supervisedmodel_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help supervisedmodel

% Last Modified by GUIDE v2.5 29-May-2025 20:03:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @supervisedmodel_OpeningFcn, ...
                   'gui_OutputFcn',  @supervisedmodel_OutputFcn, ...
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


% --- Executes just before supervisedmodel is made visible.
function supervisedmodel_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to supervisedmodel (see VARARGIN)

% Choose default command line output for supervisedmodel
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes supervisedmodel wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = supervisedmodel_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in radiobutton1.
%Random Forest Regression Hold-out
function radiobutton1_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1


% --- Executes on button press in radiobutton2.
function radiobutton2_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton2


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


% --- Executes on button press in radiobutton6.
function radiobutton6_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton6


% --- Executes on button press in radiobutton8.
function radiobutton8_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton8


% --- Executes on button press in radiobutton9.
function radiobutton9_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton9


% --- Executes on button press in radiobutton10.
function radiobutton10_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton10


% --- Executes on button press in radiobutton11.
function radiobutton11_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton11


% --- Executes on button press in radiobutton12.
function radiobutton12_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton12


% --- Executes on button press in radiobutton13.
function radiobutton13_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton13


% --- Executes on button press in radiobutton14.
function radiobutton14_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton14


% --- Executes on button press in radiobutton15.
function radiobutton15_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton15


% --- Executes on button press in radiobutton16.
function radiobutton16_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton16


% --- Executes on button press in radiobutton17.
function radiobutton17_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton17


% --- Executes on button press in radiobutton18.
function radiobutton18_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton18


% --- Executes on button press in radiobutton20.
function radiobutton20_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton20


% --- Executes on button press in radiobutton21.
function radiobutton21_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton21


% --- Executes on button press in radiobutton22.
function radiobutton22_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton22


% --- Executes on button press in radiobutton23.
function radiobutton23_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton23


% --- Executes on button press in radiobutton24.
function radiobutton24_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton24


% --- Executes on button press in radiobutton26.
function radiobutton26_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton26


% --- Executes on button press in radiobutton28.
function radiobutton28_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton28


% --- Executes on button press in radiobutton29.
function radiobutton29_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton29


% --- Executes on button press in radiobutton30.
function radiobutton30_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton30


% --- Executes on button press in radiobutton31.
function radiobutton31_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton31


% --- Executes on button press in radiobutton32.
function radiobutton32_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton32


% --- Executes on button press in radiobutton33.
function radiobutton33_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton33


% --- Executes on button press in radiobutton34.
function radiobutton34_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton34


% --- Executes on button press in radiobutton35.
function radiobutton35_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton35


% --- Executes on button press in radiobutton36.
function radiobutton36_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton36


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
isRadio1Selected = get(handles.radiobutton1, 'Value');
isRadio2Selected = get(handles.radiobutton2, 'Value');
isRadio3Selected = get(handles.radiobutton3, 'Value');
isRadio4Selected = get(handles.radiobutton4, 'Value');
isRadio5Selected = get(handles.radiobutton5, 'Value');
isRadio6Selected = get(handles.radiobutton6, 'Value');
isRadio8Selected = get(handles.radiobutton8, 'Value');
isRadio9Selected = get(handles.radiobutton9, 'Value');
isRadio10Selected = get(handles.radiobutton10, 'Value');
isRadio11Selected = get(handles.radiobutton11, 'Value');
isRadio12Selected = get(handles.radiobutton12, 'Value');
isRadio13Selected = get(handles.radiobutton13, 'Value');
isRadio26Selected = get(handles.radiobutton26, 'Value');
isRadio28Selected = get(handles.radiobutton28, 'Value');
isRadio29Selected = get(handles.radiobutton29, 'Value');
isRadio30Selected = get(handles.radiobutton30, 'Value');
isRadio31Selected = get(handles.radiobutton31, 'Value');
isRadio32Selected = get(handles.radiobutton32, 'Value');
isRadio33Selected = get(handles.radiobutton33, 'Value');
isRadio34Selected = get(handles.radiobutton34, 'Value');
isRadio35Selected = get(handles.radiobutton35, 'Value');
isRadio36Selected = get(handles.radiobutton36, 'Value');

if isRadio1Selected
    try
        RFRegressionholdout;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open RFRegressionholdout.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio2Selected
    try
        XGBootsRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open XGBootsRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio3Selected
    try
        SVRHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open SVRHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio4Selected
    try
        DTRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open DTRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio5Selected
    try
        MLRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open MLRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio6Selected
    try
        KNNRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open KNNRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio8Selected
    try
        BpANNRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BpANNRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio9Selected
    try
        LSTMRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open LSTMRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio10Selected
    try
        BiLSTMRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BiLSTMRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio11Selected
    try
        CNNRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open CNNRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio12Selected
    try
        GRURegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        disp(ME.message);
    end
end

if isRadio13Selected
    try
        GPRRegressionHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open GPRRegressionHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio26Selected
    try
        RFClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open RFClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio28Selected
    try
        SVMHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open SVMHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio29Selected
    try
      DTClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open DTClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio30Selected
    try
      MCClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open MCClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio31Selected
    try
      KNNClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open KNNClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio32Selected
    try
      BpANNClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BpANNClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio33Selected
    try
      LSTMClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open LSTMClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio34Selected
    try
      BiLSTMClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BiLSTMClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio35Selected
    try
      CNNClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open CNNClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio36Selected
    try
      GRUClassHold;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open GRUClassHold.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
isRadio1Selected = get(handles.radiobutton1, 'Value');
isRadio2Selected = get(handles.radiobutton2, 'Value');
isRadio3Selected = get(handles.radiobutton3, 'Value');
isRadio4Selected = get(handles.radiobutton4, 'Value');
isRadio5Selected = get(handles.radiobutton5, 'Value');
isRadio6Selected = get(handles.radiobutton6, 'Value');
isRadio8Selected = get(handles.radiobutton8, 'Value');
isRadio9Selected = get(handles.radiobutton9, 'Value');
isRadio10Selected = get(handles.radiobutton10, 'Value');
isRadio11Selected = get(handles.radiobutton11, 'Value');
isRadio12Selected = get(handles.radiobutton12, 'Value');
isRadio13Selected = get(handles.radiobutton13, 'Value');
isRadio26Selected = get(handles.radiobutton26, 'Value');
isRadio28Selected = get(handles.radiobutton28, 'Value');
isRadio29Selected = get(handles.radiobutton29, 'Value');
isRadio30Selected = get(handles.radiobutton30, 'Value');
isRadio31Selected = get(handles.radiobutton31, 'Value');
isRadio32Selected = get(handles.radiobutton32, 'Value');
isRadio33Selected = get(handles.radiobutton33, 'Value');
isRadio34Selected = get(handles.radiobutton34, 'Value');
isRadio35Selected = get(handles.radiobutton35, 'Value');
isRadio36Selected = get(handles.radiobutton36, 'Value');

if isRadio1Selected
    try
        RFRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open RFRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio2Selected
    try
        XGBootsRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open XGBootsRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio3Selected
    try
        SVRCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open SVRCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio4Selected
    try
        DTRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open DTRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio5Selected
    try
        MLRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open MLRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio6Selected
    try
        KNNRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open KNNRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio8Selected
    try
        BpANNRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BpANNRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio9Selected
    try
        LSTMRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open LSTMRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio10Selected
    try
        BiLSTMRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BiLSTMRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio11Selected
    try
        CNNRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open CNNRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio12Selected
    try
        GRURegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        disp(ME.message);
    end
end

if isRadio13Selected
    try
        GPRRegressionCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open GPRRegressionCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio26Selected
    try
        RFClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open RFClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio28Selected
    try
        SVMCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open SVMCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio29Selected
    try
      DTClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open DTClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio30Selected
    try
      MCClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open MCClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio31Selected
    try
      KNNClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open KNNClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio32Selected
    try
      BpANNClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BpANNClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio33Selected
    try
      LSTMClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open LSTMClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio34Selected
    try
      BiLSTMClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open BiLSTMClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio35Selected
    try
      CNNClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open CNNClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end

if isRadio36Selected
    try
      GRUClassCross;  % Assumes supervisedmodel.m/.fig exists
    catch ME
        errordlg('Unable to open GRUClassCross.fig. Please ensure the file exists.', 'File Error');
        disp(ME.message);
    end
end
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
